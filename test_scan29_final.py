"""Balanced fusion pipeline for DTU scan29 using MT-MVSNet."""

import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import open3d as o3d

from mtmvsnet_model import MTMVSNet
from eval_dtu import evaluate_point_cloud

# -------------------------------------------------------------------------
# Global settings (you can tweak these if you want different density/quality)
# -------------------------------------------------------------------------
TARGET_HEIGHT = 512
TARGET_WIDTH = 640

# Photometric + geometric filtering (tightened per CasMVSNet/CVP-MVSNet best practices)
CONFIDENCE_THRESHOLD = 0.03      # just above uniform baseline (1/48=0.021) for stage 0
GEOMETRIC_REL_ERROR = 0.05      # 5%: tight enough for accuracy, lenient enough for coarse stage 0
MIN_CONSISTENT_VIEWS = 2        # require 2 views to agree — filters noise without being too strict

# Voxel size for downsampling (adjusted automatically per unit system)
VOXEL_SIZE = 1.5  # sub-mm accuracy requires finer voxels (was 1.0)

# Depth scaling: network predictions match the camera-file depth units.
DEPTH_SCALE_TO_MM = 1.0

MAX_VIEWS = 49
DEPTH_MAX = 935.0  # default mm; overridden for meter-based scans
LOG_DIR = os.path.join("outputs", "logs")


def median_filter_depth(depth_map: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median filter to depth map to remove spike noise while preserving edges."""
    from scipy.ndimage import median_filter
    valid = depth_map > 0
    filtered = median_filter(depth_map, size=kernel_size)
    # Only apply where valid, preserve zeros
    result = np.where(valid, filtered, 0.0)
    return result


def bilateral_filter_depth(depth_map: np.ndarray, sigma_spatial: int = 7, sigma_depth: float = 2.0) -> np.ndarray:
    """Apply bilateral filter to depth map — smooths while preserving edges."""
    valid = depth_map > 0
    if not np.any(valid):
        return depth_map
    # Normalize depth to 0-255 range for cv2.bilateralFilter
    d_min = depth_map[valid].min()
    d_max = depth_map[valid].max()
    if d_max - d_min < 1e-6:
        return depth_map
    normalized = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.float32)
    normalized[~valid] = 0
    filtered = cv2.bilateralFilter(normalized, d=-1, sigmaColor=sigma_depth * 255 / (d_max - d_min),
                                    sigmaSpace=sigma_spatial)
    # Map back to original depth range
    result = filtered / 255.0 * (d_max - d_min) + d_min
    result[~valid] = 0.0
    return result


def statistical_outlier_removal(
    points: np.ndarray, colors: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0
) -> tuple:
    """Remove statistical outliers from point cloud (like Open3D SOR)."""
    if len(points) < nb_neighbors + 1:
        return points, colors
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=nb_neighbors + 1)
    # Exclude self-distance (column 0)
    mean_dists = dists[:, 1:].mean(axis=1)
    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + std_ratio * global_std
    mask = mean_dists < threshold
    return points[mask], colors[mask]


def detect_depth_units(scan_path: str) -> str:
    """Auto-detect whether a scan uses meters or millimeters.

    Reads the first camera file and checks depth_min.
    If depth_min < 10 the data is almost certainly in meters.
    Returns 'meters' or 'millimeters'.
    """
    cams_dir = os.path.join(scan_path, "cams")
    cam_files = sorted(f for f in os.listdir(cams_dir) if f.endswith("_cam.txt"))
    if not cam_files:
        return "millimeters"
    with open(os.path.join(cams_dir, cam_files[0])) as f:
        lines = f.readlines()
    depth_min = float(lines[11].split()[0])
    return "meters" if depth_min < 10 else "millimeters"


def apply_unit_settings(units: str) -> None:
    """Adjust global constants based on the detected unit system."""
    global VOXEL_SIZE, DEPTH_MAX, DEPTH_SCALE_TO_MM
    if units == "meters":
        VOXEL_SIZE = 0.001   # 1 mm in meters
        DEPTH_MAX = 20.0     # 20 m max depth (generous for outdoor)
        DEPTH_SCALE_TO_MM = 1.0  # keep native units, no conversion
        print("[Units] Detected METERS – VOXEL_SIZE=0.001, DEPTH_MAX=20.0")
    else:
        VOXEL_SIZE = 1.0
        DEPTH_MAX = 935.0
        DEPTH_SCALE_TO_MM = 1.0
        print("[Units] Detected MILLIMETERS – VOXEL_SIZE=1.0, DEPTH_MAX=935.0")


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
class ViewCache:
    """Lazy loader for per-view assets (images + cameras)."""

    def __init__(self, scan_path: str):
        self.scan_path = scan_path
        self.cache: Dict[int, Dict[str, np.ndarray]] = {}

    def get(self, view_idx: int) -> Dict[str, np.ndarray]:
        if view_idx in self.cache:
            return self.cache[view_idx]

        img_path = os.path.join(self.scan_path, "images", f"{view_idx:08d}.jpg")
        cam_path = os.path.join(self.scan_path, "cams", f"{view_idx:08d}_cam.txt")

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Missing image: {img_path}")

        # Read actual image dimensions before resize (handles any resolution)
        orig_h, orig_w = image_bgr.shape[:2]
        image_bgr = cv2.resize(image_bgr, (TARGET_WIDTH, TARGET_HEIGHT))
        image_tensor = torch.from_numpy(image_bgr).permute(2, 0, 1).float() / 255.0
        color_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with open(cam_path) as f:
            lines = f.readlines()

        # DTU cam.txt: extrinsic on lines 1–4, intrinsic on 7–9, depth line at 11
        # IMPORTANT: DTU extrinsic is world→camera (w2c).
        extrinsic_w2c = np.array(
            [list(map(float, line.split())) for line in lines[1:5]], dtype=np.float64
        )
        intrinsic = np.array(
            [list(map(float, line.split())) for line in lines[7:10]], dtype=np.float64
        )
        depth_line = lines[11].split()
        depth_min = float(depth_line[0])
        depth_interval = float(depth_line[1])

        # Scale intrinsics dynamically based on actual image dimensions
        scale_h = TARGET_HEIGHT / float(orig_h)
        scale_w = TARGET_WIDTH / float(orig_w)
        intrinsic[0, 0] *= scale_w
        intrinsic[1, 1] *= scale_h
        intrinsic[0, 2] *= scale_w
        intrinsic[1, 2] *= scale_h

        self.cache[view_idx] = {
            "image_tensor": image_tensor,
            "color_image": color_rgb,
            "intrinsic": intrinsic,
            "extrinsic_w2c": extrinsic_w2c,  # world → camera
            "depth_min": depth_min,
            "depth_interval": depth_interval,
        }
        return self.cache[view_idx]


def set_random_seeds(seed: int = 0) -> None:
    """Ensure deterministic sampling for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_pair_file(pair_path: str) -> List[Dict[str, List[int]]]:
    """Read DTU pair.txt and keep top 4 source views per reference view."""
    pairs = []
    with open(pair_path) as f:
        num_refs = int(f.readline())
        for _ in range(num_refs):
            ref_idx = int(f.readline())
            src_line = f.readline().strip().split()
            num_src = int(src_line[0])
            # src_line format: num_src id0 score0 id1 score1 ...
            src_indices = [
                int(src_line[i])
                for i in range(1, min(len(src_line), 1 + 2 * num_src), 2)
            ]
            pairs.append({"ref": ref_idx, "src": src_indices[:4]})
    return pairs


def prepare_batch(
    cache: ViewCache, ref_idx: int, src_indices: List[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a single MT-MVSNet batch (1, V, C, H, W) for a reference + source views."""
    view_ids = [ref_idx] + src_indices
    images = []
    intrinsics = []
    extrinsics = []

    for vid in view_ids:
        view_data = cache.get(vid)
        images.append(view_data["image_tensor"])
        intrinsics.append(torch.from_numpy(view_data["intrinsic"]).float())
        # For the network we use the same convention as training (world→camera)
        extrinsics.append(torch.from_numpy(view_data["extrinsic_w2c"]).float())

    images = torch.stack(images).unsqueeze(0)      # (1, V, 3, H, W)
    intrinsics = torch.stack(intrinsics).unsqueeze(0)  # (1, V, 3, 3)
    extrinsics = torch.stack(extrinsics).unsqueeze(0)  # (1, V, 4, 4)

    ref_view = cache.get(ref_idx)
    depth_min = ref_view["depth_min"]
    depth_interval = ref_view["depth_interval"]

    num_depths = int((DEPTH_MAX - depth_min) / depth_interval) + 1
    depth_values = torch.tensor(
        [depth_min + i * depth_interval for i in range(num_depths)],
        dtype=torch.float32,
    ).unsqueeze(0)  # (1, D)

    return images, intrinsics, extrinsics, depth_values


# -------------------------------------------------------------------------
# Geometric consistency & point projection
# -------------------------------------------------------------------------
def bilinear_sample(depth_map: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear sampling of a depth map at floating coordinates."""
    h, w = depth_map.shape
    x0 = np.floor(xs).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)

    dx = xs - x0
    dy = ys - y0

    top_left = depth_map[y0, x0]
    top_right = depth_map[y0, x1]
    bottom_left = depth_map[y1, x0]
    bottom_right = depth_map[y1, x1]

    top = top_left * (1 - dx) + top_right * dx
    bottom = bottom_left * (1 - dx) + bottom_right * dx
    return top * (1 - dy) + bottom * dy


def geometric_consistency_mask(
    ref_depth_mm: np.ndarray,
    ref_intrinsic: np.ndarray,
    ref_extrinsic_w2c: np.ndarray,
    src_depths_mm: List[np.ndarray],
    src_intrinsics: List[np.ndarray],
    src_extrinsics_w2c: List[np.ndarray],
) -> Tuple[np.ndarray, int, int]:
    """
    Multi-view geometric consistency in millimeters.

    - ref_extrinsic_w2c, src_extrinsics_w2c: world→camera.
    """
    h, w = ref_depth_mm.shape
    valid = ref_depth_mm > 0
    num_valid = int(np.count_nonzero(valid))
    if num_valid == 0:
        return np.zeros_like(ref_depth_mm, dtype=bool), 0, 0

    ys, xs = np.where(valid)
    depths = ref_depth_mm[valid]

    # [3, N] pixel coords
    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float64)

    # back-project to camera coords
    cam_coords = np.linalg.inv(ref_intrinsic) @ pixels * depths  # [3, N]

    # camera → world (invert world→camera)
    cam_coords_h = np.vstack([cam_coords, np.ones((1, cam_coords.shape[1]))])
    world_coords = np.linalg.inv(ref_extrinsic_w2c) @ cam_coords_h  # [4, N]

    counts = np.zeros(xs.shape[0], dtype=np.int32)

    for depth_src, intr_src, extr_src_w2c in zip(
        src_depths_mm, src_intrinsics, src_extrinsics_w2c
    ):
        if depth_src is None or intr_src is None or extr_src_w2c is None:
            continue

        # world → camera in source view
        cam_src = extr_src_w2c @ world_coords  # [4, N]
        zs = cam_src[2]
        positive = zs > 0
        if not np.any(positive):
            continue

        norm = cam_src[:3] / np.maximum(zs, 1e-6)
        proj = intr_src @ norm
        u = proj[0]
        v = proj[1]

        inside = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        candidate_idx = np.where(positive & inside)[0]
        if len(candidate_idx) == 0:
            continue

        sampled_depths = bilinear_sample(depth_src, u[candidate_idx], v[candidate_idx])
        valid_depth = sampled_depths > 0
        if not np.any(valid_depth):
            continue

        candidate_idx = candidate_idx[valid_depth]
        sampled_depths = sampled_depths[valid_depth]

        cam_depths = zs[candidate_idx]
        relative_error = np.abs(sampled_depths - cam_depths) / np.maximum(
            cam_depths, 1e-6
        )
        depth_ok = relative_error <= GEOMETRIC_REL_ERROR

        # Round-trip reprojection check: project source point back to reference
        # and verify pixel displacement < 1 pixel
        u_src = u[candidate_idx]
        v_src = v[candidate_idx]
        src_pixels = np.stack([u_src, v_src, np.ones_like(u_src)], axis=0)  # [3, N']
        src_cam_coords = np.linalg.inv(intr_src) @ src_pixels * sampled_depths  # [3, N']
        src_cam_h = np.vstack([src_cam_coords, np.ones((1, src_cam_coords.shape[1]))])
        # source camera → world → ref camera
        world_from_src = np.linalg.inv(extr_src_w2c) @ src_cam_h
        ref_cam_reproj = ref_extrinsic_w2c @ world_from_src  # [4, N']
        z_reproj = ref_cam_reproj[2]
        ref_proj = ref_intrinsic @ (ref_cam_reproj[:3] / np.maximum(z_reproj, 1e-6))
        u_reproj = ref_proj[0]
        v_reproj = ref_proj[1]
        pixel_diff_sq = (u_reproj - xs[candidate_idx])**2 + (v_reproj - ys[candidate_idx])**2
        pixel_ok = pixel_diff_sq < 1.0  # < 1 pixel displacement

        agreeing = depth_ok & pixel_ok
        counts[candidate_idx[agreeing]] += 1

    mask_flat = np.zeros(h * w, dtype=bool)
    valid_flat_indices = np.flatnonzero(valid)
    agreeing = counts >= MIN_CONSISTENT_VIEWS
    mask_flat[valid_flat_indices[agreeing]] = True
    num_consistent = int(np.count_nonzero(mask_flat))

    return mask_flat.reshape(h, w), num_valid, num_consistent


def depth_to_points(
    depth_mm: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_w2c: np.ndarray,
    color_image: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project depth map (mm) into world coordinates with colors."""
    if not np.any(mask):
        return np.empty((0, 3)), np.empty((0, 3))

    ys, xs = np.where(mask)
    depths = depth_mm[mask]

    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float64)
    cam_coords = np.linalg.inv(intrinsic) @ pixels * depths
    cam_coords_h = np.vstack([cam_coords, np.ones((1, cam_coords.shape[1]))])

    # world coords: invert world→camera
    world_coords = np.linalg.inv(extrinsic_w2c) @ cam_coords_h
    points = world_coords[:3].T
    colors = color_image[ys, xs].astype(np.float32)
    return points, colors


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, voxel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple voxel grid downsampling in 3D (units = mm)."""
    if len(points) == 0:
        return points, colors

    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    keys = (
        voxel_indices[:, 0] * 73856093
        + voxel_indices[:, 1] * 19349663
        + voxel_indices[:, 2] * 83492791
    )

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    sums_points = np.zeros((len(unique_keys), 3), dtype=np.float64)
    sums_colors = np.zeros((len(unique_keys), 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(unique_keys))

    for dim in range(3):
        sums_points[:, dim] = np.bincount(
            inverse, weights=points[:, dim], minlength=len(unique_keys)
        )
        sums_colors[:, dim] = np.bincount(
            inverse, weights=colors[:, dim], minlength=len(unique_keys)
        )

    down_points = (sums_points.T / counts).T
    down_colors = (sums_colors.T / counts).T
    return down_points.astype(np.float32), down_colors.astype(np.float32)


def visualize_prediction_vs_ground_truth(pred_path: str, gt_path: str) -> None:
    """Load and visualize predicted vs. ground-truth point clouds side-by-side."""
    missing = [p for p in [pred_path, gt_path] if not (p and os.path.exists(p))]
    if missing:
        print(f"Skipping visualization because files are missing: {missing}")
        return

    print(f"Loading predicted point cloud: {pred_path}")
    pcd_pred = o3d.io.read_point_cloud(pred_path)
    print(f"Loading ground-truth point cloud: {gt_path}")
    pcd_gt = o3d.io.read_point_cloud(gt_path)

    o3d.visualization.draw_geometries([pcd_pred, pcd_gt], window_name="Pred vs GT")


# -------------------------------------------------------------------------
# Inference + fusion
# -------------------------------------------------------------------------
def run_depth_estimation(
    model: MTMVSNet,
    cache: ViewCache,
    view_pairs: List[Dict[str, List[int]]],
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Run MT-MVSNet on all views that appear in pair.txt (single pass per view)."""
    depth_maps: Dict[int, np.ndarray] = {}
    confidences: Dict[int, np.ndarray] = {}

    all_view_ids = sorted(
        set([p["ref"] for p in view_pairs] + [sid for p in view_pairs for sid in p["src"]])
    )

    for ref_idx in all_view_ids[:MAX_VIEWS]:
        if ref_idx in depth_maps:
            continue

        pair = next((p for p in view_pairs if p["ref"] == ref_idx), None)
        source_views = pair["src"][:4] if pair else []

        images, intrinsics, extrinsics, depth_values = prepare_batch(
            cache, ref_idx, source_views
        )
        images = images.to(device)
        intrinsics = intrinsics.to(device)
        extrinsics = extrinsics.to(device)
        depth_values = depth_values.to(device)

        with torch.no_grad():
            outputs = model(images, intrinsics, extrinsics, depth_values)
        # Stage 0 is healthiest (std=0.241); later stages collapsed during training.
        # Use stage 0 depth + prob for more reliable predictions.
        depth_map = outputs[0][0][0].cpu().numpy()
        prob_volume = outputs[0][1][0].cpu()  # (D, H, W)
        confidence = torch.max(prob_volume, dim=0).values.numpy()  # (H, W)

        # Convert predicted depth to millimeters and suppress low-confidence values
        depth_map_mm = depth_map * DEPTH_SCALE_TO_MM
        depth_map_mm = np.where(confidence >= CONFIDENCE_THRESHOLD, depth_map_mm, 0.0)

        # Median filter to remove spike noise, then bilateral to smooth while preserving edges
        depth_map_mm = median_filter_depth(depth_map_mm, kernel_size=5)
        depth_map_mm = bilateral_filter_depth(depth_map_mm, sigma_spatial=7, sigma_depth=2.0)

        depth_maps[ref_idx] = depth_map_mm
        confidences[ref_idx] = confidence

    return depth_maps, confidences


def fuse_points(
    cache: ViewCache,
    view_pairs: List[Dict[str, List[int]]],
    depth_maps: Dict[int, np.ndarray],
    confidences: Dict[int, np.ndarray],
    log_entries: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-view fusion with photometric + geometric consistency."""
    point_list = []
    color_list = []

    for pair in view_pairs[:MAX_VIEWS]:
        ref_idx = pair["ref"]
        if ref_idx not in depth_maps:
            continue

        ref_view = cache.get(ref_idx)
        depth_mm = depth_maps[ref_idx]
        confidence = confidences[ref_idx]

        photo_mask = (confidence >= CONFIDENCE_THRESHOLD) & (depth_mm > 0)
        if not np.any(photo_mask):
            continue

        src_depths = []
        src_intrinsics = []
        src_extrinsics = []
        source_views = pair["src"][:4]

        for src_idx in source_views:
            if src_idx not in depth_maps:
                src_depths.append(None)
                src_intrinsics.append(None)
                src_extrinsics.append(None)
                continue
            src_depths.append(depth_maps[src_idx])
            src_intrinsics.append(cache.get(src_idx)["intrinsic"])
            src_extrinsics.append(cache.get(src_idx)["extrinsic_w2c"])

        valid_src_depths = [d for d in src_depths if d is not None]
        if len(valid_src_depths) < MIN_CONSISTENT_VIEWS:
            continue

        geom_mask, num_valid_pixels, num_consistent_pixels = geometric_consistency_mask(
            depth_mm,
            ref_view["intrinsic"],
            ref_view["extrinsic_w2c"],
            src_depths,
            src_intrinsics,
            src_extrinsics,
        )

        final_mask = photo_mask & geom_mask

        depth_vals = depth_mm[photo_mask]
        if depth_vals.size > 0:
            depth_min = float(depth_vals.min())
            depth_max = float(depth_vals.max())
        else:
            depth_min = 0.0
            depth_max = 0.0

        consistent_count = int(np.count_nonzero(final_mask))
        log_line = (
            f"Ref {ref_idx}: depth_range=[{depth_min:.1f}, {depth_max:.1f}]mm "
            f"valid_pixels={num_valid_pixels} consistent_pixels={num_consistent_pixels} "
            f"accepted_pixels={consistent_count}"
        )
        print(log_line)
        log_entries.append(log_line)

        print(
            f"[Fusion] Ref {ref_idx}: {consistent_count} pixels accepted, "
            f"depth range = [{depth_min:.1f}, {depth_max:.1f}] mm"
        )

        if consistent_count == 0:
            continue

        points, colors = depth_to_points(
            depth_mm,
            ref_view["intrinsic"],
            ref_view["extrinsic_w2c"],
            ref_view["color_image"],
            final_mask,
        )

        if len(points) > 0:
            point_list.append(points)
            color_list.append(colors)

    if not point_list:
        return np.empty((0, 3)), np.empty((0, 3))

    all_points = np.vstack(point_list)
    all_colors = np.vstack(color_list)

    # Voxel downsampling for efficiency
    down_points, down_colors = voxel_downsample(all_points, all_colors, VOXEL_SIZE)

    # Statistical outlier removal to clean stray points
    clean_points, clean_colors = statistical_outlier_removal(
        down_points, down_colors, nb_neighbors=20, std_ratio=2.0
    )
    print(f"[SOR] {len(down_points)} → {len(clean_points)} points after outlier removal")
    return clean_points, clean_colors


def save_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Save colored point cloud as ASCII PLY."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main() -> None:
    set_random_seeds(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    scan_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan29")
    pair_path = os.path.join(scan_path, "pair.txt")

    # Auto-detect depth units and adjust constants
    units = detect_depth_units(scan_path)
    apply_unit_settings(units)

    # Load model
    model = MTMVSNet(base_channels=32, num_stages=4).to(device)
    ckpt_path = os.path.join("checkpoints", "mtmvsnet_best.pth")
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        print("Loading model weights from checkpoint['model']")
        model.load_state_dict(checkpoint['model'])
    else:
        print("Loading raw state_dict")
        model.load_state_dict(checkpoint)
    # training script saved plain state_dict
    model.eval()

    cache = ViewCache(scan_path)
    view_pairs = read_pair_file(pair_path)
    log_entries: List[str] = []

    # 1) Depth estimation
    print("Running depth estimation...")
    depth_maps, confidences = run_depth_estimation(model, cache, view_pairs, device)

    # 2) Multi-view fusion
    print("Fusing depth maps into point cloud...")
    points, colors = fuse_points(cache, view_pairs, depth_maps, confidences, log_entries)

    output_path = os.path.join("outputs", "scan29_clean.ply")
    save_ply(output_path, points, colors)
    total_points = len(points)
    print(f"Saved fused point cloud with {total_points} points to {output_path}")

    # 3) Optional DTU-style evaluation (using eval_dtu.py)
    metrics_output = os.path.join("outputs", "scan29_metrics.txt")

    # Prefer fixed ASCII GT if you created it; otherwise, try original or env var
    gt_candidates = [
        os.environ.get("DTU_GT_PLY"),
        os.path.join(scan_path, "stl029_gt_ascii_fixed.ply"),
        os.path.join(scan_path, "scan29_clean.ply"),
        os.path.join(scan_path, "stl029_total.ply"),
    ]
    gt_path = next((p for p in gt_candidates if p and os.path.exists(p)), None)

    if gt_path:
        print(f"Evaluating against GT: {gt_path}")
        metrics = evaluate_point_cloud(output_path, gt_path)
    else:
        print("Ground-truth point cloud not found; metrics set to NaN")
        metrics = {
            "Accuracy": float("nan"),
            "Completeness": float("nan"),
            "Overall": float("nan"),
        }

    os.makedirs(os.path.dirname(metrics_output), exist_ok=True)
    with open(metrics_output, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {metrics['Accuracy']:.6f}\n")
        f.write(f"Completeness: {metrics['Completeness']:.6f}\n")
        f.write(f"Overall: {metrics['Overall']:.6f}\n")
        f.write(f"Number of points: {total_points}\n")
    print(f"Saved metrics to {metrics_output}")

    # 4) Log summary
    log_entries.append(f"Total points: {total_points}")
    log_entries.append(
        f"Metrics - Accuracy: {metrics['Accuracy']:.6f}, "
        f"Completeness: {metrics['Completeness']:.6f}, "
        f"Overall: {metrics['Overall']:.6f}"
    )

    os.makedirs(LOG_DIR, exist_ok=True)
    summary_path = os.path.join(LOG_DIR, "scan29_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Scan29 fusion summary\n")
        for line in log_entries:
            f.write(line + "\n")
    print(f"Saved log summary to {summary_path}")

    # Optional Open3D visualization for qualitative alignment checks
    if os.environ.get("OPEN3D_COMPARE", "0") == "1":
        pred_vis = os.environ.get("PRED_PLY", os.path.join(os.getcwd(), "stl029_total.ply"))
        gt_vis = os.environ.get("GT_PLY", os.path.join(os.getcwd(), "scan29_clean.ply"))
        visualize_prediction_vs_ground_truth(pred_vis, gt_vis)


if __name__ == "__main__":
    main()
