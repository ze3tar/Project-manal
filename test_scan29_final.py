"""Balanced fusion pipeline for DTU scan29 using MT-MVSNet."""

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from mtmvsnet_model import MTMVSNet

TARGET_HEIGHT = 512
TARGET_WIDTH = 640
CONFIDENCE_THRESHOLD = 0.6
GEOMETRIC_REL_ERROR = 0.01
MIN_CONSISTENT_VIEWS = 2
VOXEL_SIZE = 0.015  # meters
MAX_VIEWS = 49
DEPTH_MAX = 935.0  # millimeters


class ViewCache:
    """Lazy loader for per-view assets."""

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
        image_bgr = cv2.resize(image_bgr, (TARGET_WIDTH, TARGET_HEIGHT))
        image_tensor = torch.from_numpy(image_bgr).permute(2, 0, 1).float() / 255.0
        color_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with open(cam_path) as f:
            lines = f.readlines()
        extrinsic = np.array([list(map(float, line.split())) for line in lines[1:5]])
        intrinsic = np.array([list(map(float, line.split())) for line in lines[7:10]])
        depth_line = lines[11].split()
        depth_min = float(depth_line[0])
        depth_interval = float(depth_line[1])

        scale_h = TARGET_HEIGHT / 1200.0
        scale_w = TARGET_WIDTH / 1600.0
        intrinsic[0, :] *= scale_w
        intrinsic[1, :] *= scale_h

        extrinsic_m = extrinsic.copy()
        extrinsic_m[:3, 3] /= 1000.0  # convert translation to meters

        self.cache[view_idx] = {
            "image_tensor": image_tensor,
            "color_image": color_rgb,
            "intrinsic": intrinsic,
            "extrinsic_mm": extrinsic,
            "extrinsic_m": extrinsic_m,
            "depth_min": depth_min,
            "depth_interval": depth_interval,
        }
        return self.cache[view_idx]


def read_pair_file(pair_path: str) -> List[Dict[str, List[int]]]:
    pairs = []
    with open(pair_path) as f:
        num_refs = int(f.readline())
        for _ in range(num_refs):
            ref_idx = int(f.readline())
            src_line = f.readline().strip().split()
            num_src = int(src_line[0])
            src_indices = [int(src_line[i]) for i in range(1, min(len(src_line), 1 + 2 * num_src), 2)]
            pairs.append({"ref": ref_idx, "src": src_indices[:4]})
    return pairs


def prepare_batch(cache: ViewCache, ref_idx: int, src_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    view_ids = [ref_idx] + src_indices
    images = []
    intrinsics = []
    extrinsics = []

    for vid in view_ids:
        view_data = cache.get(vid)
        images.append(view_data["image_tensor"])
        intrinsics.append(torch.from_numpy(view_data["intrinsic"]).float())
        extrinsics.append(torch.from_numpy(view_data["extrinsic_mm"]).float())

    images = torch.stack(images).unsqueeze(0)
    intrinsics = torch.stack(intrinsics).unsqueeze(0)
    extrinsics = torch.stack(extrinsics).unsqueeze(0)

    ref_view = cache.get(ref_idx)
    depth_min = ref_view["depth_min"]
    depth_interval = ref_view["depth_interval"]
    num_depths = int((DEPTH_MAX - depth_min) / depth_interval) + 1
    depth_values = torch.tensor([depth_min + i * depth_interval for i in range(num_depths)], dtype=torch.float32)
    depth_values = depth_values.unsqueeze(0)

    return images, intrinsics, extrinsics, depth_values


def bilinear_sample(depth_map: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
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
    ref_depth_m: np.ndarray,
    ref_intrinsic: np.ndarray,
    ref_extrinsic_m: np.ndarray,
    src_depths_m: List[np.ndarray],
    src_intrinsics: List[np.ndarray],
    src_extrinsics_m: List[np.ndarray],
) -> np.ndarray:
    h, w = ref_depth_m.shape
    valid = ref_depth_m > 0
    if not np.any(valid):
        return np.zeros_like(ref_depth_m, dtype=bool)

    ys, xs = np.where(valid)
    depths = ref_depth_m[valid]

    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float64)
    cam_coords = np.linalg.inv(ref_intrinsic) @ pixels * depths
    cam_coords_h = np.vstack([cam_coords, np.ones((1, cam_coords.shape[1]))])
    world_coords = np.linalg.inv(ref_extrinsic_m) @ cam_coords_h

    counts = np.zeros(len(xs), dtype=np.int32)

    for depth_src, intr_src, extr_src in zip(src_depths_m, src_intrinsics, src_extrinsics_m):
        if depth_src is None or intr_src is None or extr_src is None:
            continue
        cam_src = extr_src @ world_coords
        zs = cam_src[2]
        positive = zs > 0
        if not np.any(positive):
            continue
        proj = intr_src @ (cam_src[:3] / zs)
        u = proj[0]
        v = proj[1]
        inside = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        valid_idx = np.where(positive & inside)[0]
        if len(valid_idx) == 0:
            continue
        sampled = bilinear_sample(depth_src, u[valid_idx], v[valid_idx])
        depth_valid = sampled > 0
        if not np.any(depth_valid):
            continue
        valid_idx = valid_idx[depth_valid]
        sampled = sampled[depth_valid]
        zs_valid = zs[valid_idx]
        rel_error = np.abs(sampled - zs_valid) / np.maximum(zs_valid, 1e-6)
        agree = rel_error <= GEOMETRIC_REL_ERROR
        counts[valid_idx[agree]] += 1

    mask_flat = np.zeros(h * w, dtype=bool)
    valid_flat_indices = np.flatnonzero(valid)
    agreeing = counts >= MIN_CONSISTENT_VIEWS
    mask_flat[valid_flat_indices[agreeing]] = True
    return mask_flat.reshape(h, w)


def depth_to_points(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_m: np.ndarray,
    color_image: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if not np.any(mask):
        return np.empty((0, 3)), np.empty((0, 3))

    ys, xs = np.where(mask)
    depths = depth_m[mask]
    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float64)
    cam_coords = np.linalg.inv(intrinsic) @ pixels * depths
    cam_coords_h = np.vstack([cam_coords, np.ones((1, cam_coords.shape[1]))])
    world_coords = np.linalg.inv(extrinsic_m) @ cam_coords_h
    points = world_coords[:3].T
    colors = color_image[ys, xs].astype(np.float32)
    return points, colors


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, colors
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    keys = voxel_indices[:, 0] * 73856093 + voxel_indices[:, 1] * 19349663 + voxel_indices[:, 2] * 83492791
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    sums_points = np.zeros((len(unique_keys), 3), dtype=np.float64)
    sums_colors = np.zeros((len(unique_keys), 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(unique_keys))
    for dim in range(3):
        sums_points[:, dim] = np.bincount(inverse, weights=points[:, dim], minlength=len(unique_keys))
        sums_colors[:, dim] = np.bincount(inverse, weights=colors[:, dim], minlength=len(unique_keys))
    down_points = (sums_points.T / counts).T
    down_colors = (sums_colors.T / counts).T
    return down_points.astype(np.float32), down_colors.astype(np.float32)


def run_depth_estimation(
    model: MTMVSNet,
    cache: ViewCache,
    view_pairs: List[Dict[str, List[int]]],
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    depth_maps: Dict[int, np.ndarray] = {}
    confidences: Dict[int, np.ndarray] = {}

    for pair in view_pairs[:MAX_VIEWS]:
        ref_idx = pair["ref"]
        if ref_idx in depth_maps:
            continue
        src_indices = pair["src"]
        images, intrinsics, extrinsics, depth_values = prepare_batch(cache, ref_idx, src_indices)
        images = images.to(device)
        intrinsics = intrinsics.to(device)
        extrinsics = extrinsics.to(device)
        depth_values = depth_values.to(device)

        with torch.no_grad():
            outputs = model(images, intrinsics, extrinsics, depth_values)

        depth_map = outputs[-1][0][0].cpu().numpy()
        prob_volume = outputs[-1][1][0].cpu()
        confidence = torch.max(prob_volume, dim=0).values.numpy()

        depth_maps[ref_idx] = depth_map
        confidences[ref_idx] = confidence

    return depth_maps, confidences


def fuse_points(
    cache: ViewCache,
    view_pairs: List[Dict[str, List[int]]],
    depth_maps: Dict[int, np.ndarray],
    confidences: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    point_list = []
    color_list = []

    for pair in view_pairs[:MAX_VIEWS]:
        ref_idx = pair["ref"]
        if ref_idx not in depth_maps:
            continue
        ref_view = cache.get(ref_idx)
        depth_m = depth_maps[ref_idx] / 1000.0
        confidence = confidences[ref_idx]
        photo_mask = (confidence >= CONFIDENCE_THRESHOLD) & (depth_m > 0)
        if not np.any(photo_mask):
            continue

        src_depths = []
        src_intrinsics = []
        src_extrinsics = []
        for src_idx in pair["src"]:
            if src_idx not in depth_maps:
                src_depths.append(None)
                src_intrinsics.append(None)
                src_extrinsics.append(None)
                continue
            src_depths.append(depth_maps[src_idx] / 1000.0)
            src_intrinsics.append(cache.get(src_idx)["intrinsic"])
            src_extrinsics.append(cache.get(src_idx)["extrinsic_m"])

        valid_src_depths = [d for d in src_depths if d is not None]
        if len(valid_src_depths) < MIN_CONSISTENT_VIEWS:
            continue

        geom_mask = geometric_consistency_mask(
            depth_m,
            ref_view["intrinsic"],
            ref_view["extrinsic_m"],
            src_depths,
            src_intrinsics,
            src_extrinsics,
        )

        final_mask = photo_mask & geom_mask
        if not np.any(final_mask):
            continue

        points, colors = depth_to_points(
            depth_m,
            ref_view["intrinsic"],
            ref_view["extrinsic_m"],
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
    return voxel_downsample(all_points, all_colors, VOXEL_SIZE)


def save_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scan_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan29")
    pair_path = os.path.join(scan_path, "pair.txt")

    model = MTMVSNet(base_channels=32, num_stages=4).to(device)
    checkpoint = torch.load(os.path.join("checkpoints", "mtmvsnet_trained.pth"), map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    cache = ViewCache(scan_path)
    view_pairs = read_pair_file(pair_path)

    depth_maps, confidences = run_depth_estimation(model, cache, view_pairs, device)
    points, colors = fuse_points(cache, view_pairs, depth_maps, confidences)

    output_path = os.path.join("outputs", "scan29_clean.ply")
    save_ply(output_path, points, colors)
    print(f"Saved fused point cloud with {len(points)} points to {output_path}")


if __name__ == "__main__":
    main()
