"""
MT-MVSNet Test with FAST FILTERING (NO Open3D dependency)
"""

import os
import cv2
import torch
import numpy as np
from scipy.spatial import cKDTree
from mtmvsnet_model import MTMVSNet
from data_loader import load_images_and_camera_params


def read_all_view_pairs(pair_file_path):
    """Read all reference-source pairs from pair.txt"""
    with open(pair_file_path) as f:
        num_views = int(f.readline())
        view_pairs = []
        for _ in range(num_views):
            ref_idx = int(f.readline())
            src_line = f.readline().strip().split()
            num_src = int(src_line[0])
            src_indices = [int(src_line[i]) for i in range(1, min(21, len(src_line)), 2)]
            view_pairs.append({'ref': ref_idx, 'src': src_indices[:4]})
    return view_pairs


def generate_depth_maps_batch(model, scan_path, view_pairs, device, num_views=49):
    """Generate depth maps for multiple views"""
    depth_data = {
        'depth_maps': [],
        'confidences': [],
        'intrinsics': [],
        'extrinsics': [],
        'view_indices': [],
        'img_paths': []
    }

    ORIGINAL_HEIGHT = 1200
    ORIGINAL_WIDTH = 1600
    TARGET_HEIGHT = 512
    TARGET_WIDTH = 640
    scale_h = TARGET_HEIGHT / ORIGINAL_HEIGHT
    scale_w = TARGET_WIDTH / ORIGINAL_WIDTH

    print(f"Generating depth maps for {num_views} views...")

    for i, pair in enumerate(view_pairs[:num_views]):
        ref_idx = pair['ref']
        src_indices = pair['src'][:4]

        print(f"\n[{i+1}/{num_views}] View {ref_idx} <- sources {src_indices}")

        try:
            temp_pair = os.path.join(scan_path, f"temp_pair_{ref_idx}.txt")
            with open(temp_pair, 'w') as f:
                f.write(f"{len(src_indices) + 1}\n")
                f.write(f"{ref_idx}\n")
                f.write(f"{len(src_indices)} " + " ".join([f"{idx} 1.0" for idx in src_indices]) + "\n")

            images, intrinsics, extrinsics, depth_values = load_images_and_camera_params(scan_path, temp_pair)

            images = images.to(device).unsqueeze(0)
            intrinsics_batch = intrinsics.unsqueeze(0).to(device)
            extrinsics_batch = extrinsics.unsqueeze(0).to(device)
            depth_values = depth_values.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(images, intrinsics_batch, extrinsics_batch, depth_values)

            depth_map = outputs[3][0][0].cpu().numpy()
            prob_volume = outputs[3][1][0].cpu()
            confidence = prob_volume.max(dim=0)[0].numpy()

            cam_path = os.path.join(scan_path, "cams", f"{ref_idx:08d}_cam.txt")
            with open(cam_path) as f:
                lines = f.readlines()
                extrinsic = np.array([list(map(float, line.split())) for line in lines[1:5]])
                intrinsic = np.array([list(map(float, line.split())) for line in lines[7:10]])
                intrinsic[0, :] *= scale_w
                intrinsic[1, :] *= scale_h

            img_path = os.path.join(scan_path, "images", f"{ref_idx:08d}.jpg")

            depth_data['depth_maps'].append(depth_map)
            depth_data['confidences'].append(confidence)
            depth_data['intrinsics'].append(intrinsic)
            depth_data['extrinsics'].append(extrinsic)
            depth_data['view_indices'].append(ref_idx)
            depth_data['img_paths'].append(img_path)

            valid_pixels = (depth_map > 0).sum()
            print(f"  Depth: [{depth_map.min():.1f}, {depth_map.max():.1f}]mm, Valid: {valid_pixels}")

            os.remove(temp_pair)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    return depth_data


def depth_to_points(depth_map, intrinsic, extrinsic, img_path, mask=None, target_size=(640, 512)):
    """Convert depth map to 3D points with colors"""
    H, W = depth_map.shape

    # Load image
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply mask
    if mask is not None:
        valid_mask = mask & (depth_map > 0)
    else:
        valid_mask = depth_map > 0

    y, x = np.where(valid_mask)

    if len(x) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    # Back-project to 3D
    pixels = np.stack([x, y, np.ones_like(x)], axis=0).astype(np.float32)
    depths = depth_map[valid_mask]

    K_inv = np.linalg.inv(intrinsic)
    cam_coords = K_inv @ pixels * depths
    cam_coords_homo = np.vstack([cam_coords, np.ones((1, len(depths)))])

    extrinsic_inv = np.linalg.inv(extrinsic)
    world_coords = extrinsic_inv @ cam_coords_homo
    points = world_coords[:3].T

    colors = img[y, x]

    return points, colors


def fast_voxel_downsample(points, colors, voxel_size=0.01):
    """Fast voxel downsampling using vectorized operations"""
    if len(points) == 0:
        return points, colors
    
    print(f"  Downsampling {len(points):,} points...")
    
    # Quantize points to voxel grid
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    # Create unique voxel keys (faster than loop)
    voxel_keys = (voxel_indices[:, 0].astype(np.int64) * 1000000000 + 
                  voxel_indices[:, 1].astype(np.int64) * 1000000 + 
                  voxel_indices[:, 2].astype(np.int64))
    
    # Get unique voxels and inverse indices
    unique_keys, inverse = np.unique(voxel_keys, return_inverse=True)
    
    # Average points in each voxel (vectorized)
    downsampled_points = np.zeros((len(unique_keys), 3))
    downsampled_colors = np.zeros((len(unique_keys), 3))
    
    # Use bincount for fast averaging (much faster than loop!)
    for dim in range(3):
        downsampled_points[:, dim] = np.bincount(inverse, weights=points[:, dim]) / np.bincount(inverse)
        downsampled_colors[:, dim] = np.bincount(inverse, weights=colors[:, dim]) / np.bincount(inverse)
    
    print(f"  Result: {len(downsampled_points):,} points")
    
    return downsampled_points, downsampled_colors


def fuse_with_fast_filtering(depth_data, conf_threshold=0.03, voxel_size=0.02):
    """FAST FUSION without Open3D"""
    n_views = len(depth_data['depth_maps'])

    if n_views == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    print(f"\n{'='*70}")
    print("FAST FILTERING & FUSION")
    print(f"{'='*70}")
    print(f"Parameters:")
    print(f"  - Confidence threshold: {conf_threshold}")
    print(f"  - Voxel size: {voxel_size}m")
    print(f"  - Total views: {n_views}\n")

    points_list = []
    colors_list = []

    for i in range(n_views):
        view_idx = depth_data['view_indices'][i]
        print(f"[{i+1}/{n_views}] Processing view {view_idx}...")

        depth_ref = depth_data['depth_maps'][i]
        conf_ref = depth_data['confidences'][i]
        intrinsic_ref = depth_data['intrinsics'][i]
        extrinsic_ref = depth_data['extrinsics'][i]
        img_path = depth_data['img_paths'][i]

        # Simple confidence filter
        mask = (conf_ref > conf_threshold) & (depth_ref > 0)

        valid_count = mask.sum()
        print(f"  Valid pixels: {valid_count} ({valid_count/mask.size*100:.2f}%)")

        if valid_count < 100:
            print(f"  Skipped")
            continue

        points, colors = depth_to_points(
            depth_ref, intrinsic_ref, extrinsic_ref, img_path, mask=mask
        )

        print(f"  Generated: {len(points)} points")

        if len(points) > 0:
            points_list.append(points)
            colors_list.append(colors)

    if len(points_list) == 0:
        print("\nNo valid points!")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    print(f"\n{'='*70}")
    print("POINT CLOUD FUSION")
    print(f"{'='*70}")

    # Concatenate
    all_points = np.vstack(points_list)
    all_colors = np.vstack(colors_list)
    print(f"Before downsampling: {len(all_points):,} points")

    # Fast downsampling
    print(f"Voxel downsampling (size={voxel_size}m)...")
    fused_points, fused_colors = fast_voxel_downsample(all_points, all_colors, voxel_size=voxel_size)

    print(f"\nFinal: {len(fused_points):,} points")

    return fused_points, fused_colors


def save_ply(filename, points, colors):
    """Save point cloud as PLY file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
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

        for i in range(len(points)):
            f.write(f"{points[i,0]} {points[i,1]} {points[i,2]} ")
            f.write(f"{int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}\n")


def test_scan_fast(model, scan_path, output_dir, device, num_views=49, conf=0.03, voxel_size=0.02):
    """Test scan with FAST filtering"""
    scan_name = os.path.basename(scan_path)

    print(f"\n{'='*70}")
    print(f"TESTING {scan_name.upper()} - FAST APPROACH")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  - Number of views: {num_views}")
    print(f"  - Confidence threshold: {conf}")
    print(f"  - Voxel size: {voxel_size}m")
    print(f"{'='*70}\n")

    pair_file = os.path.join(scan_path, "pair.txt")
    view_pairs = read_all_view_pairs(pair_file)

    print(f"Available views: {len(view_pairs)}")
    print(f"Processing: {min(num_views, len(view_pairs))} views\n")

    print(f"{'='*70}")
    print("STEP 1: DEPTH MAP GENERATION")
    print(f"{'='*70}\n")

    depth_data = generate_depth_maps_batch(model, scan_path, view_pairs, device, num_views)

    if len(depth_data['depth_maps']) == 0:
        print("\nNo depth maps generated!")
        return None, None

    print(f"\nGenerated {len(depth_data['depth_maps'])} depth maps")

    print(f"\n{'='*70}")
    print("STEP 2: FAST FILTERING & FUSION")
    print(f"{'='*70}")

    fused_points, fused_colors = fuse_with_fast_filtering(
        depth_data,
        conf_threshold=conf,
        voxel_size=voxel_size
    )

    if len(fused_points) == 0:
        print("\nFusion failed!")
        return None, None

    scan_output_dir = os.path.join(output_dir, scan_name)
    ply_file = os.path.join(scan_output_dir, f"{scan_name}.ply")
    save_ply(ply_file, fused_points, fused_colors)

    print(f"\n{'='*70}")
    print(f"SUCCESS!")
    print(f"{'='*70}")
    print(f"Output: {ply_file}")
    print(f"Points: {len(fused_points):,}")
    print(f"{'='*70}\n")

    return fused_points, fused_colors


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("MT-MVSNet - FAST FILTERING TEST (No Open3D)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Testing on: scan29")
    print("="*70 + "\n")

    model = MTMVSNet(base_channels=32, num_stages=4).to(device)
    checkpoint = torch.load("./checkpoints/mtmvsnet_trained.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded\n")

    output_dir = "./test_outputs_fast"
    scan_path = "/project/yuxiaojun/dtu_testing/dtu/scan29"

    points, colors = test_scan_fast(
        model, scan_path, output_dir, device,
        num_views=49,
        conf=0.03,
        voxel_size=0.02  # Larger voxel = fewer points but faster
    )

    if points is not None:
        print("="*70)
        print("TEST COMPLETE!")
        print("="*70)
        print(f"\nResult: {len(points):,} points")
        print("Check: ./test_outputs_fast/scan29/scan29.ply\n")


if __name__ == "__main__":
    main()
