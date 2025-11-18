"""
Corrected Fusion Script for MT-MVSNet
Implements proper geometric consistency filtering for DTU evaluation
"""

import os
import numpy as np
import cv2
from scipy.spatial import cKDTree

ORIGINAL_WIDTH = 1600
ORIGINAL_HEIGHT = 1200


def _maybe_scale_intrinsic(intrinsic, target_size=(640, 512)):
    """Scale intrinsic matrix if it still corresponds to the original DTU size."""

    if intrinsic is None:
        return None

    target_width, target_height = target_size
    if intrinsic[0, 2] <= target_width and intrinsic[1, 2] <= target_height:
        return intrinsic

    scale_w = target_width / ORIGINAL_WIDTH
    scale_h = target_height / ORIGINAL_HEIGHT
    scaled = intrinsic.copy()
    scaled[0, :] *= scale_w
    scaled[1, :] *= scale_h
    return scaled


def filter_depth_geometric(depth_ref, confidence_ref, depth_src_list,
                          extrinsics_ref, extrinsics_src_list,
                          intrinsic_ref, intrinsic_src_list,
                          depth_threshold=0.01, conf_threshold=0.3,
                          num_consistent=3, target_size=(640, 512)):
    """
    Filter depth map using geometric consistency across multiple views
    """
    H, W = depth_ref.shape
    
    # Initialize consistency counter
    consistency_count = np.zeros((H, W), dtype=np.int32)
    
    # Create coordinate grid
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Reference 3D points
    pixels_ref = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3).T
    depth_ref_flat = depth_ref.reshape(-1)
    
    # Back-project to camera coordinates
    intrinsic_ref = _maybe_scale_intrinsic(intrinsic_ref, target_size)
    K_inv = np.linalg.inv(intrinsic_ref)
    cam_coords = K_inv @ pixels_ref * depth_ref_flat
    
    # Transform to world coordinates
    cam_coords_homo = np.vstack([cam_coords, np.ones((1, H*W))])
    extrinsic_ref_inv = np.linalg.inv(extrinsics_ref)
    world_coords = extrinsic_ref_inv @ cam_coords_homo
    
    # Check consistency with each source view
    for depth_src, extrinsic_src, intrinsic_src in zip(depth_src_list,
                                                        extrinsics_src_list,
                                                        intrinsic_src_list):
        intrinsic_src = _maybe_scale_intrinsic(intrinsic_src, target_size)
        # Project to source view
        cam_coords_src = extrinsic_src @ world_coords
        pixels_src = intrinsic_src @ cam_coords_src[:3, :]
        
        # Normalize
        z_src = pixels_src[2, :]
        x_src = pixels_src[0, :] / (z_src + 1e-8)
        y_src = pixels_src[1, :] / (z_src + 1e-8)
        
        # Check if within image bounds
        valid_x = (x_src >= 0) & (x_src < W)
        valid_y = (y_src >= 0) & (y_src < H)
        valid_mask = valid_x & valid_y & (z_src > 0)
        
        # Sample depth from source
        x_src_int = np.round(x_src).astype(np.int32)
        y_src_int = np.round(y_src).astype(np.int32)
        
        x_src_int = np.clip(x_src_int, 0, W - 1)
        y_src_int = np.clip(y_src_int, 0, H - 1)
        
        depth_src_sampled = depth_src[y_src_int, x_src_int]
        
        # Compute relative depth error
        depth_error = np.abs(z_src - depth_src_sampled) / (z_src + 1e-8)
        
        # Check consistency
        consistent = valid_mask & (depth_error < depth_threshold)
        consistency_count += consistent.reshape(H, W)
    
    # Filter based on consistency and confidence
    mask = (consistency_count >= num_consistent) & (confidence_ref >= conf_threshold) & (depth_ref > 0)
    
    filtered_depth = depth_ref.copy()
    filtered_depth[~mask] = 0
    
    return filtered_depth, mask


def depth_to_points_filtered(depth_map, intrinsic, extrinsic, img_path, mask=None, target_size=(640, 512)):
    """Convert filtered depth map to 3D points with colors"""
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
    
    intrinsic = _maybe_scale_intrinsic(intrinsic, target_size)
    K_inv = np.linalg.inv(intrinsic)
    cam_coords = K_inv @ pixels * depths
    cam_coords_homo = np.vstack([cam_coords, np.ones((1, len(depths)))])
    
    extrinsic_inv = np.linalg.inv(extrinsic)
    world_coords = extrinsic_inv @ cam_coords_homo
    points = world_coords[:3].T
    
    colors = img[y, x]
    
    return points, colors


def statistical_outlier_removal(points, colors, k_neighbors=20, std_ratio=2.0):
    """Remove statistical outliers using k-nearest neighbors"""
    if len(points) < k_neighbors:
        return points, colors
    
    # Build KD-tree
    tree = cKDTree(points)
    
    # Compute distances to k nearest neighbors
    distances, _ = tree.query(points, k=k_neighbors + 1)
    mean_distances = distances[:, 1:].mean(axis=1)
    
    # Compute statistics
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    
    # Filter outliers
    threshold = global_mean + std_ratio * global_std
    inlier_mask = mean_distances < threshold
    
    filtered_points = points[inlier_mask]
    filtered_colors = colors[inlier_mask]
    
    return filtered_points, filtered_colors


def fuse_point_clouds(points_list, colors_list, voxel_size=0.01):
    """Fuse multiple point clouds by voxel downsampling"""
    if len(points_list) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # Concatenate all points
    all_points = np.vstack(points_list)
    all_colors = np.vstack(colors_list)
    
    # Voxel downsampling
    voxel_coords = np.floor(all_points / voxel_size).astype(np.int32)
    
    # Create unique voxel identifier
    voxel_ids = voxel_coords[:, 0] * 1000000 + voxel_coords[:, 1] * 1000 + voxel_coords[:, 2]
    
    # Get unique voxels
    unique_voxels, inverse_indices = np.unique(voxel_ids, return_inverse=True)
    
    # Average points and colors in each voxel
    fused_points = np.zeros((len(unique_voxels), 3))
    fused_colors = np.zeros((len(unique_voxels), 3))
    
    for i in range(len(unique_voxels)):
        mask = inverse_indices == i
        fused_points[i] = all_points[mask].mean(axis=0)
        fused_colors[i] = all_colors[mask].mean(axis=0)
    
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
