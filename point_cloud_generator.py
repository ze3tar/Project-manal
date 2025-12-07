# point_cloud_generator.py
import numpy as np
import open3d as o3d
import torch

def create_point_cloud_from_depth_fixed(depth_map, intrinsic, extrinsic, image=None):
    """
    Fixed version of point cloud generation with color correction for blue tint issue
    """
    print("=== FIXED POINT CLOUD GENERATION ===")
    
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(intrinsic, torch.Tensor):
        intrinsic = intrinsic.cpu().numpy()
    if isinstance(extrinsic, torch.Tensor):
        extrinsic = extrinsic.cpu().numpy()

    H, W = depth_map.shape
    print(f"Depth map shape: {depth_map.shape}")
    
    i_range = np.arange(H)
    j_range = np.arange(W)
    i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')

    # Pixel coordinates [3, H*W]
    pixels = np.stack((j_grid, i_grid, np.ones_like(j_grid)), axis=-1).reshape(-1, 3).T

    inv_K = np.linalg.inv(intrinsic)
    # Convert to millimeters (network predicts meters) for alignment with GT
    depth_map_mm = depth_map * 1000.0

    cam_coords = inv_K @ pixels * depth_map_mm.reshape(1, -1)

    # Homogeneous coordinates
    cam_coords_hom = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
    # Extrinsic provided as world → camera; invert to transform into world frame
    world_coords = np.linalg.inv(extrinsic) @ cam_coords_hom
    points = world_coords[:3].T  # [N, 3]

    # Valid depth mask
    valid = depth_map_mm.reshape(-1) > 0
    points = points[valid]
    print(f"Valid points: {valid.sum()}/{len(valid)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if image is not None:
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # Handle tensor format properly
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)  # [3,H,W] -> [H,W,3]
            print("Converted from [3,H,W] to [H,W,3]")

        if image.max() > 1.0:
            image = image / 255.0
        image = image.astype(np.float32)

        print(f"Image shape: {image.shape}")
        print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")

        # COLOR CORRECTION - This fixes the blue tint!
        r_avg, g_avg, b_avg = image[:,:,0].mean(), image[:,:,1].mean(), image[:,:,2].mean()
        print(f"Original color averages: R={r_avg:.3f}, G={g_avg:.3f}, B={b_avg:.3f}")

        # Check if this looks like a BGR image (blue channel dominates)
        if b_avg > r_avg * 1.2:  # Blue significantly higher than red
            print("FIXING: BGR->RGB correction (swapping R and B channels)")
            image = image[:, :, [2, 1, 0]]  # Swap blue and red channels
            r_avg, g_avg, b_avg = image[:,:,0].mean(), image[:,:,1].mean(), image[:,:,2].mean()
            print(f"After BGR->RGB swap: R={r_avg:.3f}, G={g_avg:.3f}, B={b_avg:.3f}")

        # Apply gentle color balancing to remove any remaining tint
        overall_mean = (r_avg + g_avg + b_avg) / 3
        if overall_mean > 0.01:  # Avoid division by very small numbers
            # Calculate correction factors (limit to reasonable range)
            r_factor = np.clip(overall_mean / (r_avg + 1e-8), 0.7, 1.4)
            g_factor = np.clip(overall_mean / (g_avg + 1e-8), 0.7, 1.4)
            b_factor = np.clip(overall_mean / (b_avg + 1e-8), 0.7, 1.4)
            
            # Apply gentle correction (30% of full correction)
            r_factor = 1.0 + (r_factor - 1.0) * 0.3
            g_factor = 1.0 + (g_factor - 1.0) * 0.3
            b_factor = 1.0 + (b_factor - 1.0) * 0.3
            
            image[:,:,0] *= r_factor
            image[:,:,1] *= g_factor
            image[:,:,2] *= b_factor
            image = np.clip(image, 0, 1)
            
            print(f"Applied color balance: R×{r_factor:.2f}, G×{g_factor:.2f}, B×{b_factor:.2f}")

        # Final brightness adjustment if image is too dark
        brightness = image.mean()
        if brightness < 0.25:
            gamma = 0.8  # Brighten
            image = np.power(image, gamma)
            print(f"Applied brightness correction: gamma={gamma}")

        # Final color check
        final_r, final_g, final_b = image[:,:,0].mean(), image[:,:,1].mean(), image[:,:,2].mean()
        print(f"Final color averages: R={final_r:.3f}, G={final_g:.3f}, B={final_b:.3f}")

        colors = image.reshape(-1, 3)[valid]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    print("Point cloud created successfully with color correction!")
    return pcd

# Keep your original function for backup/comparison
def create_point_cloud_from_depth(depth_map, intrinsic, extrinsic, image=None):
    """
    Your original function (kept for backup)
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()

    if isinstance(intrinsic, torch.Tensor):
        intrinsic = intrinsic.cpu().numpy()
    if isinstance(extrinsic, torch.Tensor):
        extrinsic = extrinsic.cpu().numpy()

    H, W = depth_map.shape
    i_range = np.arange(H)
    j_range = np.arange(W)
    i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')

    # Pixel coordinates [3, H*W]
    pixels = np.stack((j_grid, i_grid, np.ones_like(j_grid)), axis=-1).reshape(-1, 3).T

    inv_K = np.linalg.inv(intrinsic)
    depth_map_mm = depth_map * 1000.0

    cam_coords = inv_K @ pixels * depth_map_mm.reshape(1, -1)

    # Homogeneous coordinates
    cam_coords_hom = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
    world_coords = np.linalg.inv(extrinsic) @ cam_coords_hom
    points = world_coords[:3].T  # [N, 3]

    # Valid depth mask
    valid = depth_map_mm.reshape(-1) > 0
    points = points[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if image is not None:
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        if image.max() > 1.0:
            image = image / 255.0
        image = image.astype(np.float32)
        colors = image.reshape(-1, 3)[valid]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


# ---------------------------
# MeshLab/Open3D conveniences
# ---------------------------

def estimate_and_orient_normals(pcd, radius=0.02, max_nn=30):
    """
    Estimate and orient normals for nicer shading in Open3D/MeshLab.
    """
    if len(pcd.points) == 0:
        print("No points to estimate normals on.")
        return pcd
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(k=max_nn)
    except Exception:
        # Fallback if the method isn't available on some builds
        pcd.normalize_normals()
    return pcd

def save_point_cloud(pcd, path="reconstruction.ply"):
    """
    Save point cloud in a MeshLab-friendly format.
    Recommended: .ply (binary). Also supports .obj, .xyz, .pcd.
    """
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        print("No points to save.")
        return False

    if pcd.has_colors():
        col = np.asarray(pcd.colors)
        col = np.clip(col, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(col)

    ok = o3d.io.write_point_cloud(path, pcd, write_ascii=False, compressed=True)
    if ok:
        print(f"Saved point cloud -> {path}")
    else:
        print(f"Failed to save point cloud to {path}")
    return ok

def view_point_cloud(pcd, window_name="Open3D Viewer"):
    """
    Open an interactive Open3D window to inspect the reconstruction.
    """
    o3d.visualization.draw_geometries([pcd], window_name=window_name)

def save_and_view_point_cloud(pcd, path="my_reconstruction.ply", estimate_normals_first=True):
    """
    One-call convenience: (optionally) estimate normals, save PLY for MeshLab, and open Open3D viewer.
    """
    if estimate_normals_first:
        pcd = estimate_and_orient_normals(pcd)
    save_point_cloud(pcd, path=path)
    view_point_cloud(pcd, window_name=f"Open3D: {path}")
    return pcd


