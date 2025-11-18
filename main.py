# main.py - FIXED to handle 3-tuples
import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from mtmvsnet_model import MTMVSNet
from data_loader import load_images_and_camera_params
from point_cloud_generator import create_point_cloud_from_depth_fixed, save_point_cloud, view_point_cloud

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    weights_path = "mtmvsnet_veg.pth"
    load_weights = True

    scene_path = "./veg"
    pair_file = os.path.join(scene_path, "pair.txt")

    print("Loading data with FIXED depth range...")
    images, intrinsics, extrinsics, depth_values = load_images_and_camera_params(scene_path, pair_file)
    
    print(f"Loaded dataset:")
    print(f"  - Images: {images.shape}")
    print(f"  - Cameras: {intrinsics.shape[0]} cameras")
    print(f"  - Depth range: [{depth_values.min():.1f}, {depth_values.max():.1f}]")

    print("Initializing FIXED MT-MVSNet...")
    model = MTMVSNet(base_channels=32).to(device)

    if load_weights and os.path.exists(weights_path):
        print(f"Loading trained weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("⚠️  Running with random initialization (untrained model)")

    model.eval()

    with torch.no_grad():
        # Add batch dimension
        if images.dim() == 4:
            images = images.unsqueeze(0)
        if intrinsics.dim() == 3:
            intrinsics = intrinsics.unsqueeze(0)
        if extrinsics.dim() == 3:
            extrinsics = extrinsics.unsqueeze(0)
        if depth_values.dim() == 1:
            depth_values = depth_values.unsqueeze(0)

        images = images.to(device)
        intrinsics = intrinsics.to(device)
        extrinsics = extrinsics.to(device)
        depth_values = depth_values.to(device)

        print(f"Input shapes:")
        print(f"  - Images: {images.shape}")
        print(f"  - Intrinsics: {intrinsics.shape}")
        print(f"  - Extrinsics: {extrinsics.shape}")
        print(f"  - Depth values: {depth_values.shape}")

        print("Running FIXED multi-stage inference...")
        # Model returns list of (depth_map, prob_volume, depth_values) 3-tuples
        stage_outputs = model(images, intrinsics, extrinsics, depth_values)
        
        print(f"Generated {len(stage_outputs)} stages")
        
        # FIXED: Extract just the depth maps from 3-tuples
        depth_maps = [depth for depth, prob, dvals in stage_outputs]
        
        # Use finest resolution depth map (last stage)
        final_depth = depth_maps[-1]  # [1, H, W]
        print(f"Final depth map shape: {final_depth.shape}")

    # Visualize
    depth_array = final_depth[0].cpu().numpy()
    print(f"FIXED depth statistics:")
    print(f"  - Min depth: {depth_array.min():.2f}")
    print(f"  - Max depth: {depth_array.max():.2f}")
    print(f"  - Mean depth: {depth_array.mean():.2f}")

    plt.figure(figsize=(10, 8))
    plt.imshow(depth_array, cmap='plasma')
    plt.colorbar(label='Depth')
    plt.title("MT-MVSNet Depth Estimation")
    plt.show()

    # Generate 3D point cloud
    print("Generating 3D point cloud...")
    output_path = "fixed_reconstruction.ply"
    
    intrinsic_array = intrinsics[0, 0].cpu().numpy()
    extrinsic_array = extrinsics[0, 0].cpu().numpy()
    image_array = images[0, 0].cpu().numpy()

    print("Creating point cloud...")
    pcd = create_point_cloud_from_depth_fixed(
        depth_array, intrinsic_array, extrinsic_array, image=image_array
    )

    print("Saving point cloud...")
    save_point_cloud(pcd, output_path)
    print(f"Reconstruction saved to {output_path}")

    print("Opening 3D viewer...")
    view_point_cloud(pcd)

if __name__ == "__main__":
    main()











