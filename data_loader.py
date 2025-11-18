# data_loader.py - FIXED resolution
import os
import cv2
import numpy as np
import torch

def load_images_and_camera_params(scene_path, pair_file_path):
    with open(pair_file_path) as f:
        num_viewpoints = int(f.readline())
        ref_idx = int(f.readline())
        src_line = f.readline().strip().split()
        num_src = int(src_line[0])
        src_indices = [int(src_line[i]) for i in range(1, len(src_line), 2)]
    
    all_indices = [ref_idx] + src_indices[:4]  # 1 ref + 4 source = 5 views
    
    imgs = []
    intrinsics = []
    extrinsics = []
    depth_min_vals = []
    depth_interval_vals = []
    
    # DTU original and target sizes
    ORIGINAL_HEIGHT = 1200
    ORIGINAL_WIDTH = 1600
    TARGET_HEIGHT = 512
    TARGET_WIDTH = 640
    
    scale_h = TARGET_HEIGHT / ORIGINAL_HEIGHT
    scale_w = TARGET_WIDTH / ORIGINAL_WIDTH
    
    for idx in all_indices:
        img_path = os.path.join(scene_path, "images", f"{idx:08d}.jpg")
        cam_path = os.path.join(scene_path, "cams", f"{idx:08d}_cam.txt")
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))  # FIXED: 640x512
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        imgs.append(img)
        
        with open(cam_path) as f:
            lines = f.readlines()
            extrinsic = np.array([list(map(float, line.split())) for line in lines[1:5]])
            intrinsic = np.array([list(map(float, line.split())) for line in lines[7:10]])
            
            # FIXED: Scale intrinsics to match resized image
            intrinsic[0, :] *= scale_w
            intrinsic[1, :] *= scale_h
            
            depth_line = lines[11].split()
            depth_min = float(depth_line[0])
            depth_interval = float(depth_line[1])
            depth_min_vals.append(depth_min)
            depth_interval_vals.append(depth_interval)
        
        intrinsics.append(torch.from_numpy(intrinsic).float())
        extrinsics.append(torch.from_numpy(extrinsic).float())
    
    imgs = torch.stack(imgs)
    intrinsics = torch.stack(intrinsics)
    extrinsics = torch.stack(extrinsics)
    
    depth_min = depth_min_vals[0]
    depth_interval = depth_interval_vals[0]
    depth_max = 935.0  # Full DTU range
    num_depths = int((depth_max - depth_min) / depth_interval) + 1
    depth_values = torch.tensor([depth_min + i * depth_interval for i in range(num_depths)])
    
    print(f"Loaded {len(all_indices)} views at {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"Depth range: [{depth_values[0]:.1f}, {depth_values[-1]:.1f}]mm")
    
    return imgs, intrinsics, extrinsics, depth_values
