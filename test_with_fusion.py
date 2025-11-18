import os
import cv2
import torch
import numpy as np
from mtmvsnet_model import MTMVSNet
from data_loader import load_images_and_camera_params

ORIGINAL_WIDTH = 1600
ORIGINAL_HEIGHT = 1200


def _maybe_scale_intrinsic(intrinsic, target_size=(640, 512)):
    """Scale camera intrinsics if needed after resizing to target size."""

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
            view_pairs.append({'ref': ref_idx, 'src': src_indices[:4]})  # Top 4 sources
    return view_pairs

def generate_depth_map(model, scan_path, ref_idx, src_indices, device):
    """Generate depth map for one reference view"""
    # Create temporary pair file for this view combination
    temp_pair = os.path.join(scan_path, "temp_pair.txt")
    with open(temp_pair, 'w') as f:
        f.write(f"{len(src_indices) + 1}\n")
        f.write(f"{ref_idx}\n")
        f.write(f"{len(src_indices)} " + " ".join([f"{idx} 1.0" for idx in src_indices]) + "\n")
    
    # Load data
    images, intrinsics, extrinsics, depth_values = load_images_and_camera_params(scan_path, temp_pair)
    
    # Run inference
    images = images.to(device).unsqueeze(0)
    intrinsics_batch = intrinsics.unsqueeze(0).to(device)
    extrinsics_batch = extrinsics.unsqueeze(0).to(device)
    depth_values = depth_values.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(images, intrinsics_batch, extrinsics_batch, depth_values)
    
    depth_map = outputs[3][0][0].cpu().numpy()  # Stage 3 depth
    prob_volume = outputs[3][1][0].cpu()  # Stage 3 probability
    confidence = prob_volume.max(dim=0)[0].numpy()  # Photometric confidence
    
    os.remove(temp_pair)
    return depth_map, confidence

def filter_depth_by_confidence(depth_map, confidence, threshold=0.8):
    """Filter depth map using photometric confidence"""
    mask = confidence > threshold
    filtered_depth = depth_map.copy()
    filtered_depth[~mask] = 0
    return filtered_depth, mask

def depth_to_points(depth_map, intrinsic, extrinsic, img_path, target_size=(640, 512)):
    """Convert depth map to 3D points with colors"""
    H, W = depth_map.shape
    
    # Load image for colors
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Valid pixels
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

def save_ply(filename, points, colors):
    """Save point cloud as PLY file"""
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

def test_scan_with_fusion(model, scan_path, output_dir, device, max_views=10):
    """Test scan with multi-view fusion (limited to max_views for speed)"""
    scan_name = os.path.basename(scan_path)
    print(f"\n{'='*70}")
    print(f"Processing {scan_name}")
    print(f"{'='*70}")
    
    pair_file = os.path.join(scan_path, "pair.txt")
    view_pairs = read_all_view_pairs(pair_file)
    
    print(f"Total available views: {len(view_pairs)}")
    print(f"Processing first {max_views} views for testing...")
    
    all_points = []
    all_colors = []
    
    TARGET_HEIGHT = 512
    TARGET_WIDTH = 640
    scale_h = TARGET_HEIGHT / ORIGINAL_HEIGHT
    scale_w = TARGET_WIDTH / ORIGINAL_WIDTH
    
    for i, pair in enumerate(view_pairs[:max_views]):
        ref_idx = pair['ref']
        src_indices = pair['src'][:4]
        
        print(f"\n[{i+1}/{max_views}] View {ref_idx} with sources {src_indices}")
        
        try:
            # Generate depth map
            depth_map, confidence = generate_depth_map(model, scan_path, ref_idx, src_indices, device)
            
            # Filter by confidence
            filtered_depth, mask = filter_depth_by_confidence(depth_map, confidence, threshold=0.5)
            
            valid_ratio = mask.sum() / mask.size
            print(f"  Depth range: [{depth_map.min():.1f}, {depth_map.max():.1f}]mm")
            print(f"  Valid pixels: {valid_ratio*100:.1f}%")
            
            if valid_ratio < 0.01:
                print(f"  Skipped (too few valid pixels)")
                continue
            
            # Load camera parameters
            cam_path = os.path.join(scan_path, "cams", f"{ref_idx:08d}_cam.txt")
            with open(cam_path) as f:
                lines = f.readlines()
                extrinsic = np.array([list(map(float, line.split())) for line in lines[1:5]])
                intrinsic = np.array([list(map(float, line.split())) for line in lines[7:10]])
                intrinsic[0, :] *= scale_w
                intrinsic[1, :] *= scale_h
            
            # Convert to points
            img_path = os.path.join(scan_path, "images", f"{ref_idx:08d}.jpg")
            points, colors = depth_to_points(filtered_depth, intrinsic, extrinsic, img_path)
            
            print(f"  Generated {len(points)} points")
            
            if len(points) > 0:
                all_points.append(points)
                all_colors.append(colors)
                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Concatenate all points
    if len(all_points) == 0:
        print("\nNo valid points generated!")
        return None, None
    
    final_points = np.concatenate(all_points, axis=0)
    final_colors = np.concatenate(all_colors, axis=0)
    
    print(f"\n{'='*70}")
    print(f"✅ Total points: {len(final_points)}")
    print(f"{'='*70}")
    
    # Save
    scan_output_dir = os.path.join(output_dir, scan_name)
    os.makedirs(scan_output_dir, exist_ok=True)
    ply_file = os.path.join(scan_output_dir, f"{scan_name}.ply")
    save_ply(ply_file, final_points, final_colors)
    
    print(f"✅ Saved: {ply_file}")
    
    return final_points, final_colors

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("MT-MVSNet Multi-View Fusion Test")
    print("="*70)
    print(f"Device: {device}\n")
    
    # Load model
    model = MTMVSNet(base_channels=32, num_stages=4).to(device)
    checkpoint = torch.load("./checkpoints/mtmvsnet_best.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded\n")
    
    # Test first scan
    scan_path = "/project/yuxiaojun/dtu_testing/dtu/scan1"
    output_dir = "./test_outputs_fusion"
    
    points, colors = test_scan_with_fusion(model, scan_path, output_dir, device, max_views=10)
    
    print("\n" + "="*70)
    print("✅ FUSION TEST COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
