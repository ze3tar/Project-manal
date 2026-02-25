# dtu_dataset.py - DTU Training Dataset Loader
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import re

class DTUDataset(Dataset):
    """
    DTU Dataset for Training MT-MVSNet
    
    Loads images, depth maps, and camera parameters from the DTU training dataset
    for training the MT-MVSNet model.
    """
    
    def __init__(self, root_dir, num_views=5, img_height=512, img_width=640,
                 num_depths=48, augment=True):
        """
        Args:
            root_dir: Path to 'mvs_training/dtu/' folder
            num_views: Number of views to use for training (reference + source views)
            img_height: Target image height
            img_width: Target image width
            num_depths: Number of depth hypotheses for stage 0
            augment: Whether to apply color augmentation
        """
        self.root_dir = root_dir
        self.num_views = num_views
        self.img_height = img_height
        self.img_width = img_width
        self.num_depths = num_depths
        self.augment = augment
        self._scale_w = self.img_width / 1600.0
        self._scale_h = self.img_height / 1200.0
        
        # Dataset folders
        self.rectified_dir = os.path.join(root_dir, 'Rectified')
        self.depths_dir = os.path.join(root_dir, 'Depths')
        self.cameras_dir = os.path.join(root_dir, 'Cameras')
        
        # Find matching scenes (have both images and depth maps)
        self.scenes = self._find_matching_scenes()
        print(f"Found {len(self.scenes)} matching scenes for training")
        
        # Create training samples
        self.samples = self._create_training_samples()
        print(f"Created {len(self.samples)} training samples")
        
    def _find_matching_scenes(self):
        """Find scenes that exist in both Rectified and Depths folders"""
        rectified_scenes = set()
        depths_scenes = set()
        
        # Get scenes from Rectified folder
        for item in os.listdir(self.rectified_dir):
            if os.path.isdir(os.path.join(self.rectified_dir, item)):
                rectified_scenes.add(item)
        
        # Get scenes from Depths folder  
        for item in os.listdir(self.depths_dir):
            if os.path.isdir(os.path.join(self.depths_dir, item)):
                depths_scenes.add(item)
        
        # Find intersection
        matching_scenes = sorted(list(rectified_scenes.intersection(depths_scenes)))
        
        print(f"Rectified scenes: {len(rectified_scenes)}")
        print(f"Depths scenes: {len(depths_scenes)}")
        print(f"Matching scenes: {matching_scenes}")
        
        return matching_scenes
    
    def _create_training_samples(self):
        """Create list of training samples (scene, reference_view, source_views)"""
        samples = []
        
        for scene in self.scenes:
            scene_dir = os.path.join(self.rectified_dir, scene)
            
            # Get all image files in scene
            image_files = []
            for f in os.listdir(scene_dir):
                if f.endswith('.png'):
                    image_files.append(f)
            
            image_files.sort()
            
            # Extract view indices from filenames (rect_001_X_r5000.png -> X)
            view_indices = []
            for img_file in image_files:
                match = re.search(r'rect_\d+_(\d+)_r\d+\.png', img_file)
                if match:
                    view_idx = int(match.group(1))
                    view_indices.append(view_idx)
            
            view_indices.sort()
            
            # Create training samples by selecting reference + source views
            # Use every 3rd view as reference to get good baseline
            for i in range(0, len(view_indices), 3):
                if i + self.num_views - 1 < len(view_indices):
                    ref_view = view_indices[i]
                    src_views = view_indices[i+1:i+self.num_views]
                    
                    samples.append({
                        'scene': scene,
                        'ref_view': ref_view,
                        'src_views': src_views,
                        'all_views': [ref_view] + src_views
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a training sample"""
        sample = self.samples[idx]
        scene = sample['scene']
        all_views = sample['all_views']
        
        # Load images
        images = []
        for view_idx in all_views:
            img_path = os.path.join(
                self.rectified_dir, scene, 
                f'rect_001_{view_idx:03d}_r5000.png'
            )
            
            if not os.path.exists(img_path):
                # Try different filename patterns if needed
                img_path = os.path.join(
                    self.rectified_dir, scene,
                    f'rect_001_{view_idx}_r5000.png'
                )
            
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {img_path}")
                
            # Resize and normalize
            img = cv2.resize(img, (self.img_width, self.img_height))
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            images.append(img)
        
        images = torch.stack(images)  # [N, 3, H, W]

        # Color augmentation: apply same jitter to all views for consistency
        if self.augment:
            brightness = np.random.uniform(0.8, 1.2)
            contrast = np.random.uniform(0.8, 1.2)
            images = images * brightness
            mean = images.mean(dim=(2, 3), keepdim=True)
            images = (images - mean) * contrast + mean
            images = images.clamp(0.0, 1.0)
        
        # Load reference depth map
        ref_view = sample['ref_view']
        depth_path = os.path.join(
            self.depths_dir, scene,
            f'depth_map_{ref_view:04d}.pfm'
        )
        
        if not os.path.exists(depth_path):
            # Try different filename patterns
            depth_path = os.path.join(
                self.depths_dir, scene,
                f'depth_map_{ref_view:03d}.pfm'
            )
        
        depth_map = self._load_pfm(depth_path)
        
        # Resize depth map to match image resolution
        depth_map = cv2.resize(depth_map, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        depth_map = torch.from_numpy(depth_map).float()
        
        # Load camera parameters
        intrinsics, extrinsics = self._load_camera_params(all_views)
        
        # Create depth value range
        depth_min = float(depth_map[depth_map > 0].min()) if (depth_map > 0).any() else 0.1
        depth_max = float(depth_map.max()) if depth_map.max() > 0 else 100.0
        
        # Ensure valid range
        if depth_min >= depth_max:
            depth_min = 0.1
            depth_max = 100.0
            
        depth_values = torch.linspace(depth_min, depth_max, self.num_depths)
        
        # Create mask for valid depth values
        mask = (depth_map > 0).float()
        
        return {
            'images': images,           # [N, 3, H, W]
            'intrinsics': intrinsics,   # [N, 3, 3]  
            'extrinsics': extrinsics,   # [N, 4, 4]
            'depth_values': depth_values, # [64]
            'depth_gt': depth_map,      # [H, W]
            'mask': mask,               # [H, W]
            'scene': scene,
            'ref_view': ref_view
        }
    
    def _load_pfm(self, filepath):
        """Load PFM (Portable Float Map) depth file"""
        try:
            with open(filepath, 'rb') as f:
                # Read header
                header = f.readline().decode('utf-8').strip()
                if header != 'Pf':
                    raise ValueError(f"Invalid PFM header: {header}")
                
                # Read dimensions
                dims = f.readline().decode('utf-8').strip().split()
                width, height = int(dims[0]), int(dims[1])
                
                # Read scale factor
                scale = float(f.readline().decode('utf-8').strip())
                
                # Read binary data
                data = np.frombuffer(f.read(), dtype=np.float32)
                data = data.reshape((height, width))
                
                # Flip vertically (PFM convention)
                data = np.flipud(data)
                
                return data
                
        except Exception as e:
            print(f"Error loading PFM file {filepath}: {e}")
            # Return dummy depth map
            return np.ones((self.img_height, self.img_width), dtype=np.float32) * 10.0
    
    def _load_camera_params(self, view_indices):
        """Load camera intrinsics and extrinsics for given views"""
        intrinsics = []
        extrinsics = []
        
        for view_idx in view_indices:
            cam_file = os.path.join(self.cameras_dir, f'{view_idx:08d}_cam.txt')
            
            if not os.path.exists(cam_file):
                # Try train subfolder
                cam_file = os.path.join(self.cameras_dir, 'train', f'{view_idx:08d}_cam.txt')
            
            if not os.path.exists(cam_file):
                print(f"Warning: Camera file not found for view {view_idx}")
                # Create dummy camera parameters
                intrinsic = np.eye(3, dtype=np.float32) * 500
                intrinsic[2, 2] = 1
                extrinsic = np.eye(4, dtype=np.float32)
            else:
                try:
                    with open(cam_file, 'r') as f:
                        lines = f.readlines()
                        
                    # Parse extrinsic matrix (lines 1-4)
                    extrinsic = np.array([
                        [float(x) for x in lines[1].split()],
                        [float(x) for x in lines[2].split()],
                        [float(x) for x in lines[3].split()],
                        [float(x) for x in lines[4].split()]
                    ], dtype=np.float32)
                    
                    # Parse intrinsic matrix (lines 7-9) 
                    intrinsic = np.array([
                        [float(x) for x in lines[7].split()],
                        [float(x) for x in lines[8].split()],
                        [float(x) for x in lines[9].split()]
                    ], dtype=np.float32)
                    
                except Exception as e:
                    print(f"Error parsing camera file {cam_file}: {e}")
                    # Create dummy parameters
                    intrinsic = np.eye(3, dtype=np.float32) * 500
                    intrinsic[2, 2] = 1
                    extrinsic = np.eye(4, dtype=np.float32)
            
            intrinsic = intrinsic.copy()
            intrinsic[0, 0] *= self._scale_w
            intrinsic[1, 1] *= self._scale_h
            intrinsic[0, 2] *= self._scale_w
            intrinsic[1, 2] *= self._scale_h
            intrinsics.append(torch.from_numpy(intrinsic))
            extrinsics.append(torch.from_numpy(extrinsic))
        
        return torch.stack(intrinsics), torch.stack(extrinsics)
