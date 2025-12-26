import argparse
import os
from typing import Tuple

import numpy as np
import torch
import cv2
from tqdm import tqdm

from config import TrainingConfig
from fusion_with_fruit import backproject_fruit_points, fuse_fruit_views, save_fruit_csv, save_fruit_ply
from mtmvsnet_with_fruit import MTMVSNetWithFruit
from test_scan29_final import ViewCache, read_pair_file, prepare_batch


def save_depth_png(depth_map: np.ndarray, path: str) -> None:
    depth = depth_map.copy()
    if depth.max() > 0:
        depth = depth / depth.max()
    depth_uint8 = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    cv2.imwrite(path, depth_color)


def save_mask_png(mask: np.ndarray, path: str) -> None:
    mask_uint8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, mask_uint8)


def save_image_png(image_rgb: np.ndarray, path: str) -> None:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)


def main():
    parser = argparse.ArgumentParser(description="Combined depth + fruit inference")
    parser.add_argument("--scan_path", required=True, help="Path to scan folder with images/cams/pair.txt")
    parser.add_argument("--checkpoint", required=True, help="Path to MT-MVSNet backbone weights")
    parser.add_argument("--fruit_checkpoint", required=True, help="Path to fruit head weights")
    parser.add_argument("--output_ply", default="outputs/fruit_labeled.ply")
    parser.add_argument("--output_csv", default="outputs/fruit_labeled.csv")
    parser.add_argument("--max_refs", type=int, default=None)
    parser.add_argument("--save_examples_dir", default="outputs/fruit_examples")
    parser.add_argument("--num_examples", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(TrainingConfig.DEVICE if torch.cuda.is_available() else "cpu")

    model = MTMVSNetWithFruit(
        base_channels=TrainingConfig.BASE_CHANNELS,
        num_stages=TrainingConfig.NUM_STAGES,
    ).to(device)

    backbone_state = torch.load(args.checkpoint, map_location=device)
    if isinstance(backbone_state, dict) and "state_dict" in backbone_state:
        backbone_state = backbone_state["state_dict"]
    model.backbone.load_state_dict(backbone_state, strict=False)
    model.fruit_head.load_state_dict(torch.load(args.fruit_checkpoint, map_location=device))
    model.eval()

    pair_path = os.path.join(args.scan_path, "pair.txt")
    pairs = read_pair_file(pair_path)
    cache = ViewCache(args.scan_path)

    points_list = []
    colors_list = []
    labels_list = []
    examples_saved = 0
    os.makedirs(args.save_examples_dir, exist_ok=True)

    with torch.no_grad():
        for idx, pair in enumerate(tqdm(pairs, desc="Processing views")):
            if args.max_refs is not None and idx >= args.max_refs:
                break

            images, intrinsics, extrinsics, depth_values = prepare_batch(
                cache, pair["ref"], pair["src"]
            )
            images = images.to(device)
            intrinsics = intrinsics.to(device)
            extrinsics = extrinsics.to(device)
            depth_values = depth_values.to(device)

            outputs = model(images, intrinsics, extrinsics, depth_values, compute_depth=True)
            depth_map = outputs["depth"].squeeze(0).cpu().numpy()
            fruit_mask = outputs["fruit_mask"].squeeze(0).cpu().numpy()

            view_data = cache.get(pair["ref"])
            color_image = view_data["color_image"]
            intrinsic = view_data["intrinsic"]
            extrinsic = view_data["extrinsic_w2c"]

            points, colors, labels = backproject_fruit_points(
                depth_map,
                fruit_mask,
                intrinsic,
                extrinsic,
                image=color_image,
                mask_threshold=0.5,
            )

            if points.shape[0] > 0:
                points_list.append(points)
                colors_list.append(colors)
                labels_list.append(labels)

            if examples_saved < args.num_examples:
                image_path = os.path.join(args.save_examples_dir, f"image_{idx:04d}.png")
                mask_path = os.path.join(args.save_examples_dir, f"mask_{idx:04d}.png")
                depth_path = os.path.join(args.save_examples_dir, f"depth_{idx:04d}.png")
                save_image_png(color_image, image_path)
                save_mask_png(fruit_mask, mask_path)
                save_depth_png(depth_map, depth_path)
                examples_saved += 1

    fused_points, fused_colors, fused_labels = fuse_fruit_views(
        points_list, colors_list, labels_list, voxel_size=0.01
    )

    if fused_points.shape[0] == 0:
        raise RuntimeError("No fruit points were fused; check masks or depth maps.")

    save_fruit_ply(args.output_ply, fused_points, fused_colors, fused_labels)
    save_fruit_csv(args.output_csv, fused_points, fused_colors, fused_labels)
    print(f"Saved fruit point cloud to {args.output_ply}")
    print(f"Saved fruit point CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
