import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from config import TrainingConfig
from fusion_with_fruit import backproject_fruit_points, fuse_fruit_views, save_fruit_csv, save_fruit_ply
from mtmvsnet_with_fruit import MTMVSNetWithFruit
from test_scan29_final import ViewCache, read_pair_file, prepare_batch


def main():
    parser = argparse.ArgumentParser(description="Combined depth + fruit inference")
    parser.add_argument("--scan_path", required=True, help="Path to scan folder with images/cams/pair.txt")
    parser.add_argument("--checkpoint", required=True, help="Path to MT-MVSNet backbone weights")
    parser.add_argument("--fruit_checkpoint", required=True, help="Path to fruit head weights")
    parser.add_argument("--output_ply", default="outputs/fruit_labeled.ply")
    parser.add_argument("--output_csv", default="outputs/fruit_labeled.csv")
    parser.add_argument("--max_refs", type=int, default=None)
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
