import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainingConfig
from minneapple_dataset import MinneAppleDataset
from mtmvsnet_with_fruit import MTMVSNetWithFruit


def main():
    parser = argparse.ArgumentParser(description="Evaluate fruit segmentation head")
    parser.add_argument("--data_root", default=TrainingConfig.MINNEAPPLE_ROOT)
    parser.add_argument("--checkpoint", required=True, help="Path to fruit head weights")
    parser.add_argument("--batch_size", type=int, default=TrainingConfig.FRUIT_BATCH_SIZE)
    parser.add_argument("--split", default="val", help="Dataset split to evaluate (train/val/test)")
    parser.add_argument("--metrics_path", default="outputs/fruit_eval_metrics.txt")
    parser.add_argument("--extra_info_path", default="outputs/fruit_extra_info.txt")
    args = parser.parse_args()

    device = torch.device(TrainingConfig.DEVICE if torch.cuda.is_available() else "cpu")

    dataset = MinneAppleDataset(
        args.data_root,
        split=args.split,
        img_size=(TrainingConfig.IMG_WIDTH, TrainingConfig.IMG_HEIGHT),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=TrainingConfig.FRUIT_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = MTMVSNetWithFruit(
        base_channels=TrainingConfig.BASE_CHANNELS,
        num_stages=TrainingConfig.NUM_STAGES,
    ).to(device)
    model.fruit_head.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Bug 1 fix: accumulate global TP/FP/FN/TN instead of per-batch averages
    tp_sum = 0.0
    tn_sum = 0.0
    fp_sum = 0.0
    fn_sum = 0.0

    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Bug 2 fix: add .repeat() for NUM_VIEWS like training does
            images = batch["image"].to(device).unsqueeze(1).repeat(
                1, TrainingConfig.NUM_VIEWS, 1, 1, 1
            )
            masks = batch["mask"].to(device)

            outputs = model(images, compute_depth=False)
            pred_binary = (outputs["fruit_mask"] > 0.5).float()
            gt_binary = masks.float()

            tp_sum += (pred_binary * gt_binary).sum().item()
            fp_sum += (pred_binary * (1 - gt_binary)).sum().item()
            fn_sum += ((1 - pred_binary) * gt_binary).sum().item()
            tn_sum += ((1 - pred_binary) * (1 - gt_binary)).sum().item()

    # Compute final metrics from accumulated global totals
    precision = tp_sum / (tp_sum + fp_sum + 1e-6)
    recall = tp_sum / (tp_sum + fn_sum + 1e-6)
    iou = tp_sum / (tp_sum + fp_sum + fn_sum + 1e-6)
    dice = (2 * tp_sum) / (2 * tp_sum + fp_sum + fn_sum + 1e-6)
    pixel_acc = (tp_sum + tn_sum) / (tp_sum + tn_sum + fp_sum + fn_sum + 1e-6)

    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        f.write(f"IoU: {iou:.4f}\n")
        f.write(f"Dice: {dice:.4f}\n")
        f.write(f"Pixel Accuracy: {pixel_acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"TP: {tp_sum:.0f}, TN: {tn_sum:.0f}, FP: {fp_sum:.0f}, FN: {fn_sum:.0f}\n")

    elapsed = time.time() - start_time
    total_images = len(dataset)
    seconds_per_image = elapsed / max(total_images, 1)
    params_count = sum(p.numel() for p in model.fruit_head.parameters())
    max_memory_mb = 0.0
    if torch.cuda.is_available():
        max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    os.makedirs(os.path.dirname(args.extra_info_path), exist_ok=True)
    if os.path.exists(args.extra_info_path):
        with open(args.extra_info_path, "a", encoding="utf-8") as f:
            f.write(f"Inference speed (seconds per image): {seconds_per_image:.4f}\n")
    else:
        with open(args.extra_info_path, "w", encoding="utf-8") as f:
            f.write(f"Model parameters count: {params_count}\n")
            f.write(f"Memory usage (MB): {max_memory_mb:.2f}\n")
            f.write(f"Inference speed (seconds per image): {seconds_per_image:.4f}\n")

    print(f"Metrics saved to {args.metrics_path}")
    print(f"Extra info saved to {args.extra_info_path}")


if __name__ == "__main__":
    main()
