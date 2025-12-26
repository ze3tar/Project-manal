import argparse
import csv
import os
import time
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainingConfig
from losses_fruit import combined_fruit_loss, dice_loss
from minneapple_dataset import MinneAppleDataset
from mtmvsnet_with_fruit import MTMVSNetWithFruit


def load_checkpoint(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def compute_metrics(mask_pred: torch.Tensor, mask_gt: torch.Tensor) -> Tuple[float, float]:
    pred_binary = (mask_pred > 0.5).float()
    gt_binary = mask_gt.float()
    correct = (pred_binary == gt_binary).float().mean().item()
    intersection = (pred_binary * gt_binary).sum().item()
    union = pred_binary.sum().item() + gt_binary.sum().item() - intersection
    iou = intersection / (union + 1e-6)
    return correct, iou


def main():
    parser = argparse.ArgumentParser(description="Train fruit segmentation head on MinneApple")
    parser.add_argument("--data_root", default=TrainingConfig.MINNEAPPLE_ROOT)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.FRUIT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=TrainingConfig.FRUIT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=TrainingConfig.FRUIT_LEARNING_RATE)
    parser.add_argument("--lambda_ce", type=float, default=TrainingConfig.FRUIT_LAMBDA_CE)
    parser.add_argument("--lambda_dice", type=float, default=TrainingConfig.FRUIT_LAMBDA_DICE)
    parser.add_argument("--checkpoint", default=TrainingConfig.FRUIT_PRETRAINED)
    parser.add_argument("--output_dir", default=TrainingConfig.FRUIT_CHECKPOINT_DIR)
    parser.add_argument("--log_csv", default="outputs/fruit_training_metrics.csv")
    parser.add_argument("--extra_info_path", default="outputs/fruit_extra_info.txt")
    args = parser.parse_args()

    device = torch.device(TrainingConfig.DEVICE if torch.cuda.is_available() else "cpu")

    dataset = MinneAppleDataset(
        args.data_root,
        split="train",
        img_size=(TrainingConfig.IMG_WIDTH, TrainingConfig.IMG_HEIGHT),
        augment=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=TrainingConfig.FRUIT_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_dataset = MinneAppleDataset(
        args.data_root,
        split="val",
        img_size=(TrainingConfig.IMG_WIDTH, TrainingConfig.IMG_HEIGHT),
        augment=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=TrainingConfig.FRUIT_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = MTMVSNetWithFruit(
        base_channels=TrainingConfig.BASE_CHANNELS,
        num_stages=TrainingConfig.NUM_STAGES,
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        state = load_checkpoint(args.checkpoint)
        model.backbone.load_state_dict(state, strict=False)
        print(f"Loaded backbone weights from {args.checkpoint}")

    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.fruit_head.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.extra_info_path), exist_ok=True)

    with open(args.log_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "iou", "dice"])

    start_time = time.time()
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        running_iou = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            images = images.unsqueeze(1).repeat(1, TrainingConfig.NUM_VIEWS, 1, 1, 1)

            outputs = model(images, compute_depth=False)
            logits = outputs["fruit_logits"]
            loss = combined_fruit_loss(logits, masks, args.lambda_ce, args.lambda_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            acc, iou = compute_metrics(outputs["fruit_mask"].detach(), masks)
            running_acc += acc
            running_iou += iou

        num_batches = max(len(loader), 1)
        avg_loss = running_loss / num_batches
        avg_acc = running_acc / num_batches
        avg_iou = running_iou / num_batches

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device).unsqueeze(1).repeat(
                    1, TrainingConfig.NUM_VIEWS, 1, 1, 1
                )
                masks = batch["mask"].to(device)
                outputs = model(images, compute_depth=False)
                logits = outputs["fruit_logits"]
                loss = combined_fruit_loss(logits, masks, args.lambda_ce, args.lambda_dice)
                val_loss += loss.item()
                val_dice += (1.0 - dice_loss(logits, masks)).item()
                _, iou = compute_metrics(outputs["fruit_mask"], masks)
                val_iou += iou

        val_batches = max(len(val_loader), 1)
        avg_val_loss = val_loss / val_batches
        avg_val_iou = val_iou / val_batches
        avg_val_dice = val_dice / val_batches

        print(
            f"Epoch {epoch}: loss={avg_loss:.4f} acc={avg_acc:.4f} "
            f"iou={avg_iou:.4f} val_loss={avg_val_loss:.4f} val_iou={avg_val_iou:.4f}"
        )

        checkpoint_path = os.path.join(args.output_dir, f"fruit_head_epoch_{epoch:02d}.pth")
        torch.save(model.fruit_head.state_dict(), checkpoint_path)
        print(f"Saved fruit head checkpoint to {checkpoint_path}")

        with open(args.log_csv, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch, avg_loss, avg_val_loss, avg_val_iou, avg_val_dice])

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch

    total_time = time.time() - start_time
    total_minutes = total_time / 60.0
    total_hours = total_minutes / 60.0
    params_count = sum(p.numel() for p in model.fruit_head.parameters())
    max_memory_mb = 0.0
    if torch.cuda.is_available():
        max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    with open(args.extra_info_path, "w", encoding="utf-8") as f:
        f.write(f"Best epoch number: {best_epoch}\n")
        f.write(f"Total training time (hours): {total_hours:.2f}\n")
        f.write(f"Total training time (minutes): {total_minutes:.2f}\n")
        f.write(f"Model parameters count: {params_count}\n")
        f.write(f"Memory usage (MB): {max_memory_mb:.2f}\n")


if __name__ == "__main__":
    main()
