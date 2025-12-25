import argparse
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainingConfig
from losses_fruit import combined_fruit_loss
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

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        running_iou = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            images = images.unsqueeze(1)  # [B, 1, 3, H, W]

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
        print(
            f"Epoch {epoch}: loss={avg_loss:.4f} acc={avg_acc:.4f} iou={avg_iou:.4f}"
        )

        checkpoint_path = os.path.join(args.output_dir, f"fruit_head_epoch_{epoch:02d}.pth")
        torch.save(model.fruit_head.state_dict(), checkpoint_path)
        print(f"Saved fruit head checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
