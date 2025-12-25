import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainingConfig
from minneapple_dataset import MinneAppleDataset
from mtmvsnet_with_fruit import MTMVSNetWithFruit


def compute_metrics(mask_pred: torch.Tensor, mask_gt: torch.Tensor):
    pred_binary = (mask_pred > 0.5).float()
    gt_binary = mask_gt.float()

    tp = (pred_binary * gt_binary).sum().item()
    fp = (pred_binary * (1 - gt_binary)).sum().item()
    fn = ((1 - pred_binary) * gt_binary).sum().item()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return precision, recall, iou


def main():
    parser = argparse.ArgumentParser(description="Evaluate fruit segmentation head")
    parser.add_argument("--data_root", default=TrainingConfig.MINNEAPPLE_ROOT)
    parser.add_argument("--checkpoint", required=True, help="Path to fruit head weights")
    parser.add_argument("--batch_size", type=int, default=TrainingConfig.FRUIT_BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device(TrainingConfig.DEVICE if torch.cuda.is_available() else "cpu")

    dataset = MinneAppleDataset(
        args.data_root,
        split="val",
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

    precision_sum = 0.0
    recall_sum = 0.0
    iou_sum = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device).unsqueeze(1)
            masks = batch["mask"].to(device)

            outputs = model(images, compute_depth=False)
            precision, recall, iou = compute_metrics(outputs["fruit_mask"], masks)

            precision_sum += precision
            recall_sum += recall
            iou_sum += iou

    num_batches = max(len(loader), 1)
    print(f"Precision: {precision_sum / num_batches:.4f}")
    print(f"Recall: {recall_sum / num_batches:.4f}")
    print(f"IoU: {iou_sum / num_batches:.4f}")


if __name__ == "__main__":
    main()
