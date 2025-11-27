import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mtmvsnet_model import MTMVSNet
from dtu_dataset import DTUDataset
from config import TrainingConfig
from losses import focal_loss_with_prob_volume


class Trainer:
    def __init__(self, config):
        self.config = config
        config.create_dirs()

        self.device = torch.device(
            config.DEVICE if torch.cuda.is_available() else "cpu"
        )
        print(f"Training on device: {self.device}")

        # Model
        self.model = MTMVSNet(
            base_channels=config.BASE_CHANNELS,
            num_stages=config.NUM_STAGES,
        ).to(self.device)

        # Dataset (DTU-style)
        full_dataset = DTUDataset(
            root_dir=config.DTU_ROOT,
            num_views=config.NUM_VIEWS,
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
        )

        # Simple train/val split
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            drop_last=False,
        )

        # Optimizer & scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.LR_DECAY_EPOCHS,
            gamma=config.LR_DECAY_RATE,
        )

        # Logging
        self.writer = SummaryWriter(config.LOG_DIR)
        self.start_epoch = 0
        self.best_loss = float("inf")
        self.global_step = 0

        # Optionally resume
        if config.RESUME:
            self.load_checkpoint(config.RESUME)

        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Batches / epoch:   {len(self.train_loader)}")

    # --------- one training epoch ---------
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            batch_start = time.time()

            images = batch["images"].to(self.device)  # (B, V, 3, H, W)
            intrinsics = batch["intrinsics"].to(self.device)  # (B, V, 3, 3)
            extrinsics = batch["extrinsics"].to(self.device)  # (B, V, 4, 4)
            depth_values = batch["depth_values"].to(self.device)  # (B, D)
            depth_gt = batch["depth_gt"].to(self.device)  # (B, H, W)
            mask = batch["mask"].to(self.device)  # (B, H, W)

            self.optimizer.zero_grad()

            # Model should return a list of (depth, prob_volume, depth_values_stage)
            stage_outputs = self.model(images, intrinsics, extrinsics, depth_values)

            total_loss = 0.0
            stage_losses = []
            for stage_idx, (depth_pred, prob_volume, stage_depth_values) in enumerate(
                stage_outputs
            ):
                # If resolutions differ, resize GT / mask to prediction size
                if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
                    depth_gt_resized = torch.nn.functional.interpolate(
                        depth_gt.unsqueeze(1),
                        size=depth_pred.shape[-2:],
                        mode="nearest",
                    ).squeeze(1)
                    mask_resized = torch.nn.functional.interpolate(
                        mask.unsqueeze(1).float(),
                        size=depth_pred.shape[-2:],
                        mode="nearest",
                    ).squeeze(1)
                else:
                    depth_gt_resized = depth_gt
                    mask_resized = mask

                # *** IMPORTANT ***
                # Use the per-stage depth_values (from MBPS) together with the prob_volume
                stage_loss = focal_loss_with_prob_volume(
                    prob_volume,
                    depth_gt_resized,
                    stage_depth_values,
                    mask_resized,
                    gamma=self.config.FOCAL_LOSS_GAMMA,
                )
                weighted = stage_loss * self.config.STAGE_WEIGHTS[stage_idx]
                total_loss += weighted
                stage_losses.append(weighted)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.optimizer.step()

            epoch_loss += total_loss.item()
            batch_time = time.time() - batch_start
            if batch_idx % self.config.LOG_FREQ == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:02d}/{self.config.EPOCHS} | "
                    f"Batch {batch_idx:03d}/{len(self.train_loader)} | "
                    f"Loss {total_loss.item():.4f} | LR {lr:.6f} | "
                    f"Time {batch_time:.2f}s"
                )
                self.writer.add_scalar("Loss/Total", total_loss.item(), self.global_step)
                self.writer.add_scalar("LR", lr, self.global_step)
                for i, st_loss in enumerate(stage_losses):
                    self.writer.add_scalar(
                        f"Loss/Stage_{i}", st_loss.item(), self.global_step
                    )
            self.global_step += 1

        avg_loss = epoch_loss / max(1, len(self.train_loader))
        print(
            f"Epoch {epoch} finished in {time.time() - epoch_start:.1f}s "
            f"| Train Loss: {avg_loss:.4f}"
        )
        return avg_loss

    # --------- validation ---------
    def validate(self, epoch: int) -> float:
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                images = batch["images"].to(self.device)
                intrinsics = batch["intrinsics"].to(self.device)
                extrinsics = batch["extrinsics"].to(self.device)
                depth_values = batch["depth_values"].to(self.device)
                depth_gt = batch["depth_gt"].to(self.device)
                mask = batch["mask"].to(self.device)

                stage_outputs = self.model(images, intrinsics, extrinsics, depth_values)
                # Just evaluate final stage depth with L1 inside mask
                final_depth, _, _ = stage_outputs[-1]
                if final_depth.shape[-2:] != depth_gt.shape[-2:]:
                    depth_gt_resized = torch.nn.functional.interpolate(
                        depth_gt.unsqueeze(1),
                        size=final_depth.shape[-2:],
                        mode="nearest",
                    ).squeeze(1)
                    mask_resized = torch.nn.functional.interpolate(
                        mask.unsqueeze(1).float(),
                        size=final_depth.shape[-2:],
                        mode="nearest",
                    ).squeeze(1)
                else:
                    depth_gt_resized = depth_gt
                    mask_resized = mask

                loss = torch.nn.functional.l1_loss(
                    final_depth * mask_resized,
                    depth_gt_resized * mask_resized,
                    reduction="sum",
                ) / (mask_resized.sum() + 1e-6)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(self.val_loader))
        print(f"Validation loss: {avg_val_loss:.4f}")
        self.writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        return avg_val_loss

    # --------- checkpoint helpers ---------
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "global_step": self.global_step,
        }
        path = self.config.get_model_save_path(epoch)
        torch.save(ckpt, path)
        print(f"Checkpoint saved: {path}")
        if is_best:
            best_path = self.config.get_best_model_path()
            torch.save(ckpt, best_path)
            print(f"Best model updated: {best_path}")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return

        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_loss = ckpt.get("best_loss", float("inf"))
        self.global_step = ckpt.get("global_step", 0)
        print(f"Resumed training from epoch {self.start_epoch}")

    # --------- main training loop ---------
    def train(self):
        print("Starting MT-MVSNet training...")
        self.config.print_config()
        for epoch in range(self.start_epoch, self.config.EPOCHS):
            print("\n" + "-" * 60)
            print(f"Epoch {epoch}/{self.config.EPOCHS}")
            train_loss = self.train_epoch(epoch)

            if epoch % self.config.VAL_FREQ == 0:
                val_loss = self.validate(epoch)
            else:
                val_loss = float("inf")

            self.scheduler.step()

            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss

            if epoch % self.config.SAVE_FREQ == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)

        final_path = self.config.get_final_model_path()
        torch.save(self.model.state_dict(), final_path)
        print(f"\nTraining finished. Final model weights: {final_path}")
        self.writer.close()


def main():
    trainer = Trainer(TrainingConfig)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        trainer.save_checkpoint(trainer.start_epoch, trainer.best_loss, is_best=False)
        print("Checkpoint saved on interrupt.")
    except Exception as e:
        print("\nTraining crashed with an exception:")
        print(e)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
