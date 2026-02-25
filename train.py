import os
import time
import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mtmvsnet_model import MTMVSNet
from dtu_dataset import DTUDataset
from config import TrainingConfig
from losses import focal_loss_with_prob_volume, l1_depth_loss

GRAD_ACCUM_STEPS = 6  # Effective batch size = BATCH_SIZE * 6 = 6 (matches paper)

# Speed optimizations
torch.backends.cudnn.benchmark = True


def fmt_time(seconds):
    """Format seconds into human readable string."""
    if seconds < 0:
        return "--:--:--"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_gpu_stats(device):
    """Get GPU memory stats."""
    if device.type != "cuda":
        return ""
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    max_alloc = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    return f"GPU: {allocated:.1f}/{reserved:.1f}GB (peak {max_alloc:.1f}GB)"


class Trainer:
    def __init__(self, config):
        self.config = config
        config.create_dirs()

        self.device = torch.device(
            config.DEVICE if torch.cuda.is_available() else "cpu"
        )
        self.use_amp = False  # Disabled: per-pixel cascade causes float16 NaN
        print(f"Training on device: {self.device}")
        print(f"AMP enabled: {self.use_amp}")
        print(f"Gradient accumulation steps: {GRAD_ACCUM_STEPS}")

        # Model
        self.model = MTMVSNet(
            base_channels=config.BASE_CHANNELS,
            num_stages=config.NUM_STAGES,
        ).to(self.device)

        # Print model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

        # JIT compile for kernel fusion — significant speedup on CUDA
        if self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("torch.compile enabled (reduce-overhead mode)")
            except Exception as e:
                print(f"torch.compile not available: {e}")

        # Dataset (DTU-style)
        full_dataset = DTUDataset(
            root_dir=config.DTU_ROOT,
            num_views=config.NUM_VIEWS,
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            num_depths=config.NUM_DEPTHS,
            augment=True,
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
            pin_memory=self.device.type == "cuda",
            persistent_workers=config.NUM_WORKERS > 0,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            drop_last=False,
            pin_memory=self.device.type == "cuda",
            persistent_workers=config.NUM_WORKERS > 0,
        )

        # Optimizer & scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        if getattr(config, 'USE_COSINE_LR', False):
            warmup_epochs = getattr(config, 'LR_WARMUP_EPOCHS', 1)
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.EPOCHS - warmup_epochs,
                eta_min=config.LEARNING_RATE * 0.01,
            )
            if warmup_epochs > 0:
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                self.scheduler = optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs],
                )
            else:
                self.scheduler = cosine_scheduler
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=config.LR_DECAY_EPOCHS,
                gamma=config.LR_DECAY_RATE,
            )

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Logging
        self.writer = SummaryWriter(config.LOG_DIR)
        self.start_epoch = 0
        self.best_loss = float("inf")
        self.global_step = 0
        self.nan_count = 0  # Track NaN occurrences

        # Optionally resume
        if config.RESUME:
            self.load_checkpoint(config.RESUME)

        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Batches / epoch:   {len(self.train_loader)}")
        print(f"Effective batch size: {config.BATCH_SIZE * GRAD_ACCUM_STEPS}")

    # --------- one training epoch ---------
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        valid_batches = 0
        epoch_start = time.time()
        self.nan_count = 0

        self.optimizer.zero_grad()

        num_batches = len(self.train_loader)
        pbar = tqdm(
            enumerate(self.train_loader),
            total=num_batches,
            desc=f"Epoch {epoch:02d}/{self.config.EPOCHS}",
            bar_format="{l_bar}{bar:30}{r_bar}",
            dynamic_ncols=True,
        )

        running_loss = 0.0
        running_count = 0
        stage_loss_accum = [0.0] * self.config.NUM_STAGES

        for batch_idx, batch in pbar:
            batch_start = time.time()

            images = batch["images"].to(self.device)  # (B, V, 3, H, W)
            intrinsics = batch["intrinsics"].to(self.device)  # (B, V, 3, 3)
            extrinsics = batch["extrinsics"].to(self.device)  # (B, V, 4, 4)
            depth_values = batch["depth_values"].to(self.device)  # (B, D)
            depth_gt = batch["depth_gt"].to(self.device)  # (B, H, W)
            mask = batch["mask"].to(self.device)  # (B, H, W)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                stage_outputs = self.model(images, intrinsics, extrinsics, depth_values)

                total_loss = 0.0
                stage_losses = []
                for stage_idx, (depth_pred, prob_volume, stage_depth_values) in enumerate(
                    stage_outputs
                ):
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

                    stage_focal = focal_loss_with_prob_volume(
                        prob_volume,
                        depth_gt_resized,
                        stage_depth_values,
                        mask_resized,
                        gamma=self.config.FOCAL_LOSS_GAMMA,
                    )
                    stage_l1 = l1_depth_loss(depth_pred, depth_gt_resized, mask_resized)
                    stage_loss = stage_focal + stage_l1
                    weighted = stage_loss * self.config.STAGE_WEIGHTS[stage_idx]
                    total_loss += weighted
                    stage_losses.append(weighted)

                total_loss = total_loss / GRAD_ACCUM_STEPS

            # Check for NaN
            loss_val = total_loss.item() * GRAD_ACCUM_STEPS
            if math.isnan(loss_val) or math.isinf(loss_val):
                self.nan_count += 1
                self.optimizer.zero_grad()  # discard NaN gradients
                if self.nan_count <= 5 or self.nan_count % 100 == 0:
                    tqdm.write(f"  [!] NaN/Inf loss at batch {batch_idx} (count: {self.nan_count})")
                continue

            self.scaler.scale(total_loss).backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == num_batches:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            epoch_loss += loss_val
            valid_batches += 1
            running_loss += loss_val
            running_count += 1

            for i, sl in enumerate(stage_losses):
                stage_loss_accum[i] += sl.item()

            # Update progress bar
            batch_time = time.time() - batch_start
            elapsed = time.time() - epoch_start
            batches_done = batch_idx + 1
            batches_left = num_batches - batches_done
            eta_epoch = (elapsed / batches_done) * batches_left if batches_done > 0 else 0
            remaining_epochs = self.config.EPOCHS - epoch - 1
            eta_total = eta_epoch + (elapsed / batches_done) * num_batches * remaining_epochs if batches_done > 0 else 0

            avg_running = running_loss / running_count if running_count > 0 else 0
            lr = self.optimizer.param_groups[0]["lr"]

            # Build status string
            status = (
                f"loss={avg_running:.4f} | "
                f"lr={lr:.6f} | "
                f"ETA epoch={fmt_time(eta_epoch)} | "
                f"ETA total={fmt_time(eta_total)}"
            )
            if self.device.type == "cuda":
                alloc = torch.cuda.memory_allocated(self.device) / 1024 ** 3
                status += f" | VRAM={alloc:.1f}GB"
            if self.nan_count > 0:
                status += f" | NaN={self.nan_count}"

            pbar.set_postfix_str(status)

            # TensorBoard logging
            if batch_idx % self.config.LOG_FREQ == 0 and running_count > 0:
                self.writer.add_scalar("Loss/Total", avg_running, self.global_step)
                self.writer.add_scalar("LR", lr, self.global_step)
                if self.device.type == "cuda":
                    self.writer.add_scalar("GPU/Allocated_GB",
                                           torch.cuda.memory_allocated(self.device) / 1024**3,
                                           self.global_step)
                for i, sl in enumerate(stage_losses):
                    self.writer.add_scalar(f"Loss/Stage_{i}", sl.item(), self.global_step)

            self.global_step += 1

        pbar.close()

        avg_loss = epoch_loss / max(1, valid_batches)
        epoch_time = time.time() - epoch_start
        gpu_stats = get_gpu_stats(self.device) if self.device.type == "cuda" else ""

        # Per-stage loss summary
        stage_summary = " | ".join(
            f"S{i}={stage_loss_accum[i]/max(1,valid_batches):.4f}"
            for i in range(self.config.NUM_STAGES)
        )

        print(
            f"\n  Epoch {epoch} done in {fmt_time(epoch_time)} | "
            f"Loss: {avg_loss:.4f} | {stage_summary}"
        )
        if gpu_stats:
            print(f"  {gpu_stats}")
        if self.nan_count > 0:
            print(f"  WARNING: {self.nan_count}/{num_batches} batches had NaN loss "
                  f"({100*self.nan_count/num_batches:.1f}%)")

        return avg_loss

    # --------- validation ---------
    def validate(self, epoch: int) -> float:
        self.model.eval()
        val_loss = 0.0

        pbar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            desc=f"  Val {epoch:02d}",
            bar_format="{l_bar}{bar:30}{r_bar}",
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for batch_idx, batch in pbar:
                images = batch["images"].to(self.device)
                intrinsics = batch["intrinsics"].to(self.device)
                extrinsics = batch["extrinsics"].to(self.device)
                depth_values = batch["depth_values"].to(self.device)
                depth_gt = batch["depth_gt"].to(self.device)
                mask = batch["mask"].to(self.device)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    stage_outputs = self.model(images, intrinsics, extrinsics, depth_values)
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

                pbar.set_postfix_str(f"val_loss={val_loss/(batch_idx+1):.4f}")

        pbar.close()

        avg_val_loss = val_loss / max(1, len(self.val_loader))
        print(f"  Validation loss: {avg_val_loss:.4f}")
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
        print(f"  Checkpoint saved: {path}")
        if is_best:
            best_path = self.config.get_best_model_path()
            torch.save(ckpt, best_path)
            print(f"  Best model updated: {best_path}")

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
        print("\n" + "=" * 70)
        print("  MT-MVSNet Training")
        print("=" * 70)
        self.config.print_config()

        total_batches = len(self.train_loader) * (self.config.EPOCHS - self.start_epoch)
        print(f"\nTotal batches across all epochs: {total_batches:,}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(self.device)}")
            try:
                total_mem = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                print(f"GPU Memory: {total_mem:.1f} GB")
            except AttributeError:
                pass
        print("=" * 70)

        training_start = time.time()

        for epoch in range(self.start_epoch, self.config.EPOCHS):
            print(f"\n{'─' * 70}")
            print(f"  Epoch {epoch}/{self.config.EPOCHS} | "
                  f"Elapsed: {fmt_time(time.time() - training_start)} | "
                  f"Best val: {self.best_loss:.4f}")
            print(f"{'─' * 70}")

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

            # Epoch summary
            elapsed_total = time.time() - training_start
            epochs_done = epoch - self.start_epoch + 1
            epochs_left = self.config.EPOCHS - epoch - 1
            eta = (elapsed_total / epochs_done) * epochs_left if epochs_done > 0 else 0
            print(f"\n  >>> Epoch {epoch} summary: train={train_loss:.4f} val={val_loss:.4f} "
                  f"best={self.best_loss:.4f} | ETA: {fmt_time(eta)}")

        total_time = time.time() - training_start
        final_path = self.config.get_final_model_path()
        torch.save(self.model.state_dict(), final_path)
        print(f"\n{'=' * 70}")
        print(f"  Training finished in {fmt_time(total_time)}")
        print(f"  Final model: {final_path}")
        print(f"  Best val loss: {self.best_loss:.4f}")
        if self.device.type == "cuda":
            print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated(self.device)/1024**3:.2f} GB")
        print(f"{'=' * 70}")
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
