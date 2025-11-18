# train.py - FIXED to use actual depth_values from MBPS
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
        
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {self.device}")
        
        self.model = MTMVSNet(
            base_channels=config.BASE_CHANNELS,
            num_stages=config.NUM_STAGES
        ).to(self.device)
        
        self.train_dataset = DTUDataset(
            root_dir=config.DTU_ROOT,
            num_views=config.NUM_VIEWS,
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH
        )
        
        # Split into train and validation
        train_size = int(0.9 * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            drop_last=False
        )
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.LR_DECAY_EPOCHS,
            gamma=config.LR_DECAY_RATE
        )
        
        self.writer = SummaryWriter(config.LOG_DIR)
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.global_step = 0
        
        if config.RESUME:
            self.load_checkpoint(config.RESUME)
            
        print(f"Training dataset: {len(self.train_dataset)} samples")
        print(f"Batches per epoch: {len(self.train_loader)}")
    
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(self.val_loader):
            batch_start = time.time()
            
            images = batch['images'].to(self.device)
            intrinsics = batch['intrinsics'].to(self.device)
            extrinsics = batch['extrinsics'].to(self.device)
            depth_values = batch['depth_values'].to(self.device)
            depth_gt = batch['depth_gt'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # FIXED: Now returns (depth, prob_volume, depth_values) 3-tuples
            stage_outputs = self.model(images, intrinsics, extrinsics, depth_values)
            
            total_loss = 0.0
            stage_losses = []
            
            # FIXED: Unpack 3-tuple and use actual depth_values
            for stage_idx, (depth_pred, prob_volume, stage_depth_values) in enumerate(stage_outputs):
                # Resize GT if needed
                if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
                    depth_gt_resized = torch.nn.functional.interpolate(
                        depth_gt.unsqueeze(1), 
                        size=depth_pred.shape[-2:], 
                        mode='nearest'
                    ).squeeze(1)
                    mask_resized = torch.nn.functional.interpolate(
                        mask.unsqueeze(1).float(), 
                        size=depth_pred.shape[-2:], 
                        mode='nearest'
                    ).squeeze(1)
                else:
                    depth_gt_resized = depth_gt
                    mask_resized = mask
                
                # FIXED: Use actual depth_values from MBPS (not approximated)
                stage_loss = focal_loss_with_prob_volume(
                    prob_volume, depth_gt_resized, stage_depth_values,
                    mask_resized, gamma=self.config.FOCAL_LOSS_GAMMA
                )
                
                weighted_loss = stage_loss * self.config.STAGE_WEIGHTS[stage_idx]
                stage_losses.append(weighted_loss)
                total_loss += weighted_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.optimizer.step()
            
            epoch_loss += total_loss.item()
            batch_time = time.time() - batch_start
            
            if batch_idx % self.config.LOG_FREQ == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:2d}/{self.config.EPOCHS} | "
                      f"Batch {batch_idx:3d}/{len(self.train_loader)} | "
                      f"Loss: {total_loss.item():.4f} | "
                      f"LR: {lr:.6f} | "
                      f"Time: {batch_time:.2f}s")
                
                self.writer.add_scalar('Loss/Total', total_loss.item(), self.global_step)
                self.writer.add_scalar('Learning_Rate', lr, self.global_step)
                for i, sl in enumerate(stage_losses):
                    self.writer.add_scalar(f'Loss/Stage_{i}', sl.item(), self.global_step)
            
            self.global_step += 1
        
        avg_loss = epoch_loss / len(self.train_loader)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} completed in {epoch_time:.1f}s | Avg Loss: {avg_loss:.4f}")
        return avg_loss
    
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_loader):
                if batch_idx >= 5:
                    break
                    
                images = batch['images'].to(self.device)
                intrinsics = batch['intrinsics'].to(self.device)
                extrinsics = batch['extrinsics'].to(self.device)
                depth_values = batch['depth_values'].to(self.device)
                depth_gt = batch['depth_gt'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                stage_outputs = self.model(images, intrinsics, extrinsics, depth_values)
                final_depth, _, _ = stage_outputs[-1]  # Unpack 3-tuple
                loss = torch.nn.functional.l1_loss(
                    final_depth * mask, depth_gt * mask, reduction='sum'
                ) / (mask.sum() + 1e-6)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        self.writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        return avg_val_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'global_step': self.global_step
        }
        checkpoint_path = self.config.get_model_save_path(epoch)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        if is_best:
            best_path = self.config.get_best_model_path()
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['loss']
        self.global_step = checkpoint.get('global_step', 0)
        print(f"Resumed from epoch {self.start_epoch}")
    
    def train(self):
        print("Starting MT-MVSNet training (Paper specification)...")
        self.config.print_config()
        
        for epoch in range(self.start_epoch, self.config.EPOCHS):
            print(f"\nEpoch {epoch}/{self.config.EPOCHS}")
            print("-" * 50)
            
            train_loss = self.train_epoch(epoch)
            
            if epoch % self.config.VAL_FREQ == 0:
                val_loss = self.validate(epoch)
            else:
                val_loss = float('inf')
            
            self.scheduler.step()
            
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                
            if epoch % self.config.SAVE_FREQ == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
        
        final_path = self.config.get_final_model_path()
        torch.save(self.model.state_dict(), final_path)
        print(f"\nTraining completed! Final model: {final_path}")
        self.writer.close()

def main():
    trainer = Trainer(TrainingConfig)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        trainer.save_checkpoint(trainer.start_epoch, trainer.best_loss, False)
        print("State saved")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()




