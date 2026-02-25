# config.py - Training Configuration for MT-MVSNet (Paper Settings)
import os

class TrainingConfig:
    """
    Training configuration for MT-MVSNet on DTU dataset
    Matches the paper specifications
    """
    
    # Dataset paths
    DTU_ROOT = "/root/DTU/mvs_training/dtu"  # expects Rectified/, Depths/, Cameras/ subdirs
    CHECKPOINT_DIR = "./checkpoints"  
    LOG_DIR = "./logs"  
    
    # Training parameters (AS PER PAPER)
    EPOCHS = 25  
    BATCH_SIZE = 2  # Paper uses batch size 6
    NUM_WORKERS = 2  
    
    # Learning rate schedule
    LEARNING_RATE = 0.001
    LR_DECAY_EPOCHS = [18]  # Paper: decay at 6th and 8th iterations
    LR_DECAY_RATE = 0.5
    USE_COSINE_LR = False        # Cosine annealing (CasMVSNet best practice)
    LR_WARMUP_EPOCHS = 0        # Linear warmup for 1 epoch
    
    # Model parameters (PAPER SPECIFICATIONS)
    BASE_CHANNELS = 32  # Paper uses 32
    NUM_STAGES = 4  
    NUM_VIEWS = 3  # 1 ref + 2 src during training (faster; CasMVSNet uses 3)
    
    # Image dimensions (PAPER SPECIFICATIONS)
    IMG_HEIGHT = 512  # Paper uses 512
    IMG_WIDTH = 640   # Paper uses 640
    
    # Depth parameters (FIXED TO MATCH PAPER)
    NUM_DEPTHS = 48   # First stage: 48 depth hypotheses (top methods all use 48)
    STAGE_DEPTH_COUNTS = [48, 32, 8, 4]  # Matches CasMVSNet/TransMVSNet convention
    DEPTH_INTERVALS = [0.125, 0.25, 0.5, 1.0]  # CORRECTED order
    DEPTH_CASCADE_SCALES = [4.0, 2.0, 1.0]
    
    # Loss weights (as per MT-MVSNet paper)
    STAGE_WEIGHTS = [1.0, 2.0, 4.0, 8.0]  
    FOCAL_LOSS_GAMMA = 0  # Paper uses γ=0 for DTU (simple scenarios)
    
    # Training schedule
    SAVE_FREQ = 1  
    LOG_FREQ = 50  
    VAL_FREQ = 1   
    
    # Optimization
    OPTIMIZER = "Adam"  
    WEIGHT_DECAY = 0.0001  
    GRAD_CLIP = 1.0  
    
    # Device
    DEVICE = "cuda"  # Paper uses GPU
    
    # Resume training
    RESUME = None  # Set to checkpoint path to resume

    # MinneApple fruit segmentation settings
    MINNEAPPLE_ROOT = "/root/MinneApple"
    FRUIT_EPOCHS = 20
    FRUIT_BATCH_SIZE = 4
    FRUIT_NUM_WORKERS = 2
    FRUIT_LEARNING_RATE = 3e-4
    FRUIT_LAMBDA_CE = 1.0
    FRUIT_LAMBDA_DICE = 1.0
    FRUIT_CHECKPOINT_DIR = "./checkpoints_fruit"
    FRUIT_PRETRAINED = "./checkpoints/mtmvsnet_best.pth"

    # Ablation flags
    FRUIT_USE_CLASS_WEIGHTS = True
    FRUIT_USE_AUGMENTATION = True
    FRUIT_USE_CONSISTENCY_LOSS = True
    FRUIT_USE_PROGRESSIVE_UNFREEZE = True
    FRUIT_USE_COSINE_LR = False

    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        
    @classmethod
    def get_model_save_path(cls, epoch):
        """Get path to save model at given epoch"""
        return os.path.join(cls.CHECKPOINT_DIR, f"mtmvsnet_epoch_{epoch:02d}.pth")
    
    @classmethod
    def get_best_model_path(cls):
        """Get path for best model"""
        return os.path.join(cls.CHECKPOINT_DIR, "mtmvsnet_best.pth")
    
    @classmethod
    def get_final_model_path(cls):
        """Get path for final trained model"""
        return os.path.join(cls.CHECKPOINT_DIR, "mtmvsnet_trained.pth")
        
    @classmethod
    def print_config(cls):
        """Print training configuration"""
        print("=" * 60)
        print("MT-MVSNet Training Configuration (Paper Settings)")
        print("=" * 60)
        print(f"DTU Dataset Path: {cls.DTU_ROOT}")
        print(f"Checkpoint Directory: {cls.CHECKPOINT_DIR}")
        print(f"Log Directory: {cls.LOG_DIR}")
        print("-" * 60)
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"LR Decay Epochs: {cls.LR_DECAY_EPOCHS}")
        print(f"Number of Views: {cls.NUM_VIEWS}")
        print(f"Image Size: {cls.IMG_HEIGHT} x {cls.IMG_WIDTH}")
        print(f"Base Channels: {cls.BASE_CHANNELS}")
        print(f"Stage Depth Counts: {cls.STAGE_DEPTH_COUNTS}")
        print(f"Depth Intervals: {cls.DEPTH_INTERVALS}")
        print(f"Focal Loss Gamma: {cls.FOCAL_LOSS_GAMMA}")
        print(f"Device: {cls.DEVICE}")
        print("=" * 60)
