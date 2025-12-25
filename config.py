# config.py - Training Configuration for MT-MVSNet (Paper Settings)
import os

class TrainingConfig:
    """
    Training configuration for MT-MVSNet on DTU dataset
    Matches the paper specifications
    """
    
    # Dataset paths
    DTU_ROOT = "/root/DTU"  
    CHECKPOINT_DIR = "./checkpoints"  
    LOG_DIR = "./logs"  
    
    # Training parameters (AS PER PAPER)
    EPOCHS = 16  
    BATCH_SIZE = 1  # Paper uses batch size 6
    NUM_WORKERS = 2  
    
    # Learning rate schedule
    LEARNING_RATE = 0.001  
    LR_DECAY_EPOCHS = [6, 8]  # Paper: decay at 6th and 8th iterations
    LR_DECAY_RATE = 0.5  
    
    # Model parameters (PAPER SPECIFICATIONS)
    BASE_CHANNELS = 32  # Paper uses 32
    NUM_STAGES = 4  
    NUM_VIEWS = 5  # Paper uses 5 views (1 ref + 4 src)
    
    # Image dimensions (PAPER SPECIFICATIONS)
    IMG_HEIGHT = 512  # Paper uses 512
    IMG_WIDTH = 640   # Paper uses 640
    
    # Depth parameters (FIXED TO MATCH PAPER)
    NUM_DEPTHS = 32   # First stage: 32 depth hypotheses
    STAGE_DEPTH_COUNTS = [32, 16, 8, 4]  # CORRECTED from [64,32,16,8]
    DEPTH_INTERVALS = [0.125, 0.25, 0.5, 1.0]  # CORRECTED order
    
    # Loss weights (as per MT-MVSNet paper)
    STAGE_WEIGHTS = [1.0, 2.0, 4.0, 8.0]  
    FOCAL_LOSS_GAMMA = 0  # Paper uses γ=0 for DTU (simple scenarios)
    
    # Training schedule
    SAVE_FREQ = 2  
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
    FRUIT_LEARNING_RATE = 1e-4
    FRUIT_LAMBDA_CE = 1.0
    FRUIT_LAMBDA_DICE = 1.0
    FRUIT_CHECKPOINT_DIR = "./checkpoints_fruit"
    FRUIT_PRETRAINED = "./checkpoints/mtmvsnet_trained.pth"
    USE_FRUIT_MODEL = False
    
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
