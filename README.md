# MT-MVSNet DTU Pipeline

## Precision Horticulture Application

This pipeline maps directly onto **Guyot and multi-leader apple orchard systems** — the dominant training systems used in commercial Pink Lady, Fuji, and Gala production. The architecture spans from raw camera capture to robotic harvesting readiness in a single automated sequence.

### How the pipeline maps to orchard structure

| Pipeline Stage | Technical Component | Orchard Meaning |
|---------------|--------------------|--------------------|
| Camera array capture | 3+ overlapping views per position | Images taken along the orchard row at ~0.5m intervals |
| MT-MVSNet depth reconstruction | 4-stage cascade, per-pixel refinement | Dense 3D geometry of the canopy and wire structure |
| Fruit segmentation | FPN head trained on MinneApple | Per-pixel apple detection fused into 3D |
| Point cloud fusion | Geometric consistency + outlier removal | Millimetre-scale fruit positions in world coordinates |
| Wire-zone classification | `zone_classifier.py` | Each fruit point assigned to one of 5 vertical Guyot wire zones (0–2.5m) |
| Yield prediction | `yield_predictor.py` | Zone counts → tons/ha via fruit weight × planting density |
| Orchard dashboard | `dashboard.py` | Real-time web view of zone distribution, yield, and 3D canopy |

### Full pipeline command sequence

```bash
# Step 1 — Combined depth + fruit inference on a new orchard scan
python inference_combined.py \
  --scan_path /path/to/orchard_scan \
  --checkpoint checkpoints/mtmvsnet_trained.pth \
  --fruit_checkpoint checkpoints_fruit/fruit_head_best.pth \
  --save_examples_dir outputs/fruit_examples
# Produces: outputs/fruit_labeled.ply + outputs/fruit_labeled.csv

# Step 2 — Classify fruit points into Guyot wire zones
python zone_classifier.py \
  --input outputs/fruit_labeled.csv \
  --output outputs/zone_fruit_counts.csv
# Produces: outputs/zone_fruit_counts.csv  (zone_id, fruit_count per wire)

# Step 3 — Predict yield in tons/ha and estimate season progression
python yield_predictor.py \
  --zone_csv outputs/zone_fruit_counts.csv \
  --trees_per_ha 2500 \
  --avg_fruit_weight_g 185 \
  --output outputs/yield_prediction.json
# Produces: outputs/yield_prediction.json

# Step 4 — Launch orchard dashboard
pip install fastapi uvicorn
python dashboard.py
# Open: http://localhost:8000
```

### Robotic harvesting readiness

The `zone_fruit_counts.csv` output directly encodes the vertical reach envelope required by a harvest arm:

- **Zone 0 (0–0.5m)**: Ground-level drops — typically already harvested or excluded
- **Zone 1–2 (0.5–1.5m)**: Primary production zone in Guyot systems — highest fruit density
- **Zone 3–4 (1.5–2.5m)**: Upper canopy — requires extended-reach end-effector

The 3D fruit positions in `fruit_labeled.ply` provide centroid coordinates at millimetre resolution, suitable for direct input to a 6-DOF arm path planner. Each fruit point includes the wire-zone label, enabling zone-aware harvesting strategies (e.g. prioritise Zone 2 first when arm reach is constrained).

### Precision orchard management

The yield predictor fits current scan fruit counts against Pink Lady reference data `[(2016, 23), (2017, 57), (2018, 77), (2019, 89), (2020, 131)] tons/ha` to estimate the current season's growth stage. Combined with per-zone distribution, this supports:

- **Thinning decisions**: if Zone 1–2 are overloaded relative to Zone 3–4, thin early to avoid biennial bearing
- **Irrigation scheduling**: high-yield zones may need targeted drip adjustment
- **Harvest timing**: integrate with fruit size models using 3D bounding sphere radius from the point cloud

---

## Overview
This repository contains an implementation of **MT-MVSNet**, a multi-stage multi-view stereo (MVS) network that fuses feature extractors, mobile transformer blocks, and edge-aware aggregation for depth estimation on the DTU dataset. The pipeline also includes a fruit segmentation head trained on MinneApple, sharing the same backbone features.

The depth estimation pipeline uses a **4-stage coarse-to-fine cascade** with per-pixel depth refinement (CasMVSNet-style), where each stage narrows the depth search range around the previous stage's prediction. Training combines focal loss with smooth L1 depth regression, cosine learning rate scheduling with warmup, and color augmentation.

## Repository structure

### Core model
- `mtmvsnet_model.py` — MT-MVSNet backbone with cascade-aware forward pass (feature-resolution depth for cascade, full-resolution for output)
- `feature_extraction.py` — FPN-based multi-scale feature extractor
- `feature_smooth_transition.py` — Feature Smooth Transition module
- `mobile_transformer_block.py` — Mobile Transformer blocks for cross-view attention
- `edge_attention.py` — Edge Attention Fusion across scales
- `mbps.py` — Multi-stage depth reasoning with per-pixel cascade refinement
- `fusion_builder.py` — Vectorized cost volume builder (batched over all depth hypotheses)
- `cost_regularization.py` — 3D U-Net cost regularization with gradient checkpointing

### Training
- `train.py` — Training loop with AMP, gradient accumulation, cosine LR + warmup
- `config.py` — All hyperparameters (depth counts, cascade scales, LR schedule, etc.)
- `losses.py` — Focal loss + smooth L1 depth loss
- `dtu_dataset.py` — DTU dataset loader with color augmentation

### Inference & evaluation
- `test_scan29_final.py` — Scan29 inference with multi-view fusion, depth filtering (median + bilateral), geometric consistency (round-trip reprojection), statistical outlier removal
- `eval_dtu.py` — DTU-style accuracy/completeness evaluator with percentile trimming

### Fruit segmentation
- `mtmvsnet_with_fruit.py` — Combined model (depth + fruit segmentation)
- `fruit_segmentation_head.py` — Lightweight segmentation head
- `train_fruit.py` — Fruit head training on MinneApple
- `eval_fruit.py` — Fruit segmentation evaluation
- `losses_fruit.py` — CE + Dice + multi-view consistency losses
- `minneapple_dataset.py` — MinneApple dataset loader
- `inference_combined.py` — Combined depth + fruit inference pipeline
- `fusion_with_fruit.py` — Fruit-aware point cloud fusion

### Data & outputs
- `checkpoints/` — Model weights (`mtmvsnet_trained.pth`, `fruit_head_best.pth`)
- `scan29/` — DTU Scan29 images, camera files, and GT point cloud
- `outputs/` — Generated point clouds, metrics, logs, and visualizations

## Requirements & setup
1. Create a Python environment (>=3.8) and install dependencies:
   ```bash
   pip install torch torchvision numpy scipy opencv-python tensorboard tqdm
   ```
2. Download the DTU training data so that `config.TrainingConfig.DTU_ROOT` points to the `mvs_training/dtu` folder containing `Rectified/`, `Depths/`, and `Cameras/`.
3. Place pretrained checkpoints in `checkpoints/` or train from scratch.

## Training MT-MVSNet

### Configuration
Key settings in `config.py`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `NUM_DEPTHS` | 48 | Stage 0 hypotheses (matches CasMVSNet/TransMVSNet) |
| `STAGE_DEPTH_COUNTS` | [48, 32, 8, 4] | Per-stage depth hypotheses |
| `DEPTH_CASCADE_SCALES` | [4.0, 2.0, 1.0] | Cascade interval refinement factors |
| `NUM_VIEWS` | 3 | 1 ref + 2 src (faster training) |
| `EPOCHS` | 16 | With cosine LR + 1 epoch warmup |
| `LEARNING_RATE` | 0.001 | Adam optimizer |
| `STAGE_WEIGHTS` | [1.0, 2.0, 4.0, 8.0] | Later stages weighted higher |
| `FOCAL_LOSS_GAMMA` | 0 | Simple CE for DTU |

### Run training
```bash
python train.py
```

Training features:
- **Cosine LR with warmup** (1 epoch linear warmup, then cosine decay)
- **Focal loss + smooth L1** combined at each stage
- **AMP** (mixed precision) for faster training on GPU
- **Gradient checkpointing** in the 3D U-Net to reduce VRAM usage
- **torch.compile** for kernel fusion (auto-enabled on CUDA)
- **Color augmentation** (brightness/contrast jitter across all views)
- **Gradient accumulation** (effective batch size 6)

Checkpoints are saved under `checkpoints/` and TensorBoard logs under `logs/`.

### Resume training
Set `TrainingConfig.RESUME` to a checkpoint path:
```python
RESUME = "./checkpoints/mtmvsnet_epoch_08.pth"
```

## Inference & evaluation (Scan29)

### Run inference
```bash
python test_scan29_final.py
```

The inference pipeline applies:
1. **Depth estimation** with 4-stage per-pixel cascade
2. **Confidence filtering** (threshold: 0.5)
3. **Median + bilateral filtering** on depth maps
4. **Round-trip geometric consistency** check (< 1% depth error AND < 1 pixel reprojection)
5. **Multi-view fusion** requiring 3+ consistent views
6. **Voxel downsampling** at 0.4mm resolution
7. **Statistical outlier removal** (20 neighbors, 2.0 std ratio)

Outputs:
- `outputs/scan29_clean.ply` — Fused point cloud
- `outputs/scan29_metrics.txt` — Accuracy, Completeness, Overall
- `outputs/logs/scan29_summary.txt` — Per-view statistics

### Evaluate any point cloud
```bash
python eval_dtu.py --pred outputs/scan29_clean.ply --gt scan29/scan29_gt.ply --output outputs/metrics.txt
```

Evaluation uses KD-tree Chamfer distance with 90th-percentile outlier trimming (up to 1M points).

## Fruit segmentation (MinneApple)

The fruit segmentation head reuses the MT-MVSNet backbone as a frozen feature extractor. **Train the DTU backbone first**, then train the fruit head.

### Train the fruit head
```bash
python train_fruit.py --data_root /path/to/MinneApple --checkpoint checkpoints/mtmvsnet_trained.pth
```

Features: cosine LR, progressive backbone unfreezing at epoch 6, CE + Dice + multi-view consistency loss.

### Evaluate
```bash
python eval_fruit.py --data_root /path/to/MinneApple --checkpoint checkpoints_fruit/fruit_head_best.pth
```

Reports IoU, Dice, pixel accuracy, precision/recall, and TP/TN/FP/FN.

### Combined inference (depth + fruit)
```bash
python inference_combined.py \
  --scan_path /path/to/scan \
  --checkpoint checkpoints/mtmvsnet_trained.pth \
  --fruit_checkpoint checkpoints_fruit/fruit_head_best.pth \
  --save_examples_dir outputs/fruit_examples \
  --num_examples 20
```

Produces fruit-labeled point clouds (PLY + CSV) and example visualizations.

## Performance optimizations

| Optimization | Location | Impact |
|-------------|----------|--------|
| Vectorized cost volume (no Python depth loop) | `fusion_builder.py` | ~10-20x faster cost volume |
| Precomputed matrix inversions | `fusion_builder.py` | Eliminates 92 redundant inversions/sample |
| Online variance (running sum/sum_sq) | `fusion_builder.py` | ~50% less memory than stacking views |
| Gradient checkpointing | `cost_regularization.py` | ~40% less VRAM |
| `torch.compile` | `train.py` | Kernel fusion on CUDA |
| `cudnn.benchmark` | `train.py` | Auto-tuned convolution kernels |
| 3 training views (not 5) | `config.py` | ~2x fewer warp operations |
| AMP (float16) | `train.py` | ~2x faster on V100/A100 |

Tested on Tesla V100 16GB. Expected training time: ~1-1.5 days for 16 epochs.

## Reproducibility
- All scripts seed Python, NumPy, and PyTorch RNGs for determinism
- Depth filtering parameters, fusion thresholds, and evaluation settings are documented in code
- Outputs are organized under `outputs/` for consistent reproduction between runs
