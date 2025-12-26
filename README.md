# MT-MVSNet DTU Pipeline

## Overview
This repository contains an implementation of **MT-MVSNet**, a multi-stage multi-view stereo (MVS) network that fuses feature extractors, mobile transformer blocks, and edge-aware aggregation for the DTU dataset. The core network (`mtmvsnet_model.py`) couples the feature pyramid built by `feature_extraction.py`, the Feature Smooth Transition module, transformer refinement, and the MBPS coarse-to-fine depth reasoning stages to deliver final depth maps for each view.  Training utilities (`train.py`, `config.py`, `dtu_dataset.py`) reproduce the paper’s image scaling (640×512), 1+4 view sampling, and focal loss supervision, while `test_scan29_final.py` provides a deterministic Scan29 fusion/evaluation pipeline that writes point clouds plus DTU accuracy/completeness metrics.  The standalone `eval_dtu.py` script can be reused to score any predicted ASCII PLY cloud against DTU ground truth using KD-tree queries.

## Repository structure
- `mtmvsnet_model.py`, `feature_extraction.py`, `feature_smooth_transition.py`, `mobile_transformer_block.py`, `edge_attention.py`, `mbps.py`: network definition and geometric reasoning blocks.
- `train.py`, `config.py`, `losses.py`, `dtu_dataset.py`: training loop, hyper-parameters, and the DTU loader that scales intrinsics whenever images are resized to 640×512.
- `test_scan29_final.py`, `fusion_correct.py`, `test_with_fusion.py`, `point_cloud_generator.py`: inference, multi-view fusion, and auxiliary experiments around Scan29.
- `eval_dtu.py`: accuracy/completeness evaluator for ASCII PLY predictions.
- `checkpoints/`: expected location of pretrained weights (e.g., `mtmvsnet_trained.pth`).
- `scan29/`: contains DTU Scan29 images, camera files, and the optional `scan29_gt.ply` used for evaluation.

## Requirements & setup
1. Create a Python environment (≥3.8) and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the DTU training data so that `config.TrainingConfig.DTU_ROOT` points to the `mvs_training/dtu` folder that ships with Rectified images, Depth maps, and Cameras.
3. Place pretrained checkpoints in `checkpoints/` or train the model from scratch (see below).

## Preparing the DTU dataset
The `DTUDataset` class expects the canonical DTU layout: `Rectified/`, `Depths/`, and `Cameras/`. Each sample packs 5 resized images (reference + 4 sources), scaled intrinsics, per-view extrinsics, and the ground-truth PFM depth map for the reference view.  Update `TrainingConfig.DTU_ROOT` if your data lives elsewhere.

## Training MT-MVSNet
`train.py` wires the dataset, MT-MVSNet backbone, focal loss, and optimizer into a multi-epoch trainer. Default hyper-parameters (batch size, depth intervals, number of stages) live in `config.py`, and `TrainingConfig.create_dirs()` ensures checkpoints/log directories exist before training begins.  Start training with:
```bash
python train.py
```
Checkpoints are saved under `checkpoints/` every few epochs, and TensorBoard logs appear in `logs/`.

## Running Scan29 inference & fusion
`test_scan29_final.py` loads a trained MT-MVSNet checkpoint, resizes Scan29 images to 640×512, rescales intrinsics, and enforces 1 reference + 4 source views per prediction. Depth maps are converted to meters, checked for multi-view geometric consistency (≥2 agreeing source views, ≤1% relative depth error), and fused using voxel downsampling (1.5 cm cells). The script emits:
- `outputs/scan29_clean.ply`: fused point cloud in meters.
- `outputs/scan29_metrics.txt`: DTU Accuracy, Completeness, Overall, and number of fused points.
- `outputs/logs/scan29_summary.txt`: per-view depth/consistency statistics.
Run inference with:
```bash
python test_scan29_final.py
```
Environment variable `DTU_GT_PLY` can override the default GT PLY path.

## Evaluating arbitrary point clouds
To compare any predicted PLY against DTU ground truth, call:
```bash
python eval_dtu.py --pred outputs/scan29_clean.ply --gt scan29/scan29_gt.ply --output outputs/scan29_metrics.txt
```
`eval_dtu.py` loads points, builds KD-trees in both directions, and reports Accuracy (reconstruction → GT), Completeness (GT → reconstruction), and their average.

## Fruit-aware MT-MVSNet (MinneApple)
The fruit segmentation head is trained separately from the depth backbone. The baseline MT-MVSNet weights remain unchanged and are reused for feature extraction only.
Evaluation is pixel-level segmentation (not instance detection), and depth values are predicted in meters then back-projected into world coordinates for fusion.

### Train the fruit head
```bash
python train_fruit.py --data_root /path/to/MinneApple --checkpoint checkpoints/mtmvsnet_trained.pth
```
This saves segmentation head checkpoints under `checkpoints_fruit/`, and logs a CSV of training/validation metrics to `outputs/fruit_training_metrics.csv` plus extra run info to `outputs/fruit_extra_info.txt`.

Example with explicit logging paths:
```bash
python train_fruit.py \
  --data_root /path/to/MinneApple \
  --checkpoint checkpoints/mtmvsnet_trained.pth \
  --log_csv outputs/fruit_training_metrics.csv \
  --extra_info_path outputs/fruit_extra_info.txt
```

### Evaluate the fruit head
```bash
python eval_fruit.py --data_root /path/to/MinneApple --checkpoint checkpoints_fruit/fruit_head_epoch_20.pth
```
This writes evaluation metrics (IoU, Dice, pixel accuracy, precision/recall, and TP/TN/FP/FN) to `outputs/fruit_eval_metrics.txt` and appends inference speed info to `outputs/fruit_extra_info.txt`.

Example with explicit output paths:
```bash
python eval_fruit.py \
  --data_root /path/to/MinneApple \
  --checkpoint checkpoints_fruit/fruit_head_epoch_20.pth \
  --metrics_path outputs/fruit_eval_metrics.txt \
  --extra_info_path outputs/fruit_extra_info.txt
```

### Combined inference (depth + fruit mask + fusion)
```bash
python inference_combined.py \
  --scan_path /path/to/scan \
  --checkpoint checkpoints/mtmvsnet_trained.pth \
  --fruit_checkpoint checkpoints_fruit/fruit_head_epoch_20.pth
```
The script produces a fruit-labeled point cloud in both PLY and CSV formats under `outputs/`. It also saves up to 20 example inputs, predicted masks, and depth visualizations to `outputs/fruit_examples/`.

Example with explicit example saving:
```bash
python inference_combined.py \
  --scan_path /path/to/scan \
  --checkpoint checkpoints/mtmvsnet_trained.pth \
  --fruit_checkpoint checkpoints_fruit/fruit_head_epoch_20.pth \
  --save_examples_dir outputs/fruit_examples \
  --num_examples 20 \
  --output_ply outputs/fruit_labeled.ply \
  --output_csv outputs/fruit_labeled.csv
```

## Reproducibility checklist
- All inference scripts seed Python, NumPy, and PyTorch RNGs for determinism, and log the depth range, valid/consistent pixels, and accepted points for every reference image.
- Depth values are treated in meters across geometric computations, and translations are converted from millimeters to meters before fusion.
- Outputs are organized under `outputs/` to keep checkpoints, metrics, and logs reproducible between runs.
