# eval_dtu.py
"""
Simple DTU-style point cloud evaluation (ASCII PLY only, XYZ only).

- Loads two ASCII PLY point clouds (predicted and ground truth)
- Keeps only XYZ coordinates
- Cleans NaN / Inf
- Randomly downsamples to at most `max_points` points for each cloud
- Computes:
    * Accuracy:     mean distance pred → nearest gt
    * Completeness: mean distance gt   → nearest pred
    * Overall:      (Accuracy + Completeness) / 2
"""

import argparse
import os
from typing import Dict

import numpy as np
from scipy.spatial import cKDTree


def load_ascii_ply_xyz(path: str) -> np.ndarray:
    """
    Load XYZ points from an ASCII PLY file.
    Ignores normals/colors, keeps only the first 3 float columns.
    """
    print(f"[Load] {path}")

    points = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # Skip header
        while True:
            line = f.readline()
            if not line:
                # EOF before end_header → empty
                break
            if line.strip() == "end_header":
                break

        # Now read vertex lines
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = map(float, parts[:3])
            except ValueError:
                # Non-numeric line, skip
                continue
            points.append((x, y, z))

    if not points:
        print(f"[WARN] No vertices parsed from {path}")
        return np.empty((0, 3), dtype=np.float32)

    pts = np.asarray(points, dtype=np.float32)

    # Clean invalid values
    before = len(pts)
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    removed = before - len(pts)
    if removed > 0:
        print(f"[Clean] Removed {removed} invalid points (NaN/Inf) from {path}")

    print(f"[Load] Loaded {len(pts)} points from {path}")
    return pts


def random_downsample(points: np.ndarray, max_points: int, name: str) -> np.ndarray:
    """
    Randomly downsample point array to at most max_points.
    """
    n = len(points)
    if n == 0:
        return points
    if n <= max_points:
        print(f"[Downsample] {name}: {n} ≤ {max_points}, no downsampling")
        return points

    print(f"[Downsample] {name}: {n} → {max_points}")
    rng = np.random.default_rng(0)  # deterministic
    idx = rng.choice(n, size=max_points, replace=False)
    return points[idx]


def evaluate_point_cloud(
    pred_path: str,
    gt_path: str,
    max_points: int = 1_000_000,
) -> Dict[str, float]:
    """
    Evaluate predicted vs GT point cloud given their ASCII PLY paths.
    Returns a dict with Accuracy, Completeness, Overall (in same units as input).
    """

    # 1. Load point clouds
    pred = load_ascii_ply_xyz(pred_path)
    gt = load_ascii_ply_xyz(gt_path)

    print(f"Predicted points (raw): {len(pred)}")
    print(f"GT points (raw):        {len(gt)}")

    if len(pred) == 0 or len(gt) == 0:
        print("[ERROR] One of the point clouds is empty; returning NaNs")
        return {
            "Accuracy": float("nan"),
            "Completeness": float("nan"),
            "Overall": float("nan"),
        }

    # 2. Optional downsampling for speed
    if max_points and max_points > 0:
        pred = random_downsample(pred, max_points, "Pred")
        gt = random_downsample(gt, max_points, "GT")

    print(f"Predicted points (eval): {len(pred)}")
    print(f"GT points (eval):        {len(gt)}")

    # 3. Build KD-trees
    print("[Eval] Building KD-Trees...")
    pred_tree = cKDTree(pred)
    gt_tree = cKDTree(gt)

    # 4. Compute distances
    print("[Eval] Accuracy (pred → gt)...")
    acc_dists, _ = gt_tree.query(pred, k=1, workers=-1)  # distance from each pred to nearest gt

    print("[Eval] Completeness (gt → pred)...")
    comp_dists, _ = pred_tree.query(gt, k=1, workers=-1)  # distance from each gt to nearest pred

    # Percentile trimming: DTU official evaluation trims extreme outliers
    # Use 90th percentile threshold to discard grossly wrong matches
    acc_threshold = np.percentile(acc_dists, 90)
    comp_threshold = np.percentile(comp_dists, 90)
    acc_trimmed = acc_dists[acc_dists <= acc_threshold]
    comp_trimmed = comp_dists[comp_dists <= comp_threshold]

    accuracy = float(acc_trimmed.mean()) if len(acc_trimmed) > 0 else float(acc_dists.mean())
    completeness = float(comp_trimmed.mean()) if len(comp_trimmed) > 0 else float(comp_dists.mean())
    overall = float((accuracy + completeness) * 0.5)

    print("===== RESULTS =====")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Completeness: {completeness:.4f}")
    print(f"Overall:      {overall:.4f}")
    print("===================")

    return {
        "Accuracy": accuracy,
        "Completeness": completeness,
        "Overall": overall,
    }


def _write_metrics(output_path: str, metrics: Dict[str, float], num_pred: int, num_gt: int) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {metrics['Accuracy']:.6f}\n")
        f.write(f"Completeness: {metrics['Completeness']:.6f}\n")
        f.write(f"Overall: {metrics['Overall']:.6f}\n")
        f.write(f"NumPredEval: {num_pred}\n")
        f.write(f"NumGTEval: {num_gt}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DTU style point clouds (ASCII PLY)")
    parser.add_argument("--pred", required=True, help="Path to predicted ASCII PLY (XYZ)")
    parser.add_argument("--gt", required=True, help="Path to GT ASCII PLY (XYZ)")
    parser.add_argument("--output", help="Path to output metrics .txt")
    parser.add_argument(
        "--max_points",
        type=int,
        default=1000000,
        help="Max points per cloud for evaluation (after downsampling).",
    )
    args = parser.parse_args()

    metrics = evaluate_point_cloud(args.pred, args.gt, max_points=args.max_points)

    # If called from CLI, write metrics if requested
    if args.output:
        # we recompute sizes with same loader so we know eval counts
        pred_eval = random_downsample(load_ascii_ply_xyz(args.pred), args.max_points, "Pred")
        gt_eval = random_downsample(load_ascii_ply_xyz(args.gt), args.max_points, "GT")
        _write_metrics(args.output, metrics, len(pred_eval), len(gt_eval))
        print(f"[Save] Metrics saved → {args.output}")


if __name__ == "__main__":
    main()
