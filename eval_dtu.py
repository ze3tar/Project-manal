"""DTU point cloud evaluation utilities."""

import argparse
import os
from typing import Dict

import numpy as np
from scipy.spatial import cKDTree


def _load_ply_points(path: str) -> np.ndarray:
    """Load XYZ points from an ASCII PLY file."""

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        if header != "ply":
            raise ValueError(f"{path} is not a valid ASCII PLY file")

        vertex_count = 0
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of PLY header")
            line = line.strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
            if line == "end_header":
                break

        points = []
        for _ in range(vertex_count):
            values = f.readline()
            if not values:
                break
            parts = values.strip().split()
            if len(parts) < 3:
                continue
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def evaluate_point_cloud(pred_path: str, gt_path: str) -> Dict[str, float]:
    """Compute DTU accuracy/completeness metrics using KD-Trees."""

    pred_points = _load_ply_points(pred_path)
    gt_points = _load_ply_points(gt_path)

    metrics = {"Accuracy": float("nan"), "Completeness": float("nan"), "Overall": float("nan")}

    if len(pred_points) == 0 or len(gt_points) == 0:
        return metrics

    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)

    accuracy_dists, _ = gt_tree.query(pred_points, k=1)
    completeness_dists, _ = pred_tree.query(gt_points, k=1)

    accuracy = float(accuracy_dists.mean())
    completeness = float(completeness_dists.mean())
    overall = float((accuracy + completeness) / 2.0)

    metrics.update({"Accuracy": accuracy, "Completeness": completeness, "Overall": overall})
    return metrics


def _write_metrics(metrics: Dict[str, float], output_path: str, num_points: int) -> None:
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {metrics['Accuracy']:.6f}\n")
        f.write(f"Completeness: {metrics['Completeness']:.6f}\n")
        f.write(f"Overall: {metrics['Overall']:.6f}\n")
        f.write(f"Number of points: {num_points}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DTU point clouds")
    parser.add_argument("--pred", required=True, help="Path to predicted PLY")
    parser.add_argument("--gt", required=True, help="Path to ground-truth PLY")
    parser.add_argument("--output", help="Optional metrics text file")
    args = parser.parse_args()

    metrics = evaluate_point_cloud(args.pred, args.gt)
    print(
        f"Accuracy: {metrics['Accuracy']:.6f}, Completeness: {metrics['Completeness']:.6f}, Overall: {metrics['Overall']:.6f}"
    )

    if args.output:
        _write_metrics(metrics, args.output, num_points=len(_load_ply_points(args.pred)))


if __name__ == "__main__":
    main()

