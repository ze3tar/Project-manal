import os
from typing import Optional, Tuple

import numpy as np
import cv2

from fusion import _maybe_scale_intrinsic, fuse_point_clouds


def backproject_fruit_points(
    depth_map: np.ndarray,
    fruit_mask: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_w2c: np.ndarray,
    image: Optional[np.ndarray] = None,
    mask_threshold: float = 0.5,
    depth_scale: float = 1.0,
    target_size: Tuple[int, int] = (640, 512),
):
    """Back-project ALL valid-depth pixels into world coordinates.

    Returns per-point labels: ``1`` for fruit (above *mask_threshold*),
    ``0`` for background.  Also returns per-point confidence scores
    (raw softmax probability).
    """
    depth_map = depth_map.astype(np.float32) * depth_scale
    fruit_mask = fruit_mask.astype(np.float32)
    h, w = depth_map.shape

    if fruit_mask.shape != depth_map.shape:
        fruit_mask = cv2.resize(fruit_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Back-project ALL pixels with valid depth (not just fruit)
    valid = depth_map > 0
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return (np.zeros((0, 3)), np.zeros((0, 3)),
                np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32))

    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float32)
    depths = depth_map[valid]

    intrinsic = _maybe_scale_intrinsic(intrinsic, target_size)
    inv_k = np.linalg.inv(intrinsic)
    cam_coords = inv_k @ pixels * depths
    cam_coords_homo = np.vstack([cam_coords, np.ones((1, cam_coords.shape[1]))])

    world_coords = np.linalg.inv(extrinsic_w2c) @ cam_coords_homo
    points = world_coords[:3].T

    if image is None:
        colors = np.zeros((points.shape[0], 3), dtype=np.float32)
    else:
        if image.shape[0] != h or image.shape[1] != w:
            image = cv2.resize(image, (w, h))
        colors = image[ys, xs].astype(np.float32)
        if colors.max() > 1.0:
            colors /= 255.0

    # Per-point fruit labels and raw confidence scores
    confidences = fruit_mask[valid].astype(np.float32)
    labels = (confidences > mask_threshold).astype(np.int32)
    return points, colors, labels, confidences


def save_fruit_ply(path: str, points: np.ndarray, colors: np.ndarray,
                   labels: np.ndarray, confidences: Optional[np.ndarray] = None):
    """Save point cloud as PLY with selective coloring and optional confidence.

    Points with ``labels == 1`` (fruit) are colored RED (255, 0, 0).
    All other points keep their original RGB from ``colors``.
    """
    if points.shape[0] == 0:
        raise ValueError("No points to save")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    labels = labels.astype(np.int32)

    # Build per-point colors: fruit → red, non-fruit → original RGB
    if colors.max() <= 1.0 and colors.dtype in (np.float32, np.float64):
        colors_uint8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    else:
        colors_uint8 = np.clip(colors, 0, 255).astype(np.uint8)

    fruit_mask = labels == 1
    colors_uint8[fruit_mask] = np.array([255, 0, 0], dtype=np.uint8)

    has_confidence = confidences is not None and len(confidences) == len(points)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int fruit\n")
        if has_confidence:
            f.write("property float confidence\n")
        f.write("end_header\n")
        for i in range(points.shape[0]):
            line = (f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                    f"{int(colors_uint8[i, 0])} {int(colors_uint8[i, 1])} {int(colors_uint8[i, 2])} "
                    f"{int(labels[i])}")
            if has_confidence:
                line += f" {confidences[i]:.6f}"
            f.write(line + "\n")


def save_fruit_csv(path: str, points: np.ndarray, colors: np.ndarray,
                   labels: np.ndarray, confidences: Optional[np.ndarray] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    colors_uint8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    cols = [points, colors_uint8.astype(np.float32), labels[:, None].astype(np.float32)]
    header = "x,y,z,r,g,b,fruit"
    if confidences is not None and len(confidences) == len(points):
        cols.append(confidences[:, None].astype(np.float32))
        header += ",confidence"
    data = np.concatenate(cols, axis=1)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def fuse_fruit_views(points_list, colors_list, labels_list,
                     confidences_list=None, voxel_size=0.01):
    """Fuse multiple views with confidence-weighted averaging.

    When ``confidences_list`` is provided, voxel labels are determined by
    averaging the raw confidence scores instead of majority-voting on binary
    labels.
    """
    if len(points_list) == 0:
        return (np.zeros((0, 3)), np.zeros((0, 3)),
                np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32))

    all_points = np.vstack(points_list)
    all_colors = np.vstack(colors_list)
    all_labels = np.concatenate(labels_list)

    use_confidence = (confidences_list is not None and len(confidences_list) == len(points_list))
    if use_confidence:
        all_confidences = np.concatenate(confidences_list).astype(np.float64)
    else:
        all_confidences = all_labels.astype(np.float64)

    if len(all_points) == 0:
        return (all_points, all_colors, all_labels,
                np.zeros((0,), dtype=np.float32))

    # Voxel downsampling that also aggregates labels/confidences
    voxel_coords = np.floor(all_points / voxel_size).astype(np.int64)
    voxel_ids = (
        voxel_coords[:, 0] * 73856093
        + voxel_coords[:, 1] * 19349663
        + voxel_coords[:, 2] * 83492791
    )
    unique_ids, inverse = np.unique(voxel_ids, return_inverse=True)
    n_voxels = len(unique_ids)

    sums_points = np.zeros((n_voxels, 3), dtype=np.float64)
    sums_colors = np.zeros((n_voxels, 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=n_voxels).astype(np.float64)
    conf_sums = np.bincount(inverse, weights=all_confidences, minlength=n_voxels)

    for dim in range(3):
        sums_points[:, dim] = np.bincount(inverse, weights=all_points[:, dim], minlength=n_voxels)
        sums_colors[:, dim] = np.bincount(inverse, weights=all_colors[:, dim], minlength=n_voxels)

    fused_points = (sums_points.T / counts).T.astype(np.float32)
    fused_colors = (sums_colors.T / counts).T.astype(np.float32)

    # Confidence-weighted: average confidence, threshold at 0.5
    avg_confidence = (conf_sums / counts).astype(np.float32)
    fused_labels = (avg_confidence > 0.5).astype(np.int32)

    return fused_points, fused_colors, fused_labels, avg_confidence
