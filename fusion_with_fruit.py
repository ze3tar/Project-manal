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
    depth_scale: float = 1000.0,
    target_size: Tuple[int, int] = (640, 512),
):
    """Back-project fruit pixels into world coordinates using meter-based depth."""
    depth_map = depth_map.astype(np.float32) * depth_scale
    fruit_mask = fruit_mask.astype(np.float32)
    h, w = depth_map.shape

    if fruit_mask.shape != depth_map.shape:
        fruit_mask = cv2.resize(fruit_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    valid = (depth_map > 0) & (fruit_mask > mask_threshold)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,))

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

    labels = np.ones(points.shape[0], dtype=np.int32)
    return points, colors, labels


def save_fruit_ply(path: str, points: np.ndarray, colors: np.ndarray, labels: np.ndarray):
    if points.shape[0] == 0:
        raise ValueError("No points to save")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    colors_uint8 = np.zeros((points.shape[0], 3), dtype=np.uint8)
    colors_uint8[:] = np.array([255, 0, 0], dtype=np.uint8)
    labels = labels.astype(np.int32)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int fruit\n")
        f.write("end_header\n")
        for point, color, label in zip(points, colors_uint8, labels):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {int(label)}\n"
            )


def save_fruit_csv(path: str, points: np.ndarray, colors: np.ndarray, labels: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    colors_uint8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    data = np.concatenate(
        [points, colors_uint8.astype(np.float32), labels[:, None].astype(np.float32)], axis=1
    )
    header = "x,y,z,r,g,b,fruit"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def fuse_fruit_views(points_list, colors_list, labels_list, voxel_size=0.01):
    if len(points_list) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,))

    points, colors = fuse_point_clouds(points_list, colors_list, voxel_size=voxel_size)
    labels = np.ones(points.shape[0], dtype=np.int32)
    return points, colors, labels
