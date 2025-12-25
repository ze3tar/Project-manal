import os
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MinneAppleDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: Tuple[int, int] = (640, 512),
        augment: bool = False,
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_width, self.img_height = img_size
        self.augment = augment

        split_dir = os.path.join(root_dir, split)
        if os.path.isdir(split_dir):
            image_dir = os.path.join(split_dir, "images")
            mask_dir = os.path.join(split_dir, "masks")
        else:
            image_dir = os.path.join(root_dir, "images")
            mask_dir = os.path.join(root_dir, "masks")

        self.image_paths, self.mask_paths = self._collect_pairs(image_dir, mask_dir)
        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No image/mask pairs found under {image_dir} and {mask_dir}."
            )

    def _collect_pairs(self, image_dir: str, mask_dir: str) -> Tuple[List[str], List[str]]:
        image_paths = []
        mask_paths = []
        for filename in sorted(os.listdir(image_dir)):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            image_path = os.path.join(image_dir, filename)
            stem, _ = os.path.splitext(filename)
            mask_path = os.path.join(mask_dir, f"{stem}.png")
            if not os.path.exists(mask_path):
                continue
            image_paths.append(image_path)
            mask_paths.append(mask_path)
        return image_paths, mask_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < 0.2:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        return image, mask

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {mask_path}")

        image = cv2.resize(image, (self.img_width, self.img_height))
        mask = cv2.resize(mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = (mask > 0).astype(np.int64)
        mask = torch.from_numpy(mask)

        return {
            "image": image,
            "mask": mask,
            "path": image_path,
        }
