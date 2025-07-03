"""Simple dataset wrapper for solar panel images organised as:
<root>/<class_name>/<image>.jpg"""

from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

class SolarPanelDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = list(self._gather())
        self.classes = sorted({label for _, label in self.samples})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def _gather(self):
        for label_dir in self.root.iterdir():
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for img_path in label_dir.glob("*.jpg"):
                yield img_path, label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label_str = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[label_str]
        return image, label
