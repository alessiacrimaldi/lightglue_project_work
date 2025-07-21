import urllib
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from lightglue.utils import ImagePreprocessor, load_image


class HPatchesDataset(torch.utils.data.Dataset):
    """
    adapted from https://github.com/cvg/glue-factory/blob/main/gluefactory/datasets/hpatches.py
    """

    url = "https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip"

    def __init__(
        self, root: str = "./data/hpatches-sequences-release", resize: int = 480
    ):
        self.root = Path(root)
        self.preprocessor = ImagePreprocessor(resize=resize)

        if not self.root.exists():
            self._download()

        self.sequences = sorted([x.name for x in self.root.iterdir()])
        self.items = [(seq, i) for seq in self.sequences for i in range(2, 7)]

    def _download(self):
        data_dir = self.root.parent
        data_dir.mkdir(parents=True, exist_ok=True)

        zip_name = self.url.split("/")[-1]
        zip_path = data_dir / zip_name

        urllib.request.urlretrieve(self.url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        zip_path.unlink()

    def _read_image(self, seq: str, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        img = load_image(self.root / seq / f"{idx}.ppm")
        img_resized, scale = self.preprocessor(img)
        T = np.diag([scale[0].item(), scale[1].item(), 1])
        return img_resized, T

    @staticmethod
    def read_homography(path):
        with open(path, "r") as f:
            lines = f.readlines()

        matrix = []
        for line in lines:
            elements = line.strip().split()
            if elements:
                matrix.append([float(e) for e in elements])
        return np.array(matrix, dtype=np.float64)

    def __getitem__(self, idx: int):
        seq, q_idx = self.items[idx]

        data0, T0 = self._read_image(seq, 1)
        data1, T1 = self._read_image(seq, q_idx)

        H = self.read_homography(self.root / seq / f"H_1_{q_idx}")
        H = T0 @ H @ np.linalg.inv(T1)

        return {
            "H_0to1": H.astype(np.float32),
            "scene": seq,
            "image0": data0,
            "image1": data1,
        }

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    ds = HPatchesDataset()
    ds[0]
