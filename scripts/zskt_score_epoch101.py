"""Score 10240 images from CMI ZSKT epoch 101 (200-epoch schedule) with the existing Table 2 SVM.

Uses steps 40..79 of that epoch (40 batches x 256). Does not refit PCA or the SVM.
Does not read Invdata/cifar10/zskt.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from joblib import load
from PIL import Image
from torch.utils.data import DataLoader, Dataset

INVDET = Path(os.environ.get("INVDET_ROOT", "invdet"))
CALLS = Path(os.environ.get("CMI_ROOT", "CMI")) / "run" / "zskt_stages" / "epoch101_calls"
PNG_ROOT = INVDET / "Invdata" / "cifar10" / "zskt_e101_t200" / "0"
SVM_PATH = INVDET / "checkpoints" / "cifar10" / "svm" / "[table2_10k]relu_512resnet_512.joblib"
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
FIRST_STEP = 40
LAST_STEP = 79

os.chdir(INVDET)
sys.path.insert(0, str(INVDET))
import reproduce_table2  # noqa: E402  stock PCA transform, then function import
from function import extract_features_cifar10  # noqa: E402


class Uint8Images(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index].astype(np.float32) / 255.0)
        image = (image - MEAN) / STD
        return image, 0


def load_selected():
    parts = []
    for step in range(FIRST_STEP, LAST_STEP + 1):
        path = CALLS / f"step_{step:02d}.npy"
        arr = np.load(path)
        if arr.dtype != np.uint8 or arr.shape != (256, 3, 32, 32):
            raise SystemExit(f"bad batch {path} shape={arr.shape} dtype={arr.dtype}")
        parts.append(arr)
    images = np.concatenate(parts, axis=0)
    if len(images) != 10240:
        raise SystemExit(f"expected 10240 images, got {len(images)}")
    return images


def write_pngs(images):
    if PNG_ROOT.exists():
        for old in PNG_ROOT.glob("*.png"):
            old.unlink()
    PNG_ROOT.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        Image.fromarray(image.transpose(1, 2, 0)).save(PNG_ROOT / f"{index:05d}.png")
    print(f"WROTE_PNG n={len(images)} dir={PNG_ROOT}", flush=True)


def main():
    images = load_selected()
    print(
        f"SCORE_START n={len(images)} epoch=101 steps={FIRST_STEP}-{LAST_STEP}",
        flush=True,
    )
    write_pngs(images)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(Uint8Images(images), batch_size=128, shuffle=False, num_workers=0)
    x_wrn, x_res, _ = extract_features_cifar10(
        loader, ["cmi", "ipr"], device, 512, 512, train=False)
    hybrid = np.concatenate([x_wrn, x_res], axis=1)
    svm = load(SVM_PATH)
    print(f"svm_classes={list(svm.classes_)} svm={SVM_PATH.name}", flush=True)
    pred = svm.predict(hybrid)
    scores = svm.decision_function(hybrid)
    pred_inv = float(np.mean(pred == "Inv"))
    print(
        f"ZSKT_EVAL n={len(pred)} pred_inv={pred_inv:.4f} inv_percent={pred_inv * 100:.2f} "
        f"decision_mean={float(scores.mean()):.4f} "
        f"p10={float(np.percentile(scores, 10)):.4f} "
        f"p50={float(np.percentile(scores, 50)):.4f} "
        f"p90={float(np.percentile(scores, 90)):.4f} "
        f"frac_neg={float(np.mean(scores < 0)):.4f}",
        flush=True,
    )
    print("SCORE_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        os._exit(code)
    os._exit(0)
