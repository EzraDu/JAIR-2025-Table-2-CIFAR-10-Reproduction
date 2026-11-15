"""Reduced-sample reproduction of paper Table 2 (CIFAR-10, CMI+IPR SVM).

Sample sizes requested for this run:
- 10240 inverted images per method (paper: 50000)
- single-method detectors would then train on 16384 = 80% of 10240+10240
  (paper: about 80000)

Table 2 mixes three sources, not one. This script keeps 10240 from each of
original CIFAR-10 train, CMI and IPR, then uses the same 80/20 split as
InverseDetect[combined].py. That trains on 24576 images. Pass --match-single
to draw a seeded 20480-image pool instead, so the training split is 16384.

The classifier is the GTSRB combined-detector procedure: concatenate the
512-d victim feature with the 512-d ImageNet ResNet18 feature and fit
SVC(kernel="rbf", class_weight="balanced"). CIFAR-10 does not refit PCA.
Victim features are the author's 8192-d relu map, transformed by the saved
[['cmi', 'ipr']]relu_512 PCA.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from joblib import dump, load
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

# The author PCA was pickled as daal4py.PCA without n_jobs. Its transform then
# crashes. Project with stock scikit-learn, which matches components_ exactly.
def _use_stock_pca_transform():
    import daal4py.sklearn.decomposition._pca as dpca

    def transform(self, X):
        return dpca.PCA_original.transform(self, X)

    dpca.PCA.transform = transform


_use_stock_pca_transform()
from function import extract_features_cifar10
from registry import get_dataset

PER_SOURCE = 10240
RELU_DIM = 512
RESNET_DIM = 512
# Author PCA was fit on WRN-40-2 relu(bn(block3)), flattened: 128 x 8 x 8.
PCA_IN_DIM = 8192
BATCH = 128
# Same tag InverseDetect[combined].py writes for combined_method=["cmi", "ipr"].
TRAIN_METHOD = ["cmi", "ipr"]
METHODS = ["noInv", "cmi", "dafl", "deepinv", "dfq", "ipr", "zskt"]
ROOT = Path(__file__).resolve().parent


def subset_first_n(dataset, n, seed):
    if len(dataset) < n:
        raise SystemExit(f"need {n} images, found {len(dataset)}")
    generator = torch.Generator().manual_seed(seed)
    chosen = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    return Subset(dataset, chosen)


def count_images(method):
    folder = ROOT / "Invdata" / "cifar10" / method
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-single", action="store_true",
                        help="use a 20480-image pool so the 80% training split is 16384")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os_cwd = Path.cwd()
    if os_cwd != ROOT:
        import os
        os.chdir(ROOT)

    missing = [m for m in ("cmi", "ipr") if count_images(m) < PER_SOURCE]
    if missing:
        detail = {m: count_images(m) for m in ("cmi", "ipr")}
        raise SystemExit(
            "Table 2 trains on original CIFAR-10 + CMI + IPR, "
            f"with {PER_SOURCE} images each. Missing or short: {detail}."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    ori = subset_first_n(get_dataset("cifar10", "noInv", train=True), PER_SOURCE, args.seed)
    cmi = subset_first_n(get_dataset("cifar10", "cmi", train=True), PER_SOURCE, args.seed + 1)
    ipr = subset_first_n(get_dataset("cifar10", "ipr", train=True), PER_SOURCE, args.seed + 2)
    pool = ConcatDataset([ori, cmi, ipr])
    if args.match_single:
        pool = subset_first_n(pool, PER_SOURCE * 2, args.seed + 3)
    train_size = int(len(pool) * 0.8)
    test_size = len(pool) - train_size
    train_set, holdout = random_split(
        pool, [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"pool={len(pool)} train={train_size} holdout={test_size}", flush=True)

    pca_path = ROOT / "checkpoints" / "cifar10" / "pca" / f"[{TRAIN_METHOD}]relu_{RELU_DIM}.joblib"
    pca = load(pca_path)
    if int(pca.n_features_in_) != PCA_IN_DIM or int(pca.n_components_) != RELU_DIM:
        raise SystemExit(
            f"{pca_path.name} expects {PCA_IN_DIM}->{RELU_DIM}, "
            f"got {pca.n_features_in_}->{pca.n_components_}"
        )
    print(
        f"pca={pca_path.name} in={pca.n_features_in_} out={pca.n_components_} "
        f"fit_samples={pca.n_samples_}",
        flush=True,
    )

    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=0)
    holdout_loader = DataLoader(holdout, batch_size=BATCH, shuffle=False, num_workers=0)
    # train=False loads the author PCA and only transforms. train=True would refit it.
    x_train, x_train_res, y_train = extract_features_cifar10(
        train_loader, TRAIN_METHOD, device, RELU_DIM, RESNET_DIM, train=False)
    x_hold, x_hold_res, y_hold = extract_features_cifar10(
        holdout_loader, TRAIN_METHOD, device, RELU_DIM, RESNET_DIM, train=False)
    if x_train.shape[1] != RELU_DIM or x_train_res.shape[1] != RESNET_DIM:
        raise SystemExit(
            f"expected PCA {RELU_DIM} and ResNet {RESNET_DIM}, "
            f"got {x_train.shape[1]} and {x_train_res.shape[1]}"
        )
    x_train = np.concatenate([x_train, x_train_res], axis=1)
    x_hold = np.concatenate([x_hold, x_hold_res], axis=1)

    svm = SVC(kernel="rbf", class_weight="balanced")
    print(f"fitting SVM on {len(x_train)} x {x_train.shape[1]}", flush=True)
    svm.fit(x_train, y_train.ravel())
    svm_path = ROOT / "checkpoints" / "cifar10" / "svm" / (
        f"[table2_10k]relu_{RELU_DIM}resnet_{RESNET_DIM}.joblib"
    )
    svm_path.parent.mkdir(parents=True, exist_ok=True)
    dump(svm, svm_path)
    hold_pred = svm.predict(x_hold)
    print(f"holdout accuracy={accuracy_score(y_hold, hold_pred):.4f}")
    print(classification_report(y_hold, hold_pred))

    print("method,n,inv_percent", flush=True)
    for method in METHODS:
        if method != "noInv" and count_images(method) == 0:
            print(f"{method},0,MISSING", flush=True)
            continue
        data = get_dataset("cifar10", method, train=False)
        if method != "noInv" and len(data) > PER_SOURCE:
            data = subset_first_n(data, PER_SOURCE, args.seed + 10)
        loader = DataLoader(data, batch_size=BATCH, shuffle=False, num_workers=0)
        x_wrn, x_res, _ = extract_features_cifar10(
            loader, TRAIN_METHOD, device, RELU_DIM, RESNET_DIM, train=False)
        x = np.concatenate([x_wrn, x_res], axis=1)
        pred = svm.predict(x)
        ratio = float(np.count_nonzero(pred == "Inv")) / len(pred)
        print(f"{method},{len(pred)},{ratio * 100:.2f}", flush=True)
    print(f"svm={svm_path}", flush=True)


if __name__ == "__main__":
    main()
