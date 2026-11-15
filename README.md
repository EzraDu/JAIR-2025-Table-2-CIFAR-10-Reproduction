# JAIR 2025 Table 2 (CIFAR-10) Reproduction

Reproduction of Table 2 from Yu et al., *Detecting Generative Model Inversion Attacks* (JAIR 2025). The victim model is a WRN-40-2 trained on CIFAR-10. The numbers in the table are detection rates: the proportion of samples classified as inverted images.

All inverted images were generated using the source code from the [CMI repository](https://github.com/zju-vipa/CMI) without any modifications to that repository.

## Results

Raw numbers are in [results/table2.csv](results/table2.csv).

| Test Set | Paper (%) | This Run (%) | # Images |
| --- | ---: | ---: | ---: |
| CIFAR-10 Test Set (false positive) | 4.74 | 8.60 | 10000 |
| CMI | 94.48 | 90.97 | 10240 |
| DAFL | 95.95 | 99.55 | 10240 |
| DeepInv | 79.98 | 73.37 | 10240 |
| DFQ | 97.25 | 80.48 | 10240 |
| IPR | 98.26 | 95.67 | 10240 |
| ZSKT | 88.77 | 91.05 | 10240 |
| Average of six inversion methods | 92.45±6.35 | 88.52±8.94 | — |

False positives were evaluated on the full CIFAR-10 test set. Each of the six inversion methods used 10240 images. The mean and standard deviation are computed only over these six methods; the standard deviation is the population standard deviation (square root of sum of squared deviations divided by 6). Hold-out accuracy was 0.9181 (trained on 24576 images, held out 6144 images).

## Inversion

The entry point is CMI’s `datafree_kd.py`. Coefficients match those in the repository’s `scripts/<method>/*_wrn402_wrn161.sh`. The student network is WRN-16-1 and the teacher is WRN-40-2.

| Method | Procedure | Best Student Accuracy |
| --- | --- | ---: |
| CMI | `CMISynthesizer` | — |
| DAFL | Official dafl script, 200 epochs, kept the last 10240 images | 66.98% |
| DFQ | Official dfq script, 200 epochs, kept the last 10240 images | 85.42% |
| DeepInv | Official deepinv script, 40 epochs; 1000 generator update steps per image | 61.36% |
| IPR | Images synthesized with `CMISynthesizer` (scratch coefficients, 40 epochs), then distilled to WRN-16-1 with the same repository’s `vanilla_kd.py` (`T=1`) | Synthesis 77.94%; after distillation 71.52% |
| ZSKT | Official zskt script with `--epochs 200`. Used synthesis calls from epoch 101, steps 40–79 (10240 images total), scored with the already-trained Table 2 SVM | 65.48% at that epoch (learning rate 0.0492) |

On Windows, CMI’s internal DataLoader with `num_workers=4` hangs after the first batch. In the collection process it was set to 0; the CMI source code itself was left unchanged.

## Detector

PCA was not re-fitted.

- Victim features: `ReLU(BN(block3))` of WRN-40-2, 128×8×8, flattened to 8192 dimensions, then reduced to 512 dimensions using the existing `[['cmi', 'ipr']]relu_512` PCA (fitted on 126400 samples).
- Image features: 512-dimensional features from ImageNet ResNet-18 with the fully-connected layer removed. Inputs were normalized with CIFAR-10 mean and variance (not ImageNet normalization).
- The two feature sets were concatenated into a 1024-dimensional vector and fed to `SVC(kernel="rbf", class_weight="balanced")`.
- Training pool: 10240 images each from the CIFAR-10 training set, CMI, and IPR; 80/20 split with seed 0. `--match-single` was not used.

After loading the PCA via sklearnex, `transform` is missing `n_jobs`. `reproduce_table2.py` switches to the original scikit-learn `transform` before importing `function`.

`reproduce_table2.py` must be run with `function.py` and `registry.py` in the same directory, and it reads `Invdata` and `checkpoints` from that directory. When evaluating ZSKT, do not re-run the script (it would re-fit the SVM). The 10240 ZSKT images are scored by `zskt_score_epoch101.py` using the already-saved SVM.

## Scripts

Set `CMI_ROOT` (root of the CMI repository) and `INVDET_ROOT` (root of the detection code) before running. If unset, the scripts look for relative paths `CMI` and `invdet`.

| File | Purpose |
| --- | --- |
| [scripts/reproduce_table2.py](scripts/reproduce_table2.py) | Trains the Table 2 SVM and evaluates noInv, CMI, DAFL, DeepInv, DFQ, IPR, and ZSKT. |
| [scripts/collect_cmi_methods.py](scripts/collect_cmi_methods.py) | Collects DAFL, DFQ, DeepInv, and IPR images using CMI source code. |
| [scripts/zskt_collect_epoch101.py](scripts/zskt_collect_epoch101.py) | Runs CMI’s zskt script up to epoch 101, saves the synthesized images from that epoch, stops, and restores student weights. |
| [scripts/zskt_score_epoch101.py](scripts/zskt_score_epoch101.py) | Takes steps 40–79 of epoch 101 (10240 images) and scores them with the existing SVM. Does not re-fit PCA or the SVM. |
| [scripts/zskt_epoch101_run.py](scripts/zskt_epoch101_run.py) | First collects epoch 101, then scores. |

Environment: Python 3.11, torch 2.11+cu128, torchvision 0.26.
