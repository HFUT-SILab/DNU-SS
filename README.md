# DNU-SS: Dual Non-Uniformity-Aware Selective Spiking for Open-Set Palmprint Verification

This repository provides the PyTorch implementation of DNU-SS, including the
complete model, seven component ablations, and training/evaluation on PolyU,
Tongji, HFUT, and the Blue, Red and Green multispectral evaluation sets.

## Method

DNU-SS combines local texture representation and selective spiking interaction:

- **DGT-Stem** uses dual-scale learnable Gabor filters and cross-scale response
  modulation to extract directional palmprint textures.
- **TEB** produces a fine-grained local texture descriptor.
- **SGSB** combines a palmprint saliency prior and multi-step spiking responses.
  Four **SPGTR** blocks use **STRI**, **SERT** and **TIMR-gated SERT** to update
  selected positions, while unselected positions retain their input features.
- TEB and SGSB descriptors are fused with weights 0.7 and 0.3. Training uses
  ArcFace supervision; verification uses cosine similarity between L2-normalized
  descriptors without the classification head.

The complete model uses four computational steps and update counts
225/191/157/123.

## Repository Structure

```text
DNU_SS/
  dnu_ss/
    models/
      dnu_ss.py          # Complete network and descriptor fusion
      dgt_stem.py        # Dual-scale Gabor texture stem
      teb.py             # Texture Enhancement Branch
      prior.py           # Palmprint saliency prior
      sgsb.py            # Saliency-Guided Spiking Branch
      spgtr.py           # Fixed-budget position selection and update
      scoring.py         # STRI, SERT, TIMR and TIMRGate
      spiking.py         # SpatialIndexedLIF, SpikingSSA and SpikingMLP
      layers.py          # Gabor, ECA and ArcFace layers
      initialization.py # Initialization and optimizer parameter order
    splits/              # Open/closed protocol index files; no images
    data.py              # ROI loading, augmentation and normalization
    train.py             # Training
    evaluate.py          # Feature extraction, metrics and ROC export
    run_ablations.py     # Shared A1-A7 training/evaluation pipeline
  scripts/
    train_test.sh
    run_ablations.sh
  requirements.txt
  pyproject.toml
```

## Requirements

Python 3.10 or later is required. Install a matching PyTorch/torchvision build
for your CUDA environment, then install the dependencies and package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The manuscript reports experiments using PyTorch 2.12.1 and CUDA 13.0 on
Ubuntu 22.04 with an NVIDIA GeForce RTX 4090. CuPy is optional; the commands
below select the Torch LIF backend explicitly.

## Data Preparation

Obtain the datasets under their original terms and arrange the ROI images as:

```text
DATA_ROOT/
  TJC/
  PolyUII_20/
  HFUT/
  PolyUM/Blue/
  PolyUM/Red/
  PolyUM/Green/
```

The 24 index files in `dnu_ss/splits/{open,closed}` use this format:

```text
relative/image/path label
```

Each palm is a separate identity. Open-set training/test identities are disjoint;
closed-set verification shares identities and separates acquisition sessions.
Keep the released labels and index order unchanged. All ROIs are loaded in
grayscale and resized to 128 x 128, with foreground normalization. Training
randomly applies one augmentation; testing has no random augmentation.

Validate your ROI export before training:

```bash
python -m dnu_ss.preflight --dataset Tongji --protocol open \
  --data-root /path/to/DATA_ROOT --output outputs/tongji_preflight.json
```

Dataset images and pretrained weights are not included.

## Training

Complete Tongji open-set model:

```bash
python -u -m dnu_ss.train \
  --dataset Tongji --protocol open --data-root /path/to/DATA_ROOT \
  --output outputs/tongji_full --seed 42 \
  --epochs 400 --batch-size 128 --num-workers 8 --save-interval 1000 \
  --device cuda:0 --lif-backend torch --amp
```

Replace `Tongji` with `PolyU`, `HFUT`, `Blue`, `Red` or `Green` for other datasets,
or use `--protocol closed` for closed-set verification. The identity count is
inferred from the training index. The seed above is an example, not a declared
seed for a particular manuscript result.

The model uses AdamW, a base learning rate of 1e-4, weight decay 0.05, ten-epoch
linear LR warmup and cosine annealing to 1e-6. Prior gain is introduced over
20 epochs. ArcFace uses scale 30 and margin 0.5, with dropout 0.5 before the
classifier. The checkpoint with highest training classification accuracy is
selected; later ties win. Test EER is not used for checkpoint selection.

`--save-interval 1000` avoids periodic checkpoints during 400 epochs while
retaining the best weight. Keep this interval fixed between comparisons because
it also controls the extra prior-visualization batch. Use tmux to protect a run
against SSH disconnection.

## Evaluation

```bash
python -u -m dnu_ss.evaluate \
  --checkpoint outputs/tongji_full/checkpoints/best.pt \
  --data-root /path/to/DATA_ROOT --output outputs/tongji_full/evaluation \
  --batch-size 256 --num-workers 0 --device cuda:0 --lif-backend torch
```

Evaluation strictly restores the checkpoint configuration and its update
budgets. All distinct unordered test-image pairs are evaluated, excluding
self-pairs and duplicate pairs. Same-palm pairs are genuine; different-palm
pairs are impostors. Scores use cosine similarity between normalized descriptors.

Outputs include `features.npy`, `labels.npy`, `metrics.json`, `roc.npz`,
`roc.png` and `roc.pdf`. Raw features and ROC arrays support later redrawing.
AUC/EER are percentages; d-prime uses separate population variances.

### Component Ablations

Run A1-A7 once each with an explicit example seed:

```bash
DATA_ROOT=/path/to/DATA_ROOT RESULT_ROOT=/path/to/results \
  bash scripts/run_ablations.sh Tongji 42
```

Select one configuration with `dnu_ss.train --ablation NAME`:

| ID | NAME | Configuration |
| --- | --- | --- |
| A1 | `teb_only` | TEB only |
| A2 | `dense_no_prior` | TEB + SGSB, no selection or prior |
| A3 | `prune_stri` | Fixed-budget STRI selection, no prior |
| A4 | `dense_prior` | Prior, no selection |
| A5 | `prune_stri_prior` | Fixed-budget STRI + prior |
| A6 | `prune_stri_prior_sert` | A5 + SERT |
| A7 | `full` | Complete model with TIMR-gated SERT |

Each group has its own checkpoint and evaluation outputs. `summary.csv` is
updated after each completed group. Testing restores its configuration rather
than applying a different ablation or pruning override.

## Experimental Results

The following tables report the **manuscript results**, not results generated
by the example commands above. For intra-dataset open-set comparison, the
manuscript reports the lowest EER over ten independent runs with different
random seeds, with d-prime taken from that same run.

### Closed-Set Verification (Table I)

Each cell is **EER (%) / d-prime**. `--` denotes an unreported result.

| Method | PolyU | Tongji | HFUT | Blue | Red | Green |
| --- | --- | --- | --- | --- | --- | --- |
| CompNet | 0.0055 / 10.3465 | 0.0006 / 12.3881 | -- | 0.0170 / 11.1666 | 0.0106 / 11.4253 | 0.0001 / 11.6262 |
| CO3Net | 0.0000 / 15.7157 | 0.0000 / 14.0948 | 0.0054 / 13.5854 | 0.0000 / 17.2842 | 0.0000 / 19.9190 | 0.0000 / 16.4601 |
| CCNet | 0.0000 / 17.4620 | 0.0000 / 14.5821 | 0.0000 / 14.4584 | 0.0000 / 17.2932 | 0.0000 / 18.4603 | 0.0000 / 16.4791 |
| SF2Net | 0.0000 / 15.7231 | 0.0006 / 15.1278 | 0.0017 / 14.6093 | 0.0000 / 15.8126 | 0.0000 / 17.2619 | 0.0000 / 15.3094 |
| DNU-SS | 0.0000 / 18.8096 | 0.0000 / 16.7859 | 0.0001 / 15.9379 | 0.0000 / 18.2800 | 0.0000 / 18.2986 | 0.0000 / 17.7276 |

DNU-SS matches the best reported EER on PolyU, Tongji, Blue, Red and Green.
Its d-prime is highest on five sets; on Red, CO3Net and CCNet have higher d-prime.

### Intra-Dataset Open-Set Verification (Table II)

Each cell is **EER (%) / d-prime**.

| Method | PolyU | Tongji | HFUT | Blue | Red | Green |
| --- | --- | --- | --- | --- | --- | --- |
| CompNet | 0.4314 / 5.3427 | 0.3404 / 5.2632 | 0.5882 / 5.6048 | 0.1775 / 6.2807 | 0.0818 / 7.0004 | 0.2271 / 5.9850 |
| CO3Net | 0.5437 / 4.8510 | 0.3491 / 4.8751 | 1.1079 / 4.6307 | 0.2496 / 6.0052 | 0.0970 / 6.6987 | 0.3512 / 5.7183 |
| CCNet | 0.2739 / 5.4274 | 0.2386 / 5.0572 | 0.5325 / 5.3196 | 0.0970 / 6.5993 | 0.0242 / 7.3052 | 0.1212 / 6.2572 |
| SF2Net | 0.3308 / 5.8661 | 0.3617 / 5.5468 | 0.5952 / 5.7889 | 0.2413 / 6.9922 | 0.0438 / 7.8835 | 0.1695 / 6.8246 |
| DNU-SS | 0.1474 / 6.1805 | 0.1271 / 5.8036 | 0.3650 / 5.9655 | 0.1091 / 7.3667 | 0.0186 / 7.9328 | 0.0922 / 7.0572 |

DNU-SS has the lowest EER on five of six sets, with relative reductions of
23.1%-46.7% over CCNet. On Blue it ranks second behind CCNet. DNU-SS has the
highest reported d-prime on all six sets.

### Component Ablations on Tongji (Table III)

| Configuration | A1 | A2 | A3 | A4 | A5 | A6 | A7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EER (%) | 0.1474 | 0.1316 | 0.1298 | 0.1295 | 0.1309 | 0.1281 | 0.1271 |

SGSB complements TEB, reducing EER from 0.1474% to 0.1316%. STRI selection
and the saliency prior separately improve A2. Their combination in A5 remains
below A2 but is slightly above A3 and A4. Adding SERT and TIMR gating reduces
EER to 0.1281% and 0.1271%, respectively, under the same update budget.

### Early- Versus Late-Response Priority (Table IV)

| Priority | AUC (%) | EER (%) | d-prime |
| --- | --- | --- | --- |
| Early response (adopted) | 99.9990 | 0.1271 | 5.8036 |
| Late response | 99.9989 | 0.1316 | 5.8031 |

The manuscript comparison supports early-response priority in the combined
SERT-TIMR pathway. The released configurations use the adopted early-response
definition; a late-response experimental option is not exposed.

### Computational Efficiency on Tongji (Table V)

| Method | EER (%) | Parameters (M) | Network inference time (s) | Throughput (samples/s) | Peak allocated memory (MB) |
| --- | --- | --- | --- | --- | --- |
| CompNet | 0.3404 | 3.27 | 0.075472 | 79499.70 | 128.73 |
| CO3Net | 0.3491 | 20.57 | 1.104900 | 5430.36 | 821.95 |
| CCNet | 0.2386 | 20.57 | 0.786072 | 7632.89 | 1341.01 |
| SF2Net | 0.3617 | 13.08 | 0.651118 | 9214.92 | 969.97 |
| DNU-SS | 0.1271 | 18.14 | 2.081880 | 2882.01 | 1311.83 |

The manuscript uses 6000 test images, an RTX 4090, FP32 evaluation and batch
256. Preprocessed inputs are on the GPU before timing. After three warm-up
batches, five measurements are made with CUDA synchronization; median time
is reported. Timing includes feature extraction and final L2 normalization,
excluding loading, preprocessing, transfer and pair scoring. Peak memory is
the absolute peak allocated after resetting statistics following warm-up,
reported in decimal MB (1 MB = 1,000,000 bytes). Baseline parameter counts
follow X-Palm, as stated in the manuscript.

### Dense Reference Versus Complete Model (Table VI)

| Metric | Dense (A2) | Complete (A7) | Change |
| --- | --- | --- | --- |
| EER (%) | 0.1316 | 0.1271 | -0.0045 percentage points |
| Network inference time (s) | 2.338957 | 2.081880 | -10.99% |
| Throughput (samples/s) | 2565.25 | 2882.01 | +12.35% |
| Peak allocated memory (MB) | 1311.37 | 1311.83 | +0.46 MB |
| Token budget (%) | 100.00 | 77.33 | -22.67% |

The complete model reduces the summed update count from 900 to 696 and
improves verification error and extraction time relative to the dense
reference under the manuscript's measurement settings.
