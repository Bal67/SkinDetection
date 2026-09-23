# SkinDetection: skin-condition classification measured across skin tones

> **Research prototype. Not a medical device, not a diagnosis, not clinically validated, and not a
> replacement for a dermatologist.** Model outputs are visual-similarity scores among a fixed set of
> 26 conditions and are frequently wrong.

## Purpose

The research question here is not only "how accurate is the model?" but:

> **How accurate is the model, and does its performance materially change across skin tones?**

Dermatology image datasets and models have historically under-represented darker skin. This POC
trains a small image classifier and **reports its performance separately for every Fitzpatrick
skin type (I–VI) and for lighter (I–III) vs. darker (IV–VI) skin**, with sample sizes and
confidence intervals. It does not claim to have solved bias. It measures it.

## Dataset

- **Fitzpatrick17k** ([Groh et al., 2021](https://github.com/mattgroh/fitzpatrick17k)); metadata is in
  `data/fitzpatrick17k.csv`. The current model is trained on Fitzpatrick17k only. It has **not**
  been trained on Google's SCIN dataset (see `docs/SCIN_INTEGRATION_PLAN.md` for future plans).
- The subset used: the 26 conditions of the original project, images from `atlasdermatologico.com.br`
  (the `dermaamin.com` images are excluded, as in the original project), minus rows the dataset's
  QC flagged "wrongly labelled". **2,346 images** were downloadable (3 of 2,349 failed).
- Skin tone is the dataset's `fitzpatrick_scale` annotation: an estimate made from the image by
  annotators, not self-reported. The dataset's second annotation agrees with it only about 46% of
  the time, so skin-tone labels are themselves noisy. 97 images have unknown skin type and are
  reported as a separate "unknown" group.
- There is no patient or lesion identifier, so different photos of the same patient may fall into
  different splits. Exact duplicates and perceptual near-duplicates are grouped before splitting.

## Limitations

- Research prototype; **no clinical validation; not a medical device; not for diagnosis or triage.**
- Only 26 conditions. Every image, including healthy skin or a non-skin photo, gets assigned to one
  of them.
- Small dataset (1,642 training images; some classes have fewer than 20 training images) from a
  single atlas source, with textbook-style clinical photos that differ from phone photos taken by patients.
- Skin-type annotations are noisy (see above), and subgroup test sets are small (e.g. Fitzpatrick I
  and VI each have fewer than 30 test images), so per-type numbers have wide confidence intervals.
- Possible patient-level leakage (no patient IDs).
- Softmax scores are not calibrated probabilities.

## Architecture

```
data/fitzpatrick17k.csv
  └─ scripts/prepare_dataset.py    load → filter labels → drop duplicates → fetch images
                                    → group near-duplicates → GROUP-LEVEL stratified split
                                    → data/splits.csv (one row per original image: train/val/test)
  └─ scripts/train.py              train+val only. On-the-fly augmentation (train only)
                                    → MobileNetV2 two-stage fine-tuning → best-val checkpoint
                                    → models/skin_mobilenetv2.keras + models/class_names.json
  └─ scripts/evaluate.py           test split only → reports/ (overall, per class, per Fitzpatrick type)
  └─ app.py                        Streamlit demo: same preprocessing, same class mapping
```

Package `skin_detection/`:

| module | responsibility |
|---|---|
| `config.py` | paths (all overridable via `SKIN_*` env vars), seed, default labels |
| `data.py` | dataset adapter to a common schema, filtering, duplicate detection, group split, leakage checks, sample weights |
| `preprocessing.py` | **the only** image preprocessing: EXIF orientation, RGB, resize, MobileNetV2 normalization |
| `model.py` | model construction, stage-B unfreezing, validated save/load (model ↔ class names) |
| `training.py` | image loading, augmentation, tf.data pipeline, two-stage training |
| `evaluation.py` | metrics incl. Fitzpatrick-stratified metrics, gaps, CIs, calibration |
| `inference.py` | top-k, uncertainty display rule |

**Model:** ImageNet MobileNetV2 → GlobalAveragePooling → Dense(256, ReLU, L2 1e-4) → Dropout(0.5)
→ softmax(26). Input 224×224.

**Preprocessing** (identical everywhere): EXIF transpose → RGB → scale the longer side to 224
(bilinear), then pad the shorter side symmetrically with black. There is no stretching and no
cropping, so a lesion near the edge is not cut off. Then `x / 127.5 − 1` (MobileNetV2 normalization).

**Training** (`scripts/train.py`):
- *Stage A*: backbone frozen, head trained (Adam 1e-3, up to 15 epochs).
- *Stage B (actual fine-tuning)*: layers from `block_13_expand` onward are unfrozen (last 4
  inverted-residual blocks + final conv). **All BatchNormalization layers stay frozen**, and the
  backbone always runs in inference mode. Recompiled with Adam 1e-5, up to 20 epochs.
- Early stopping and checkpointing monitor **validation balanced
  accuracy** (LR reduction monitors validation loss); the saved model is the best validation checkpoint across both stages.
- Seeds are fixed (`--seed`, default 42); `--deterministic` enables TF op determinism.

**Augmentation** (training images only, on the fly, never saved): horizontal flip, rotation ±14°,
translation ±5%, zoom ±10%, brightness ±10%, contrast ×0.9–1.1. **No color inversion, no hue or
saturation shifts, no vertical flips.** Skin and lesion color carry diagnostic information, and
recoloring light-skin images does not produce realistic darker-skin examples.

**Class imbalance and skin-tone representation**: the original scripts undersampled every class to
the rarest class, which discarded about 75% of images. Nothing is discarded now. Each training image
gets a loss weight: tone groups (I–III / IV–VI / unknown) are weighted to equal total weight, then
rescaled so every class carries equal total weight, clipped at 10× the mean.

**Class mapping:** training writes `models/class_names.json` (class order + image size + resize mode).
Loading fails loudly if the model's output size or input size disagrees with it.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10-3.12
pip install -r requirements-train.txt               # app only: pip install -r requirements.txt
```

## Reproduce

```bash
# 1. Fetch images (~2.3k files, ~150 MB) into data/images/ and build data/splits.csv.
#    data/splits.csv is committed. Re-running with the same seed and the same available images
#    reproduces it.
python scripts/prepare_dataset.py --download url --near-duplicates
#    Alternative source: images in S3 as <prefix><md5hash>.jpg, credentials via the normal AWS chain
#    SKIN_S3_BUCKET=my-bucket python scripts/prepare_dataset.py --download s3 --near-duplicates

# 2. Train (CPU is fine: roughly 1-2 h on a laptop; faster with a GPU)
python scripts/train.py

# 3. Final evaluation on the held-out test split
python scripts/evaluate.py
python scripts/evaluate.py --legacy      # the original 2024 model, for comparison (see caveat)

# Tests
pytest
```

**Held-out test set:** the test split in `data/splits.csv` is used **only** by `scripts/evaluate.py`.
It must not be used for model selection, hyperparameter tuning, augmentation design, early stopping
or threshold choices. Use `python scripts/evaluate.py --split val` during development.

## Evaluation

`scripts/evaluate.py` writes to `reports/`:

- `metrics.json`: all metrics, machine-readable
- `per_class.csv`: precision, recall, F1 and number of test images per class
- `fitzpatrick_metrics.csv`: n, accuracy (95% bootstrap CI), balanced accuracy, macro
  precision/recall/F1, top-3 accuracy, for Fitzpatrick I…VI, lighter (I–III), darker (IV–VI) and unknown
- `per_class_recall_by_tone.csv`: per-class recall for lighter vs. darker skin
- `confusion_matrix.csv/.png`, `fitzpatrick_accuracy.png`, `predictions.csv`

Overall metrics: accuracy, balanced accuracy, macro precision/recall/F1, top-3 accuracy, confusion
matrix, and calibration (expected calibration error and a reliability table).

Fairness metrics: `accuracy_gap = lighter_accuracy − darker_accuracy` (positive means worse on darker
skin), plus the same gap for balanced accuracy, macro recall, macro F1 and top-3 accuracy, and a
bootstrap 95% CI for the accuracy gap. Groups with fewer than 100 test images are flagged
`reliable: false`. Macro metrics inside a subgroup cover only the classes present in that subgroup.
**A small or statistically insignificant gap is not evidence of fairness.** With these sample sizes
the study can only detect large differences.

RESULTS_PLACEHOLDER

## Running the app

```bash
streamlit run app.py
```

The app loads `models/skin_mobilenetv2.keras` + `models/class_names.json`. If those don't exist, it
loads the legacy model and says so in a banner. `SKIN_MODEL_PATH` / `SKIN_MODEL_META_PATH` override
the paths. If loading fails, the app shows an error and makes no predictions. It never falls back
to an untrained network. The app shows the top 3 model outputs, an uncertainty message when the top
score is below 50% or the scores are spread out (these cut-offs are UX choices, not clinical
thresholds), and a research disclaimer. It gives no treatment advice.

## Legacy model

`models/finetuned_mobilenetv2.h5` (and `non_fine_tuned_mobilenetv2.h5`) are the original 2024
models, kept for comparison. The scripts that produced them are in `scripts/legacy/` (see its README).
Despite its name, the "fine-tuned" model has a frozen ImageNet backbone (verified: its backbone
weights are identical to ImageNet's). It was evaluated on a split with augmentation leakage, so the
previously reported ~56% accuracy is not a valid held-out estimate. Its class order was reconstructed as
the alphabetical `LabelEncoder` order in `models/finetuned_mobilenetv2_class_names.json`. The old
app's hard-coded list had "granuloma annulare" and "granuloma pyogenic" swapped (and the old app never
loaded the trained weights at all).

## Citation

```
@inproceedings{groh2021evaluating,
  title={Evaluating deep neural networks trained on clinical images in dermatology with the fitzpatrick 17k dataset},
  author={Groh, Matthew and Harris, Caleb and Soenksen, Luis and Lau, Felix and Han, Rachel and Kim, Aerin and Koochek, Arash and Badri, Omar},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1820--1828},
  year={2021}
}
```
