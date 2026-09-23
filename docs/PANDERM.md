# PanDerm_Base in this project

PanDerm is a dermatology foundation model: a ViT pretrained with self-supervised learning on more than
2 million dermatology images across four imaging modalities (Yan et al., *Nature Medicine* 2025).
This project uses **PanDerm_Base (ViT-B/16)** as the dermatology-specific model and compares it with
the TensorFlow baselines on the **same leakage-safe split** (`data/splits.csv`, same 352-image test set).

| role | model | framework |
|---|---|---|
| historical baseline | MobileNetV2 | TensorFlow |
| modern general-purpose CNN baseline | EfficientNetV2-B0 | TensorFlow |
| dermatology-specific model | PanDerm_Base, linear probe and partial fine-tuning | PyTorch |

## Source, checkpoint, licence

- Official repository: https://github.com/SiyuanYan1/PanDerm, pinned to commit
  `fd7a80748ba7fc3e203fed88f909f4689d0d6f24`.
- Checkpoint: `panderm_bb_data6_checkpoint-499.pth` (PanDerm_Base, ViT-B/16, released 04/2025), from
  the official Google Drive link in the repository README (file id `17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB`).
  343 MB, SHA-256 `be1e0fb108b3bc58721cb5195f136c948160799438f222acf1fd142230ac1ff1`. The loader
  refuses any other file.
- **Licence: CC-BY-NC-ND 4.0** (code and weights). Non-commercial academic research only, with
  attribution, and no distribution of adapted material. Consequences for this repo:
  - the official code is **not copied** into this repository. `skin_detection/panderm.py` imports the
    official `classification/models/modeling_finetune.py` from a local clone in `third_party/PanDerm`
    (gitignored);
  - the pretrained weights, the trained heads/fine-tuned blocks (`models/panderm_base/*.pt`) and the cached
    features (`data/features/`) are **gitignored**. Do not publish them;
  - any commercial use of the PanDerm results would need separate permission from the authors.

## Setup

```bash
python3.10 -m venv .venv-panderm && source .venv-panderm/bin/activate    # 3.10 as upstream; 3.11 tested
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA; omit on Mac
pip install -r requirements-panderm.txt

git clone https://github.com/SiyuanYan1/PanDerm third_party/PanDerm
git -C third_party/PanDerm checkout fd7a80748ba7fc3e203fed88f909f4689d0d6f24

gdown 17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB \
      -O models/panderm_base/pretrained/panderm_bb_data6_checkpoint-499.pth
```

If gdown fails (Drive quota), download the file manually from the link in the official README and place it at
`models/panderm_base/pretrained/panderm_bb_data6_checkpoint-499.pth`, or point `SKIN_PANDERM_CHECKPOINT`
at it. `SKIN_PANDERM_REPO` overrides the code location.

## Commands

```bash
python scripts/train.py    --backbone panderm_base --mode linear_probe
python scripts/train.py    --backbone panderm_base --mode partial_finetune      # CUDA: add --grad-checkpointing if needed
python scripts/evaluate.py --backbone panderm_base --mode linear_probe --split val   # development
python scripts/evaluate.py --backbone panderm_base --mode linear_probe             # final, test split
```

Outputs: `models/panderm_base/class_names.json` (shared class order + preprocessing),
`models/panderm_base/<mode>_best.pt` (only the trained tensors: head, plus unfrozen blocks for partial
fine-tuning; the rest is always the official checkpoint), `models/panderm_base/<mode>_meta.json`, and
`reports/panderm_base_<mode>/`.

## What follows the official implementation, and what deliberately does not

| aspect | official PanDerm | here |
|---|---|---|
| architecture | `panderm_base_patch16_224` / `_finetune` | same (imported from the official file) |
| weight loading | `strict=False` | strict: any missing/unexpected key is an error |
| eval preprocessing | Resize(256) → CenterCrop(224) → mean (0.485, 0.456, 0.406), std (0.228, 0.224, 0.225) | identical; verified against torchvision to 1e-5 (std 0.228 is the official value; standard ImageNet uses 0.229) |
| linear-probe features | final-norm CLS token | same |
| linear probe | L-BFGS logistic regression, 1000 iterations, cost C = 768·K/100, no validation use, no class weighting | same solver/objective. C is **selected on validation** from {0.01, 0.1, 1, 10, 100, 1000} × official (100× won; the first grid topped out at 10×). Class- and skin-tone-balanced sample weights in the loss |
| fine-tuning features | mean-pooled patch tokens + `fc_norm` (checkpoint `norm` → `fc_norm`) | same |
| fine-tuned layers | all 12 blocks, layer decay 0.65 | **last 2 blocks** + final norm + head (`--trainable-blocks`). Layer decay 0.65 within those; backbone LR 2e-5 vs head LR 1e-3 |
| relative position bias | adds zero-initialised tables (`rel_pos_bias=True`) | off: the checkpoint has none, and in frozen blocks they would stay zero |
| optimiser | AdamW, wd 0.05, warmup + cosine, drop path 0.2, label smoothing 0.1 | same values (1 warmup epoch, 15 epochs max, early stopping patience 4 on validation balanced accuracy) |
| class balancing | `WeightedRandomSampler`, inverse class frequency | `WeightedRandomSampler` with this project's class + skin-tone balanced weights |
| train augmentation | RandomResizedCrop(0.75–1), H+V flips, rotation 45°, ColorJitter(hue 0.2), Mixup 0.8, CutMix 1.0 | RandomResizedCrop(0.75–1), H flip, rotation 14°, brightness/contrast ±10%. **No hue jitter, no vertical flip, no mixup/cutmix**: this project does not recolor skin (see README) |
| batch | 128 | physical 8 × gradient accumulation 8 = 64 |
| test-time augmentation | optional (`--TTA`) | none |

## Hardware

Targeted at a single 16 GB CUDA GPU: mixed precision (bfloat16 where supported, otherwise float16 with
a GradScaler), physical batch 8 (use `--batch-size 4 --grad-accum 16` if memory is short), frozen blocks
run under `no_grad` so only the unfrozen blocks keep activations, and optional `--grad-checkpointing`.
Inference is batched under `torch.inference_mode()`.

The results in the README were produced on an **Apple M2 (16 GB unified memory) with the MPS backend in
float32**. This machine has no CUDA GPU, so CUDA memory use and mixed-precision numerics have **not**
been measured yet.
