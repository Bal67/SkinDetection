# Legacy scripts (2024, Colab)

Kept for reference only. They produced `models/finetuned_mobilenetv2.h5` and
`models/non_fine_tuned_mobilenetv2.h5`. **Do not use them for new experiments.** Known problems:

- `features.py` writes augmented copies (flips, color inversion, hue jitter) as new rows *before*
  the split, and the training scripts split those rows at random. Augmented copies of one original
  image end up in train, validation and test (data leakage), which inflates the reported accuracy.
- Color inversion and hue jitter (hue=0.2) were applied only to darker-skin images, producing
  non-physiological skin colors.
- The training scripts undersample every class to the size of the rarest class, discarding about 76%
  of the augmented rows.
- `fine_tuned_model.py` sets `base_model.trainable = False`: this is frozen-backbone transfer
  learning, not fine-tuning. It is architecturally identical to `non_fine_tuned_model.py`.
- Hard-coded Colab/Google Drive paths and bucket name; `dataset.py` references an undefined
  global `s3_client` inside `upload_to_s3` when imported.
- Only accuracy (and precision/recall for the SVM) was reported, with no skin-tone stratification.

The replacement pipeline is `scripts/prepare_dataset.py`, `scripts/train.py` and `scripts/evaluate.py`.
