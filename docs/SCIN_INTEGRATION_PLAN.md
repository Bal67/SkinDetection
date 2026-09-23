# SCIN integration plan (not implemented)

This repository currently trains and evaluates **only on Fitzpatrick17k**. Nothing from Google's
SCIN dataset has been downloaded or mixed in. This note describes what adding SCIN (or another
dermatology dataset with better skin-tone diversity) would require, and what must not be skipped.

## 1. Schema adapter

`skin_detection/data.py` defines a common schema, one row per original image:

| column | Fitzpatrick17k source | SCIN (expected) |
|---|---|---|
| `image_id` | `md5hash` | per-image id (SCIN cases can hold up to 3 images) |
| `label` | `label` (one of 114 conditions) | dermatologist differential (weighted list of conditions per case) |
| `fitzpatrick` | `fitzpatrick_scale` (annotator estimate) | self-reported FST and/or dermatologist-estimated FST |
| `source` | `"fitzpatrick17k"` | `"scin"` |
| `url` | atlas URL | GCS path |
| `group_id` | `image_id` (+ near-duplicate merge) | **`case_id`** (all images of one case share it) |

Work needed: a `load_scin(...)` function returning these columns, plus extra columns worth
keeping for analysis (Monk skin tone, body site, image shot type, self-reported race/ethnicity).
Check SCIN's licence and terms of use before downloading, and record them in the README.

## 2. Labels differ and cannot be assumed equivalent

- SCIN labels are **differentials with confidence weights** from several dermatologists, not a single
  label. A policy is needed: e.g. keep a case only when the top-weighted condition is above a threshold
  and dermatologists agree; record the policy and how many cases it drops.
- Vocabularies differ (naming, granularity, synonyms, umbrella terms such as "eczema" vs. specific
  dermatitis types). Build an explicit, reviewed mapping table
  (`data/label_map_scin_to_f17k.csv`: source label, target label, mapping type exact/broader/narrower/none,
  reviewer note). Unmapped labels stay out; they are never silently merged.
- SCIN is consumer-captured (phone photos of common, often non-severe concerns); Fitzpatrick17k is
  atlas images of often textbook presentations. Even after harmonization, the same label name can
  refer to a different visual distribution. **Label harmonization should be done or reviewed by
  someone with dermatology expertise.** A wrong mapping injects label noise that is concentrated in
  specific classes and can look like a skin-tone effect.
- Many of the current 26 classes (e.g. tungiasis, myiasis, Ehlers-Danlos) likely have few or no SCIN
  examples; conversely SCIN covers conditions absent here. Decide per class; do not force coverage.

## 3. Duplicates and overlap

- Within SCIN: split by `case_id` so the images of one case never cross partitions.
- Across datasets: run the existing perceptual-hash grouping (`--near-duplicates`) over the **union**
  of both datasets before splitting, so a re-used image can't sit in F17k-train and SCIN-test.
  Overlap is unlikely (different sources) but the check is cheap.

## 4. Avoiding train/test leakage between datasets

- Build one combined split file where `group_id` is namespaced (`scin:<case_id>`,
  `f17k:<md5hash>`), and use the existing `split_by_group` + `check_no_leakage`.
- Keep a **frozen per-dataset test set** as well, and always report metrics per source dataset.
  Otherwise an improvement could come from dataset composition rather than from better
  performance on darker skin.
- Do not re-split the existing Fitzpatrick17k test set; add SCIN test cases alongside it so the
  F17k numbers stay comparable with earlier runs.

## 5. Skin-tone annotation normalization

- Fitzpatrick17k: `fitzpatrick_scale` is a crowd annotation from the image. Its second annotation
  (`fitzpatrick_centaur`) agrees only ~46% of the time.
- SCIN: self-reported Fitzpatrick type (from the contributor), dermatologist/labeller estimates, and
  Monk Skin Tone (10-point). These measure different things. Self-reported FST describes a
  sun-reaction phenotype; image-estimated FST and MST describe apparent tone in a photo.
- Store each annotation in its own column (`fst_self`, `fst_estimated`, `monk`) and say which one the
  fairness report stratifies by. Do not average them, and do not convert MST to FST unless a
  documented and validated mapping exists.
- Keep "unknown" as its own group; never impute skin tone.

## 6. Order of work

1. Adapter + licence review, no training changes.
2. Label map with expert review; report per-class counts per dataset and per skin-tone group.
3. Combined split, leakage checks, then the first model trained on F17k only, evaluated on the SCIN test set
   (a domain-shift baseline), before any mixed training.
