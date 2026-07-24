# Dermatology AI Fairness Auditor

A tool for auditing **other people's** dermatology AI models for accuracy
disparities across Fitzpatrick skin-tone groups. You supply a CSV of a
model's predictions alongside ground-truth labels and skin-tone scores; this
tool computes accuracy/precision/recall/F1 overall and per skin-tone group,
and flags statistically significant gaps.

**This is not a diagnostic tool.** It does not analyze images, does not
produce medical predictions, and makes no medical claims about any patient
or condition. It only evaluates prediction/ground-truth pairs that you
already generated elsewhere, from a model you already have. That scope —
auditing existing outputs rather than generating new diagnoses — is
deliberate: it is what keeps this tool out of medical-device regulatory
territory while still surfacing a genuinely important problem, since
dermatology AI models are frequently under-tested on darker skin tones.

## Quick start

```bash
pip install -r requirements.txt
streamlit run audit_app.py
```

The app has two tabs: upload your own predictions CSV, or load a bundled
demo dataset to see the report format (see [Limitations](#limitations) below
— the bundled demo is synthetic, not a real measurement).

## Predictions CSV schema

| Column              | Required | Type          | Notes                                              |
|---------------------|----------|---------------|-----------------------------------------------------|
| `image_id`           | yes      | string        | must be unique                                     |
| `true_label`          | yes      | string        | ground-truth condition label                       |
| `predicted_label`     | yes      | string        | the audited model's predicted label                |
| `fitzpatrick_scale`   | yes      | integer 1-6   | `-1` or blank means "unknown"; unknown rows are excluded from per-group stats |
| `confidence`          | no       | float 0-1     | the audited model's confidence score, if available |

Full validation rules live in `fairness_audit/schema.py`. The app's "Expected
CSV format" panel also lets you download a sample template.

## Project structure

- `audit_app.py` — the Streamlit fairness-audit UI (upload-your-own and demo tabs).
- `fairness_audit/` — the UI-agnostic audit engine:
  - `schema.py` — loads and validates a predictions CSV against the schema above.
  - `grouping.py` — maps raw Fitzpatrick scale values (1-6) to skin-tone groups (light/mid/dark, or a light/dark binary split).
  - `metrics.py` — per-group accuracy/precision/recall/F1 and cross-group disparity/significance testing.
  - `tests/` — unit tests for the above.
- `scripts/generate_demo_predictions.py` — generates a *real* demo dataset by running the two saved `.h5` models on real downloaded images. Written and documented, but not yet actually run end-to-end (see Limitations).
- `scripts/generate_synthetic_demo_predictions.py` — generates the synthetic fallback dataset currently bundled with the app.
- `data/` — the FitzPatrick17k dataset CSV and the bundled synthetic demo predictions.
- `models/` — the two saved `.h5` models (`finetuned_mobilenetv2.h5`, `non_fine_tuned_mobilenetv2.h5`) from the original course project, retained only to generate illustrative demo predictions for this tool (see Background).
- `legacy/app.py` — the original diagnostic-app prototype, superseded by `audit_app.py`. Kept for reference only; see the header comment in that file.
- `scripts/basicmodel.py`, `scripts/dataset.py`, `scripts/features.py`, `scripts/fine_tuned_model.py`, `scripts/non_fine_tuned_model.py` — the original course project's data pipeline and model-training scripts (SVM baseline + the two MobileNetV2 variants). Not part of the audit tool; retained as-is for reference.

## Background

This repository started as a course project that trained image classifiers
(an SVM baseline and two MobileNetV2 variants) to diagnose skin conditions
from photos, using the FitzPatrick17k dataset. That original diagnostic app
is preserved at `legacy/app.py`.

It has since been repositioned as a fairness-auditing tool. Diagnosing
patients from photos at the accuracy those models achieved (see below) is a
real regulatory and liability problem; auditing *someone else's* model's
already-generated predictions is not. The one part of the original project
worth carrying forward was its attention to dark-skin-tone representation —
an area dermatology AI still under-serves — so that became the whole point
of the new tool instead of a side note in a diagnosis app.

The original models and training scripts (`models/`, `scripts/basicmodel.py`,
`scripts/dataset.py`, `scripts/features.py`, `scripts/fine_tuned_model.py`,
`scripts/non_fine_tuned_model.py`) are kept only as historical artifacts —
currently used solely to generate illustrative demo predictions for this
tool's "try the demo" tab — not as a product in their own right. Their
original reported test accuracies were: SVM baseline ~22%, non-fine-tuned
MobileNetV2 ~42%, "fine-tuned" MobileNetV2 ~56%.

## Limitations

Be aware of two things before relying on anything from this repo's demo
data or model artifacts:

- **The bundled demo dataset is synthetic.** `data/demo_predictions_synthetic.csv`
  uses real label and Fitzpatrick-scale distributions from the FitzPatrick17k
  dataset, but the predicted labels and confidences are simulated (with a
  deliberately injected skin-tone accuracy gap), not real model output. The
  script that would produce a real demo dataset by running the two saved
  `.h5` models on real downloaded images, `scripts/generate_demo_predictions.py`,
  is written and documented but has not actually been run successfully yet —
  it depends on downloading images from `atlasdermatologico.com.br`, which
  isn't reachable from the sandbox this was developed in. The app clearly
  labels the demo tab as synthetic/illustrative for this reason.
- **The "fine-tuned" model doesn't actually fine-tune anything.** Both
  `scripts/fine_tuned_model.py` and `scripts/non_fine_tuned_model.py` set
  `base_model.trainable = False` before training, so in both cases only the
  small classification head was trained — the MobileNetV2 backbone was never
  updated. This was confirmed by diffing the saved backbone weights in both
  `.h5` files against a freshly downloaded ImageNet MobileNetV2: they are
  bit-identical. The "fine-tuned" and "non-fine-tuned" models differ only in
  their classification head, not the backbone, despite the naming. (See the
  docstring of `scripts/generate_demo_predictions.py` for the full
  investigation.)

## Acknowledgments

Data sourced from the FitzPatrick17k dataset (Matt Groh et al.):
https://github.com/mattgroh/fitzpatrick17k

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
