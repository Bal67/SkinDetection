| Model | status | n_test | Accuracy | Balanced Accuracy | Macro F1 | Top-3 | Light I-III acc | Light n | Light macro F1 | Dark IV-VI acc | Dark n | Dark macro F1 | Gap (acc, light-dark) | Gap 95% CI | Macro-F1 gap | Case-mix controlled recall gap | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MobileNetV2 | test | 352 | 0.469 | 0.415 | 0.410 | 0.713 | 0.448 | 163 | 0.340 | 0.483 | 178 | 0.406 | -0.035 | [-0.135, +0.075] | -0.067 | -0.057 | 0.040 |
| EfficientNetV2-B0 | test | 352 | 0.562 | 0.551 | 0.525 | 0.790 | 0.528 | 163 | 0.484 | 0.590 | 178 | 0.508 | -0.062 | [-0.170, +0.043] | -0.025 | -0.017 | 0.068 |
| PanDerm_Base Linear Probe | test | 352 | 0.759 | 0.741 | 0.751 | 0.932 | 0.730 | 163 | 0.669 | 0.803 | 178 | 0.753 | -0.073 | [-0.164, +0.014] | -0.083 | -0.071 | 0.159 |
| PanDerm_Base Partial FT | test | 352 | 0.659 | 0.693 | 0.655 | 0.864 | 0.650 | 163 | 0.635 | 0.669 | 178 | 0.615 | -0.018 | [-0.120, +0.088] | 0.020 | 0.006 | 0.172 |
