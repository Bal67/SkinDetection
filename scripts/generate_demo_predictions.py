"""
Generate real demo predictions from the two saved baseline models
(non_fine_tuned_mobilenetv2.h5, finetuned_mobilenetv2.h5) for use as a
credible "our own baseline models are biased too" showcase in the fairness
audit demo.

This script:
  1. Loads data/fitzpatrick17k.csv and filters to rows whose image URL is
     NOT hosted on dermaamin.com. dermaamin.com is excluded because the
     repo's own scripts/dataset.py already treats it as an unreliable image
     host (see preprocess_data() there) -- in practice essentially every
     "reliable" URL in this dataset resolves to atlasdermatologico.com.br.
  2. Further restricts to rows whose `label` is one of the 26 condition
     names the saved models were actually trained on (see CLASS_LABELS
     below and the module docstring section "Label-set determination").
  3. Draws a fixed-seed sample stratified by fitzpatrick_scale so all
     skin-tone groups (including the -1 "unknown" bucket) get some
     representation, rather than taking the first N rows.
  4. Downloads each image (short timeout, at most one retry, failures are
     skipped -- never fabricated), preprocesses it to 128x128x3 with
     tf.keras.applications.mobilenet_v2.preprocess_input (matching the
     training scripts), and runs BOTH saved models on it.
  5. Writes data/demo_predictions_finetuned.csv and
     data/demo_predictions_nonfinetuned.csv with columns:
     image_id, true_label, predicted_label, confidence, fitzpatrick_scale.

Label-set determination
------------------------
Both .h5 files were inspected directly (see scripts/ used during
investigation, not checked in) via h5py weight shapes and
tf_keras/tf.keras loading:
  - Input shape: (None, 128, 128, 3) for both models.
  - Final Dense layer: 26 units, softmax, for BOTH models
    (models/non_fine_tuned_mobilenetv2.h5: dense_1 kernel (256, 26);
     models/finetuned_mobilenetv2.h5: dense_5 kernel (256, 26)).

26 does not match len(df['label'].unique()) == 114, nor
nine_partition_label (9), nor three_partition_label (3). However, all 26
condition strings hardcoded in app.py's `conditions` list are an EXACT
(zero missing) match against 26 distinct values of the CSV's `label`
column. That is very strong (though not 100%-certain) evidence these are
the actual training classes -- it would be a wild coincidence for the app
author to hand-type 26 label strings, all of which happen to appear
verbatim in the dataset's free-text label column, if these weren't the
real class list used when the model was built.

Class ORDER is a separate, less certain question. Both training scripts
(scripts/fine_tuned_model.py, scripts/non_fine_tuned_model.py) build labels
with `sklearn.preprocessing.LabelEncoder().fit_transform(df['label'])`,
which assigns class indices via `numpy.unique` on the label strings, i.e.
ascending alphabetical order. app.py's hardcoded `conditions` list is NOT
alphabetically sorted (e.g. it lists "granuloma pyogenic" before
"granuloma annulare"), so it cannot be a faithful reproduction of the
LabelEncoder's class order -- it's likely just the author's own ad hoc
ordering of the same 26 names. We therefore use the ALPHABETICALLY SORTED
list as CLASS_LABELS here, since that is what the training code would
actually have produced, not app.py's order. This is the one genuinely
ambiguous piece of this determination; flagged here and in the script's
printed summary.
"""

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd
import requests
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "data", "fitzpatrick17k.csv")
MODEL_NONFT_PATH = os.path.join(REPO_ROOT, "models", "non_fine_tuned_mobilenetv2.h5")
MODEL_FT_PATH = os.path.join(REPO_ROOT, "models", "finetuned_mobilenetv2.h5")
OUT_FT_PATH = os.path.join(REPO_ROOT, "data", "demo_predictions_finetuned.csv")
OUT_NONFT_PATH = os.path.join(REPO_ROOT, "data", "demo_predictions_nonfinetuned.csv")

IMG_SIZE = (128, 128)

# See "Label-set determination" above: alphabetically sorted to match
# sklearn LabelEncoder's fit_transform ordering used by the training scripts.
CLASS_LABELS = sorted([
    "allergic contact dermatitis",
    "basal cell carcinoma",
    "dariers disease",
    "ehlers danlos syndrome",
    "erythema multiforme",
    "folliculitis",
    "granuloma pyogenic",
    "granuloma annulare",
    "hailey hailey disease",
    "kaposi sarcoma",
    "keloid",
    "lichen planus",
    "lupus erythematosus",
    "melanoma",
    "mycosis fungoides",
    "myiasis",
    "nematode infection",
    "neutrophilic dermatoses",
    "photodermatoses",
    "pityriasis rosea",
    "psoriasis",
    "scabies",
    "scleroderma",
    "squamous cell carcinoma",
    "tungiasis",
    "vitiligo",
])
assert len(CLASS_LABELS) == 26


def _find_kernel_bias(h5file, group_name):
    """Recursively pull the `kernel` and `bias` datasets out of a saved
    Dense layer's weight group, regardless of the exact nesting/':0' suffix
    conventions used by the Keras version that wrote the file."""
    import h5py
    kernel = bias = None

    def visit(name, obj):
        nonlocal kernel, bias
        if isinstance(obj, h5py.Dataset):
            base = name.split("/")[-1].split(":")[0]
            if base == "kernel":
                kernel = obj[:]
            elif base == "bias":
                bias = obj[:]

    h5file["model_weights"][group_name].visititems(visit)
    return kernel, bias


def load_keras_model_compat(path, dense256_group, dense26_group):
    """Reconstruct one of the saved models and load its trained weights.

    Directly deserializing these .h5 files with the installed Keras 3 (both
    natively and via the legacy `tf_keras` shim) fails on this TF/Keras
    version -- the files mix old-format quirks (BatchNormalization `axis`
    saved as a length-1 list; nested Sequential-wrapping-Functional configs
    that Keras 3's Sequential.from_config mis-rebuilds, producing duplicate
    inbound tensors). Both are Keras-version-skew issues in the *full
    model* deserialization path, not problems with the underlying weights.

    Both training scripts (scripts/fine_tuned_model.py,
    scripts/non_fine_tuned_model.py) set `base_model.trainable = False`
    *before* compiling/fitting, in both the "fine-tuned" and
    "non-fine-tuned" scripts -- so despite the filename, neither script
    actually updates the MobileNetV2 convolutional backbone; only the
    Dense(256)->Dropout->Dense(26) head was trained. This was verified by
    diffing the saved Conv1/Conv_1 kernel weights against a freshly
    downloaded ImageNet-pretrained MobileNetV2 -- they are bit-identical in
    both .h5 files. So we rebuild the exact architecture from
    scripts/*_model.py with a fresh ImageNet backbone (sidestepping the
    fragile full-model deserialization entirely) and load only the two head
    Dense layers' weights directly out of the .h5 via h5py.
    """
    import h5py
    import tensorflow as tf

    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(CLASS_LABELS), activation="softmax"),
    ])
    model.build((None, 128, 128, 3))

    with h5py.File(path, "r") as h:
        k256, b256 = _find_kernel_bias(h, dense256_group)
        k26, b26 = _find_kernel_bias(h, dense26_group)
    if k256 is None or k26 is None:
        raise ValueError(f"Could not locate head layer weights in {path} "
                          f"(groups {dense256_group!r}, {dense26_group!r})")
    model.layers[-3].set_weights([k256, b256])
    model.layers[-1].set_weights([k26, b26])
    return model


def load_and_filter_dataframe():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["url"])
    df = df[~df["url"].str.contains("dermaamin.com")]
    df = df[df["label"].isin(CLASS_LABELS)]
    return df


def stratified_sample(df, n, seed):
    """Sample up to n rows, stratified proportionally across
    fitzpatrick_scale (including the -1 'unknown' bucket) so no group is
    silently dropped, rather than taking the first n rows."""
    groups = df.groupby("fitzpatrick_scale", group_keys=False)
    total = len(df)
    parts = []
    for scale, g in groups:
        quota = max(1, round(n * len(g) / total))
        quota = min(quota, len(g))
        parts.append(g.sample(n=quota, random_state=seed))
    sample = pd.concat(parts)
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed)
    return sample.sample(frac=1, random_state=seed).reset_index(drop=True)


def download_image(url, timeout):
    """Download + decode an image. One retry on failure, then give up."""
    for attempt in range(2):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception:
            if attempt == 1:
                return None
    return None


def preprocess(img):
    import tensorflow as tf
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=700,
                         help="Target number of candidate rows to attempt downloading (default 700).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=8.0, help="Per-request timeout in seconds.")
    args = parser.parse_args()

    df = load_and_filter_dataframe()
    print(f"Rows available after dermaamin/label-set filtering: {len(df)}")
    if len(df) == 0:
        print("No candidate rows available -- aborting.")
        sys.exit(1)

    sample = stratified_sample(df, min(args.sample_size, len(df)), args.seed)
    print(f"Sampled {len(sample)} candidate rows (stratified by fitzpatrick_scale, seed={args.seed}).")

    print("Loading models...")
    # Head-layer weight group names as stored in each specific .h5 file
    # (differ because they were saved by different Keras/TF versions --
    # see load_keras_model_compat docstring).
    model_nonft = load_keras_model_compat(MODEL_NONFT_PATH, "dense", "dense_1")
    model_ft = load_keras_model_compat(MODEL_FT_PATH, "dense_4", "dense_5")
    print(f"non-finetuned output shape: {model_nonft.output_shape}")
    print(f"finetuned output shape: {model_ft.output_shape}")

    rows_ft, rows_nonft = [], []
    attempted = 0
    succeeded = 0
    for _, row in sample.iterrows():
        attempted += 1
        img = download_image(row["url"], args.timeout)
        if img is None:
            continue
        try:
            arr = preprocess(img)
        except Exception:
            continue

        batch = np.expand_dims(arr, axis=0)
        fitz = int(row["fitzpatrick_scale"]) if pd.notna(row["fitzpatrick_scale"]) else -1

        pred_ft = model_ft.predict(batch, verbose=0)[0]
        pred_nonft = model_nonft.predict(batch, verbose=0)[0]

        rows_ft.append({
            "image_id": row["md5hash"],
            "true_label": row["label"],
            "predicted_label": CLASS_LABELS[int(np.argmax(pred_ft))],
            "confidence": float(np.max(pred_ft)),
            "fitzpatrick_scale": fitz,
        })
        rows_nonft.append({
            "image_id": row["md5hash"],
            "true_label": row["label"],
            "predicted_label": CLASS_LABELS[int(np.argmax(pred_nonft))],
            "confidence": float(np.max(pred_nonft)),
            "fitzpatrick_scale": fitz,
        })
        succeeded += 1
        if succeeded % 25 == 0:
            print(f"  ... {succeeded} images processed ({attempted} attempted)")

    pd.DataFrame(rows_ft).to_csv(OUT_FT_PATH, index=False)
    pd.DataFrame(rows_nonft).to_csv(OUT_NONFT_PATH, index=False)

    print("\n=== Summary ===")
    print(f"Attempted downloads: {attempted}")
    print(f"Successfully processed (both models run): {succeeded}")
    print(f"Class set used ({len(CLASS_LABELS)} classes): {CLASS_LABELS}")
    print(f"Wrote {len(rows_ft)} rows to {OUT_FT_PATH}")
    print(f"Wrote {len(rows_nonft)} rows to {OUT_NONFT_PATH}")


if __name__ == "__main__":
    main()
