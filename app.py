"""Streamlit research demo.  Run with:  streamlit run app.py

Loads the trained model + its models/skin_<backbone>_class_names.json once, validates that they agree, and shows
the top-3 model outputs. It never builds or uses an untrained network: if loading fails, the
app shows the error and stops.
"""

import os
from pathlib import Path

import streamlit as st

from skin_detection import config
from skin_detection.inference import (HIGH_NORMALIZED_ENTROPY, LOW_TOP1_PROBABILITY, is_uncertain,
                                      normalized_entropy, predict_probs, top_k)
from skin_detection.model import BACKBONES, ModelLoadError, load_trained_model
from skin_detection.preprocessing import load_image

DISCLAIMER = (
    "This prototype is for research and educational purposes and is not a medical diagnosis. "
    "A healthcare professional should evaluate concerning skin changes."
)


def use_panderm():
    """Opt-in only: MODEL_TYPE=panderm_base (PANDERM_MODE=linear_probe|partial_finetune). Needs the
    PyTorch environment (requirements-panderm.txt). The default app model is unchanged."""
    return os.environ.get("MODEL_TYPE") == "panderm_base"


@st.cache_resource(show_spinner="Loading PanDerm model...")
def get_panderm_model(mode: str):
    from skin_detection import panderm

    return panderm.load_classifier(mode, panderm.pick_device())


def resolve_model_paths():
    """Explicit SKIN_MODEL_PATH wins. Otherwise the configured backbone (SKIN_BACKBONE, default
    EfficientNetV2-B0), then any other retrained backbone, then the legacy model. Whichever is
    used is named in the UI."""
    if "SKIN_MODEL_PATH" in os.environ:
        return config.MODEL_PATH, config.MODEL_META_PATH, False
    for backbone in [config.BACKBONE] + [b for b in BACKBONES if b != config.BACKBONE]:
        model_path, meta_path = config.model_paths(backbone)
        if model_path.exists():
            return model_path, meta_path, False
    return config.LEGACY_MODEL_PATH, config.LEGACY_MODEL_META_PATH, True


@st.cache_resource(show_spinner="Loading model...")
def get_model(model_path: str, meta_path: str):
    return load_trained_model(Path(model_path), Path(meta_path))


st.set_page_config(page_title="Skin Condition Classifier (Research POC)")
st.title("Skin-condition image classifier")
st.caption("Research prototype. Not a diagnostic tool.")
st.warning(DISCLAIMER)

is_legacy = False
try:
    if use_panderm():
        panderm_mode = os.environ.get("PANDERM_MODE", "linear_probe")
        model, meta = get_panderm_model(panderm_mode)
        model_label = f"PanDerm_Base {panderm_mode} (CC-BY-NC-ND 4.0: non-commercial research use only)"
    else:
        model_path, meta_path, is_legacy = resolve_model_paths()
        model, meta = get_model(str(model_path), str(meta_path))
        model_label = f"`{model_path.name}` ({meta.get('backbone', 'legacy MobileNetV2')})"
except (ModelLoadError, ImportError) as exc:
    st.error(f"The model could not be loaded, so no predictions can be made.\n\n{exc}")
    st.stop()

class_names = meta["class_names"]
st.caption(f"Model: {model_label}")

if is_legacy:
    st.info(
        "Using the **legacy 2024 model** (`finetuned_mobilenetv2.h5`). It was trained with a frozen "
        "backbone on a split that leaked augmented copies between train and test, so its reported "
        "accuracy is not trustworthy. Retrain with `scripts/train.py` to use the corrected model."
    )

st.markdown(
    f"The model only knows **{len(class_names)} conditions** from the Fitzpatrick17k dataset. "
    "It will always pick among these, even for an image of healthy skin or something unrelated. "
    "Its outputs describe visual similarity to those categories, not what a person has."
)
with st.expander("Supported conditions"):
    st.write(", ".join(class_names))

uploaded = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        image = load_image(uploaded.getvalue())  # EXIF orientation + RGB conversion
    except ValueError:
        st.error("That file could not be opened as an image. Please upload a JPEG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded image", width="stretch")
    probs = predict_probs(model, meta, [image])[0]

    st.subheader("Possible classifications (model output)")
    for rank, (name, p) in enumerate(top_k(probs, class_names, k=3), start=1):
        st.markdown(f"{rank}. **{name.capitalize()}**: {p:.0%}")
        st.progress(float(p))

    if is_uncertain(probs):
        st.warning(
            "**The model is uncertain about this image.** No category stands out clearly, so these "
            "outputs should not be relied on. The image may show a condition the model does not "
            "cover, or the photo may be unclear."
        )
    with st.expander("How to read these percentages"):
        st.markdown(
            "- Percentages are the model's relative scores among its supported categories. They add up to "
            "100% across those categories and are **not** the probability of having a condition.\n"
            "- Scores from this kind of model are often over-confident, and performance varies by skin "
            "tone and condition (see `reports/`).\n"
            f"- The uncertainty message appears when the top score is below {LOW_TOP1_PROBABILITY:.0%} or "
            f"the scores are spread out (normalized entropy above {HIGH_NORMALIZED_ENTROPY:.2f}; this image: "
            f"{normalized_entropy(probs):.2f}). These cut-offs are display choices for this research demo, "
            "not clinically validated thresholds."
        )

st.divider()
st.caption(DISCLAIMER)
