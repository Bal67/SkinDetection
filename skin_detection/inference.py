"""Inference helpers shared by the Streamlit app and evaluation.

Softmax outputs are NOT calibrated probabilities of disease. They are relative scores among the
supported classes only; an image of anything (including healthy skin or a non-skin photo) will
still be assigned to one of those classes.
"""

from typing import List, Sequence, Tuple

import numpy as np

from .preprocessing import preprocess

# UX/research display thresholds only. They are NOT clinically derived and carry no diagnostic
# meaning; they only decide when the app shows an extra "the model is uncertain" message.
LOW_TOP1_PROBABILITY = 0.50
HIGH_NORMALIZED_ENTROPY = 0.60


def top_k(probs: Sequence[float], class_names: Sequence[str], k: int = 3) -> List[Tuple[str, float]]:
    probs = np.asarray(probs, dtype=np.float64).ravel()
    if probs.shape[0] != len(class_names):
        raise ValueError(f"{probs.shape[0]} probabilities but {len(class_names)} class names")
    k = min(k, len(class_names))
    order = np.argsort(-probs, kind="stable")[:k]
    return [(class_names[i], float(probs[i])) for i in order]


def normalized_entropy(probs: Sequence[float]) -> float:
    """Shannon entropy / log(num_classes): 0 = all mass on one class, 1 = uniform."""
    p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def is_uncertain(probs: Sequence[float]) -> bool:
    return float(np.max(probs)) < LOW_TOP1_PROBABILITY or normalized_entropy(probs) > HIGH_NORMALIZED_ENTROPY


def predict_probs(model, meta: dict, images) -> np.ndarray:
    """images: iterable of RGB PIL images -> (n, num_classes) probabilities.
    Preprocessing comes from the model's metadata, i.e. exactly what it was trained with."""
    batch = np.stack([preprocess(img, meta["image_size"], meta["resize_mode"], meta["normalization"])
                      for img in images])
    if meta.get("backbone") == "panderm_base":  # PyTorch model
        from . import panderm

        probs = panderm.predict_probs(model, batch, next(model.parameters()).device)
    else:
        probs = np.asarray(model.predict(batch, verbose=0))
    if probs.shape[1] != len(meta["class_names"]):
        raise ValueError("Model output size does not match class_names")
    return probs
