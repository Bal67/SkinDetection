"""Metadata loading, filtering, duplicate detection, leakage-safe splitting, and image access.

Dataset-specific code is isolated in `load_fitzpatrick17k`, which maps the raw CSV to a common
schema. A future dataset (e.g. SCIN) needs its own loader returning the same columns; see
docs/SCIN_INTEGRATION_PLAN.md.

Common schema (one row per ORIGINAL image; augmentation never creates rows):
    image_id      unique id of the original image (Fitzpatrick17k: md5hash)
    label         condition name (string)
    fitzpatrick   int 1..6, or -1 if unknown
    source        dataset name
    url           original image URL (may be empty)
    group_id      leakage group: all rows sharing a group_id go to the same split
"""

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from . import config

SPLITS = ("train", "val", "test")
COMMON_COLUMNS = ["image_id", "label", "fitzpatrick", "source", "url", "group_id"]


# --------------------------------------------------------------------------- 1. metadata


def load_fitzpatrick17k(csv_path=config.METADATA_CSV) -> pd.DataFrame:
    """Fitzpatrick17k adapter -> common schema.

    Uses the `fitzpatrick_scale` column (the dataset's primary annotation). The dataset also has
    a second annotation (`fitzpatrick_centaur`), kept as `fitzpatrick_centaur` for analysis; the two
    agree on only ~46% of images, which is itself a limitation of skin-tone labels.
    Rows flagged '3 Wrongly labelled' in the dataset's QC column are dropped.
    """
    raw = pd.read_csv(csv_path)
    raw = raw[raw["qc"].fillna("") != "3 Wrongly labelled"]
    df = pd.DataFrame({
        "image_id": raw["md5hash"].astype(str),
        "label": raw["label"].astype(str).str.strip(),
        "fitzpatrick": pd.to_numeric(raw["fitzpatrick_scale"], errors="coerce").fillna(-1).astype(int),
        "fitzpatrick_centaur": pd.to_numeric(raw["fitzpatrick_centaur"], errors="coerce").fillna(-1).astype(int),
        "source": "fitzpatrick17k",
        "url": raw["url"].fillna("").astype(str),
    })
    df.loc[~df["fitzpatrick"].isin([1, 2, 3, 4, 5, 6]), "fitzpatrick"] = -1
    df["group_id"] = df["image_id"]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- 2. label filtering


def filter_labels(df: pd.DataFrame, labels: Optional[Sequence[str]] = None,
                  excluded_domains: Iterable[str] = config.DEFAULT_EXCLUDED_DOMAINS) -> pd.DataFrame:
    df = df[df["label"].notna() & (df["label"] != "")]
    df = df[df["url"] != ""]
    for domain in excluded_domains:
        df = df[~df["url"].str.contains(domain, regex=False)]
    if labels is not None:
        missing = set(labels) - set(df["label"])
        if missing:
            raise ValueError(f"Requested labels not present after filtering: {sorted(missing)}")
        df = df[df["label"].isin(labels)]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- 3. duplicates


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a repeated image_id or repeated source URL (same file twice)."""
    df = df.drop_duplicates(subset="image_id", keep="first")
    has_url = df["url"] != ""
    df = pd.concat([df[has_url].drop_duplicates(subset="url", keep="first"), df[~has_url]])
    return df.reset_index(drop=True)


def dhash(img, hash_size: int = 16) -> np.ndarray:
    """Difference hash of a PIL image -> bool array of hash_size**2 bits.

    16x16 (256 bits) rather than the common 8x8: on Fitzpatrick17k an 8x8 hash chained 41
    unrelated images from 17 conditions into one group; at 256 bits genuine re-uploads sit at
    distance <= 15 while the closest unrelated pair is at 18.
    """
    gray = img.convert("L").resize((hash_size + 1, hash_size))
    px = np.asarray(gray, dtype=np.int16)
    return (px[:, 1:] > px[:, :-1]).flatten()


def group_near_duplicates(image_ids: Sequence[str], hashes: np.ndarray, max_distance: int = 15) -> dict:
    """Union-find over perceptual hashes. Images whose dHash Hamming distance <= max_distance
    (e.g. re-crops/re-encodes of the same photo) share a group. Returns {image_id: group_id}."""
    n = len(image_ids)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    h = np.asarray(hashes, dtype=bool)
    for i in range(n):
        dist = (h[i + 1:] != h[i]).sum(axis=1)
        for j in np.nonzero(dist <= max_distance)[0] + i + 1:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    return {image_ids[i]: image_ids[find(i)] for i in range(n)}


# --------------------------------------------------------------------------- 4. split


def split_by_group(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15,
                   seed: int = config.SEED) -> pd.DataFrame:
    """Assign every row to train/val/test such that each group_id lands in exactly one split.

    Splitting happens on ORIGINAL images before any augmentation, so augmented views of a
    training image can never appear in validation or test. Stratified by label at group level
    (label of a multi-image group = its most common label).

    Fitzpatrick17k has no patient or lesion identifier, so the finest leakage group available
    is the image itself (md5hash), optionally merged with perceptual near-duplicates. Different
    photos of the same patient can therefore still fall into different splits (the atlas numbers
    images sequentially within each disease, so image-ID adjacency cannot recover patients).
    """
    from sklearn.model_selection import train_test_split

    groups = df.groupby("group_id")["label"].agg(lambda s: s.value_counts().index[0]).reset_index()
    strat = groups["label"].copy()
    counts = strat.value_counts()
    strat[strat.map(counts) < 3] = "__rare__"  # too few groups to stratify individually
    if (strat == "__rare__").sum() == 1:
        strat[strat == "__rare__"] = strat.mode()[0]

    trainval, test = train_test_split(groups, test_size=test_frac, random_state=seed, stratify=strat)
    strat_tv = strat.loc[trainval.index]
    counts_tv = strat_tv.value_counts()
    strat_tv = strat_tv.where(strat_tv.map(counts_tv) >= 2, strat_tv.mode()[0])
    train, val = train_test_split(trainval, test_size=val_frac / (1 - test_frac),
                                  random_state=seed, stratify=strat_tv)
    assignment = {**dict.fromkeys(train["group_id"], "train"),
                  **dict.fromkeys(val["group_id"], "val"),
                  **dict.fromkeys(test["group_id"], "test")}
    out = df.copy()
    out["split"] = out["group_id"].map(assignment)
    check_no_leakage(out)
    return out


def check_no_leakage(df: pd.DataFrame) -> None:
    """Raise if any image_id or group_id appears in more than one split."""
    for col in ("image_id", "group_id"):
        n_splits = df.groupby(col)["split"].nunique()
        leaked = n_splits[n_splits > 1]
        if len(leaked):
            raise AssertionError(f"{len(leaked)} {col}s appear in more than one split, e.g. {leaked.index[:3].tolist()}")
    if df["split"].isna().any() or not set(df["split"]) <= set(SPLITS):
        raise AssertionError("Every row must be assigned to exactly one of train/val/test")


def load_splits(path=config.SPLITS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"image_id": str, "group_id": str})
    check_no_leakage(df)
    return df


# --------------------------------------------------------------------------- 5. images


def local_image_path(image_id: str, images_dir=config.IMAGES_DIR) -> Path:
    return Path(images_dir) / f"{image_id}.jpg"


def fetch_from_url(url: str, timeout: int = 15) -> bytes:
    import requests

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "SkinDetection-research-POC"})
    resp.raise_for_status()
    return resp.content


def fetch_from_s3(image_id: str, bucket: str, prefix: str = config.S3_PREFIX) -> bytes:
    """Read <prefix><image_id>.jpg from S3 using the default AWS credential chain."""
    import boto3

    obj = boto3.client("s3").get_object(Bucket=bucket, Key=f"{prefix}{image_id}.jpg")
    return obj["Body"].read()


# --------------------------------------------------------------------------- 6. skin-tone helpers


def tone_group(fitzpatrick: int) -> str:
    if fitzpatrick in config.LIGHTER_TYPES:
        return "lighter_I-III"
    if fitzpatrick in config.DARKER_TYPES:
        return "darker_IV-VI"
    return "unknown"


# --------------------------------------------------------------------------- 7. imbalance


def compute_sample_weights(df: pd.DataFrame, class_names: Sequence[str], tone_balance: bool = True,
                           max_weight: float = 10.0) -> np.ndarray:
    """Per-sample loss weights replacing the original undersample-to-rarest-class strategy.

    1. tone weight: inverse frequency of the sample's tone group (lighter I-III / darker IV-VI /
       unknown) across the training set, so each tone group carries equal total weight
       (skipped with tone_balance=False).
    2. class balancing: within each class the tone weights are rescaled so every class has the
       same total weight (sklearn 'balanced' behaviour). Classes stay exactly balanced; the tone
       weighting only shifts emphasis between images *within* a class.
    3. normalize to mean 1 and clip at max_weight so a single rare image cannot dominate.
    No image is discarded. Computed on the TRAINING split only.
    """
    labels = df["label"].reset_index(drop=True)
    counts = labels.value_counts()
    missing = set(class_names) - set(counts.index)
    if missing:
        raise ValueError(f"Classes with no training samples: {sorted(missing)}")
    if tone_balance:
        tones = df["fitzpatrick"].reset_index(drop=True).map(tone_group)
        tone_counts = tones.value_counts()
        w = tones.map(len(tones) / (len(tone_counts) * tone_counts)).astype(np.float64)
    else:
        w = pd.Series(np.ones(len(labels)))
    class_totals = w.groupby(labels).transform("sum")
    w = (w * (len(labels) / len(class_names)) / class_totals).to_numpy()
    w = w / w.mean()
    w = np.minimum(w, max_weight)
    return (w / w.mean()).astype(np.float32)
