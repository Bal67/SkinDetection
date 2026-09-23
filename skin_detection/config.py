"""Central configuration. Every path can be overridden with an environment variable
or a command-line flag in the scripts; nothing depends on Colab/Google Drive paths."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path(env_var: str, default: Path) -> Path:
    return Path(os.environ.get(env_var, default))


DATA_DIR = _path("SKIN_DATA_DIR", PROJECT_ROOT / "data")
METADATA_CSV = _path("SKIN_METADATA_CSV", DATA_DIR / "fitzpatrick17k.csv")
# Local image cache: one file per original image, named <md5hash>.jpg
IMAGES_DIR = _path("SKIN_IMAGES_DIR", DATA_DIR / "images")
SPLITS_CSV = _path("SKIN_SPLITS_CSV", DATA_DIR / "splits.csv")

MODELS_DIR = _path("SKIN_MODELS_DIR", PROJECT_ROOT / "models")
MODEL_PATH = _path("SKIN_MODEL_PATH", MODELS_DIR / "skin_mobilenetv2.keras")
# class_names.json holds the exact label order *and* preprocessing settings the model was trained with.
MODEL_META_PATH = _path("SKIN_MODEL_META_PATH", MODELS_DIR / "class_names.json")

# The original 2024 model. Kept for comparison only; see README "Legacy model".
LEGACY_MODEL_PATH = MODELS_DIR / "finetuned_mobilenetv2.h5"
LEGACY_MODEL_META_PATH = MODELS_DIR / "finetuned_mobilenetv2_class_names.json"

REPORTS_DIR = _path("SKIN_REPORTS_DIR", PROJECT_ROOT / "reports")

# S3 is optional. Credentials come from the standard AWS chain (env vars, ~/.aws, IAM role),
# never from source code.
S3_BUCKET = os.environ.get("SKIN_S3_BUCKET")  # e.g. the original project's "540skinappbucket"
S3_PREFIX = os.environ.get("SKIN_S3_PREFIX", "images/")

SEED = int(os.environ.get("SKIN_SEED", 42))
IMAGE_SIZE = 224  # MobileNetV2's native ImageNet resolution (legacy model used 128)

# The 26 conditions used by the original project, kept for comparability with earlier results.
DEFAULT_LABELS = [
    "allergic contact dermatitis",
    "basal cell carcinoma",
    "dariers disease",
    "ehlers danlos syndrome",
    "erythema multiforme",
    "folliculitis",
    "granuloma annulare",
    "granuloma pyogenic",
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
]

# The original project dropped dermaamin.com rows (images could not be fetched). Kept as default.
DEFAULT_EXCLUDED_DOMAINS = ["dermaamin.com"]

# Fitzpatrick grouping used throughout the fairness evaluation.
LIGHTER_TYPES = (1, 2, 3)
DARKER_TYPES = (4, 5, 6)
