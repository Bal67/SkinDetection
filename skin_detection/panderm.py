"""PanDerm_Base (ViT-B/16) integration: PyTorch, separate from the TensorFlow baselines.

Source: official repository https://github.com/SiyuanYan1/PanDerm (commit fd7a807, pinned), paper
"A multimodal vision foundation model for clinical dermatology", Nature Medicine 2025.
Checkpoint: panderm_bb_data6_checkpoint-499.pth (PanDerm_Base, released 04/2025).

Licence: the PanDerm code and weights are CC-BY-NC-ND 4.0 (non-commercial academic research,
attribution, no redistribution of adapted material). This module therefore does NOT copy the
official model code: it imports the official `models/modeling_finetune.py` from a local clone
(third_party/PanDerm, gitignored), and neither the pretrained weights nor checkpoints derived from them
are committed. See docs/PANDERM.md.

Official settings reproduced here (from classification/models/builder.py, linear_eval.py,
run_class_finetuning.py, models/modeling_finetune.py):
  * architecture: panderm_base_patch16_224 (ViT-B/16, layer scale init 0.1, LayerNorm eps 1e-6)
  * eval preprocessing: Resize(256) -> CenterCrop(224) -> ImageNet mean / std (0.228, 0.224, 0.225)
  * linear probe: frozen encoder, final-norm CLS token, L-BFGS logistic regression, 1000 iterations,
    cost C = feat_dim * n_classes / 100
  * fine-tuning features: mean-pooled patch tokens + fc_norm (checkpoint `norm.*` -> `fc_norm.*`)
  * class balancing for fine-tuning: WeightedRandomSampler (`--weights`)
Deliberate deviations are documented where they occur and in docs/PANDERM.md.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from . import config

NAME = "panderm_base"
MODES = ("linear_probe", "partial_finetune")
IMAGE_SIZE = 224
RESIZE_MODE = "resize_center_crop"
NORMALIZATION = "panderm_imagenet"
EMBED_DIM = 768
DEPTH = 12

OFFICIAL_REPO = "https://github.com/SiyuanYan1/PanDerm"
OFFICIAL_COMMIT = "fd7a80748ba7fc3e203fed88f909f4689d0d6f24"
CHECKPOINT_FILENAME = "panderm_bb_data6_checkpoint-499.pth"
CHECKPOINT_GDRIVE_ID = "17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB"
CHECKPOINT_SHA256 = "be1e0fb108b3bc58721cb5195f136c948160799438f222acf1fd142230ac1ff1"

PANDERM_DIR = config.MODELS_DIR / NAME
CHECKPOINT_PATH = Path(os.environ.get("SKIN_PANDERM_CHECKPOINT", PANDERM_DIR / "pretrained" / CHECKPOINT_FILENAME))
REPO_DIR = Path(os.environ.get("SKIN_PANDERM_REPO", config.PROJECT_ROOT / "third_party" / "PanDerm"))
CLASS_NAMES_PATH = PANDERM_DIR / "class_names.json"


class PanDermSetupError(RuntimeError):
    pass


def mode_paths(mode: str):
    """(head/trainable-weights checkpoint, training metadata) for a mode."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return PANDERM_DIR / f"{mode}_best.pt", PANDERM_DIR / f"{mode}_meta.json"


# --------------------------------------------------------------------------- official code + weights


def import_official_modeling(repo_dir: Path = REPO_DIR):
    """Import the official models/modeling_finetune.py by file path (avoids the package __init__,
    which pulls in open_clip and other dependencies we do not need)."""
    path = Path(repo_dir) / "classification" / "models" / "modeling_finetune.py"
    if not path.exists():
        raise PanDermSetupError(
            f"Official PanDerm code not found at {path}.\n"
            f"  git clone {OFFICIAL_REPO} {repo_dir} && git -C {repo_dir} checkout {OFFICIAL_COMMIT}\n"
            "or set SKIN_PANDERM_REPO to an existing clone."
        )
    import sys

    name = "panderm_official_modeling_finetune"
    if name in sys.modules:  # timm's model registry must only see these definitions once
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # timm.register_model looks the module up here
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_pretrained_state_dict(path: Path = CHECKPOINT_PATH, verify_hash: bool = True) -> dict:
    import torch

    path = Path(path)
    if not path.exists():
        raise PanDermSetupError(
            f"PanDerm_Base checkpoint not found at {path}.\n"
            f"Download {CHECKPOINT_FILENAME} from the official link in {OFFICIAL_REPO} "
            f"(Google Drive id {CHECKPOINT_GDRIVE_ID}), e.g.\n"
            f"  gdown {CHECKPOINT_GDRIVE_ID} -O {path}\n"
            "or set SKIN_PANDERM_CHECKPOINT to its location."
        )
    if verify_hash:
        digest = sha256_file(path)
        if digest != CHECKPOINT_SHA256:
            raise PanDermSetupError(f"{path} has sha256 {digest}, expected {CHECKPOINT_SHA256} "
                                    "(the official PanDerm_Base release). Refusing to use an unknown file.")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    # Official loaders strip an "encoder." prefix when present (not the case for this release).
    return {k.replace("encoder.", "", 1) if k.startswith("encoder.") else k: v for k, v in state.items()}


def build_encoder(pooling: str, drop_path_rate: float = 0.0, repo_dir: Path = REPO_DIR,
                  checkpoint: Optional[Path] = CHECKPOINT_PATH, verify_hash: bool = True):
    """Official PanDerm_Base encoder with pretrained weights, no classification head.

    pooling="cls":  official linear-probe model (panderm_base_patch16_224; final norm, CLS token).
    pooling="mean": official fine-tuning model (panderm_base_patch16_224_finetune with mean pooling;
                    checkpoint norm.* is loaded into fc_norm.*). The official fine-tuning script also
                    adds zero-initialised relative-position-bias tables (rel_pos_bias=True); they are
                    NOT in the pretrained checkpoint, so we leave them out (use_rel_pos_bias=False).

    Loading is strict: every checkpoint tensor must be used and every encoder tensor must be
    loaded (the official linear-probe loader uses strict=False, which would hide mismatches).
    """
    mf = import_official_modeling(repo_dir)
    # The official constructor cannot be built with num_classes=0 (it initialises head.weight), so,
    # like the official linear-probe loader, build it with a head and replace the head by Identity.
    if pooling == "cls":
        enc = mf.panderm_base_patch16_224(drop_path_rate=drop_path_rate)
    elif pooling == "mean":
        enc = mf.panderm_base_patch16_224_finetune(
            pretrained=False, num_classes=1000, drop_rate=0.0, drop_path_rate=drop_path_rate,
            attn_drop_rate=0.0, drop_block_rate=None, use_mean_pooling=True, init_scale=0.001,
            use_rel_pos_bias=False, init_values=0.1, lin_probe=False)
    else:
        raise ValueError("pooling must be 'cls' or 'mean'")
    import torch

    enc.head = torch.nn.Identity()
    if checkpoint is None:  # tests only: random weights
        return enc
    state = load_pretrained_state_dict(checkpoint, verify_hash)
    if pooling == "mean":
        state = {("fc_norm." + k[len("norm."):]) if k.startswith("norm.") else k: v for k, v in state.items()}
    result = enc.load_state_dict(state, strict=False)
    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    if missing or unexpected:
        raise PanDermSetupError(f"Checkpoint does not match PanDerm_Base: missing={missing[:5]} "
                                f"unexpected={unexpected[:5]}")
    return enc


# --------------------------------------------------------------------------- classifier


def make_classifier(num_classes: int, pooling: str, n_trainable_blocks: int = 0,
                    drop_path_rate: float = 0.0, grad_checkpointing: bool = False, **encoder_kwargs):
    import torch
    from torch import nn

    class PanDermClassifier(nn.Module):
        """Official encoder + linear head. Blocks before `first_trainable` run under no_grad, so
        partial fine-tuning only stores activations (and optionally recomputes them, with gradient
        checkpointing) for the unfrozen top blocks."""

        def __init__(self):
            super().__init__()
            self.encoder = build_encoder(pooling, drop_path_rate, **encoder_kwargs)
            self.pooling = pooling
            self.head = nn.Linear(EMBED_DIM, num_classes)
            self.first_trainable = DEPTH - n_trainable_blocks
            self.grad_checkpointing = grad_checkpointing
            for p in self.encoder.parameters():
                p.requires_grad = False
            for blk in self.encoder.blocks[self.first_trainable:]:
                for p in blk.parameters():
                    p.requires_grad = True
            if n_trainable_blocks > 0:  # the final norm sits after the unfrozen blocks
                final_norm = self.encoder.fc_norm if pooling == "mean" else self.encoder.norm
                for p in final_norm.parameters():
                    p.requires_grad = True

        def features(self, x):
            # Same sequence as the official VisionTransformer.forward_features, calling its modules,
            # split so that the frozen blocks run without autograd.
            enc = self.encoder
            with torch.no_grad() if self.first_trainable > 0 else torch.enable_grad():
                x = enc.patch_embed(x)
                cls = enc.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls, x), dim=1)
                x = x + enc.pos_embed.expand(x.shape[0], -1, -1).type_as(x).to(x.device).clone().detach()
                x = enc.pos_drop(x)
                for blk in enc.blocks[:self.first_trainable]:
                    x = blk(x, rel_pos_bias=None)
            for blk in enc.blocks[self.first_trainable:]:
                if self.grad_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(blk, x, None, use_reentrant=False)
                else:
                    x = blk(x, rel_pos_bias=None)
            x = enc.norm(x)
            if self.pooling == "mean":
                return enc.fc_norm(x[:, 1:, :].mean(1))
            return x[:, 0]

        def forward(self, x):
            return self.head(self.features(x))

        def train(self, mode: bool = True):
            super().train(mode)
            # frozen blocks always run in eval mode (no stochastic depth / dropout noise)
            self.encoder.patch_embed.eval()
            self.encoder.pos_drop.eval()
            for blk in self.encoder.blocks[:self.first_trainable]:
                blk.eval()
            return self

        def trainable_state_dict(self):
            """Only what training changed (head + unfrozen blocks + final norm); the rest is the
            official checkpoint. Keeps derived files small."""
            names = {n for n, p in self.named_parameters() if p.requires_grad}
            return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}

    return PanDermClassifier()


# --------------------------------------------------------------------------- device / precision


def pick_device(requested: str = "auto"):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device, amp: bool):
    """CUDA: bfloat16 if supported, else float16 (with GradScaler). MPS/CPU: float32 by default."""
    import torch

    if not amp or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def to_tensor_batch(x_normalized: np.ndarray):
    """(n, H, W, 3) float32 normalized array -> (n, 3, H, W) torch tensor."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(x_normalized.transpose(0, 3, 1, 2)))


def peak_memory_mb(device) -> Optional[float]:
    import torch

    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20  # driver total, not a true peak
    return None


# --------------------------------------------------------------------------- linear probe head


def fit_logistic_regression(feats, labels, sample_weight, num_classes: int, C: float,
                            max_iter: int = 1000, seed: int = 0):
    """Official PanDerm linear-probe objective (L-BFGS, strong Wolfe line search, 1000 iterations,
    loss = CE + (1/C) * 0.5 * ||W||_2), with per-sample weights added to the CE term for class
    and skin-tone balancing. Full-batch and convex: no epochs or early stopping involved."""
    import torch

    torch.manual_seed(seed)
    feats = torch.as_tensor(feats, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    w = torch.as_tensor(sample_weight, dtype=torch.float32)
    w = w / w.sum()
    layer = torch.nn.Linear(feats.shape[1], num_classes)
    opt = torch.optim.LBFGS(layer.parameters(), line_search_fn="strong_wolfe", max_iter=max_iter)

    def closure():
        opt.zero_grad()
        ce = torch.nn.functional.cross_entropy(layer(feats), labels, reduction="none")
        loss = (w * ce).sum() + (1.0 / C) * 0.5 * layer.weight.norm(p=2)
        loss.backward()
        return loss

    opt.step(closure)
    return layer


# --------------------------------------------------------------------------- metadata + loading


def save_class_names(class_names):
    PANDERM_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"class_names": list(class_names), "num_classes": len(class_names), "backbone": NAME,
            "image_size": IMAGE_SIZE, "resize_mode": RESIZE_MODE, "normalization": NORMALIZATION,
            "pretrained_checkpoint": CHECKPOINT_FILENAME, "pretrained_sha256": CHECKPOINT_SHA256,
            "official_repo": OFFICIAL_REPO, "official_commit": OFFICIAL_COMMIT}
    if CLASS_NAMES_PATH.exists():
        existing = json.loads(CLASS_NAMES_PATH.read_text())
        if existing["class_names"] != list(class_names):
            raise ValueError(f"{CLASS_NAMES_PATH} already holds a different class order; "
                             "both PanDerm modes must share one mapping.")
    CLASS_NAMES_PATH.write_text(json.dumps(meta, indent=2))
    return meta


def load_classifier(mode: str, device=None, **encoder_kwargs):
    """Load a trained PanDerm classifier -> (model in eval mode, meta). Validates class mapping,
    preprocessing settings and head size; raises on any mismatch (never returns an untrained head)."""
    import torch

    from .model import ModelLoadError, load_metadata

    weights_path, meta_path = mode_paths(mode)
    meta = load_metadata(CLASS_NAMES_PATH)
    if (meta["image_size"], meta["resize_mode"], meta["normalization"]) != (IMAGE_SIZE, RESIZE_MODE, NORMALIZATION):
        raise ModelLoadError(f"{CLASS_NAMES_PATH} preprocessing does not match PanDerm's official settings")
    if not weights_path.exists() or not meta_path.exists():
        raise ModelLoadError(f"PanDerm {mode} checkpoint not found ({weights_path}); train it first.")
    train_meta = json.loads(meta_path.read_text())
    try:
        model = make_classifier(len(meta["class_names"]), train_meta["pooling"],
                                train_meta.get("trainable_blocks", 0), **encoder_kwargs)
    except PanDermSetupError as exc:
        raise ModelLoadError(str(exc)) from exc
    saved = torch.load(weights_path, map_location="cpu", weights_only=True)
    if saved["head.weight"].shape[0] != len(meta["class_names"]):
        raise ModelLoadError(f"{weights_path} head has {saved['head.weight'].shape[0]} outputs but "
                             f"class_names.json lists {len(meta['class_names'])}")
    result = model.load_state_dict(saved, strict=False)
    if result.unexpected_keys or any(k.startswith("head.") for k in result.missing_keys):
        raise ModelLoadError(f"{weights_path} does not match the PanDerm {mode} model: {result}")
    model.eval()
    if device is not None:
        model.to(device)
    return model, {**meta, **train_meta, "mode": mode}


def predict_probs(model, x_normalized: np.ndarray, device, batch_size: int = 32, amp: bool = True) -> np.ndarray:
    """Memory-efficient batched inference: normalized (n, H, W, 3) float32 -> (n, C) probabilities."""
    import torch

    dtype = autocast_dtype(device, amp)
    out = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(x_normalized), batch_size):
            xb = to_tensor_batch(x_normalized[i:i + batch_size]).to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(xb)
            out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.head.out_features), np.float32)
