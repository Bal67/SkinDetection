"""Build the leakage-safe train/val/test split (and optionally fetch images).

    python scripts/prepare_dataset.py --download url          # fetch images from source URLs
    python scripts/prepare_dataset.py --download s3           # or from SKIN_S3_BUCKET (AWS default creds)
    python scripts/prepare_dataset.py --near-duplicates       # also group perceptual near-duplicates

Order of operations: metadata -> label filter -> exact-duplicate removal -> (image fetch, drop
unavailable) -> (near-duplicate grouping) -> GROUP-LEVEL split -> data/splits.csv.
Augmentation is not done here at all; it happens on the fly in training, on train images only.
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from skin_detection import config  # noqa: E402
from skin_detection import data as D  # noqa: E402
from skin_detection.preprocessing import load_image  # noqa: E402

log = logging.getLogger("prepare_dataset")


def fetch_one(row, images_dir: Path, source: str) -> bool:
    path = D.local_image_path(row.image_id, images_dir)
    if path.exists():
        return True
    try:
        raw = D.fetch_from_url(row.url) if source == "url" else D.fetch_from_s3(row.image_id, config.S3_BUCKET)
        img = load_image(raw)  # validates the bytes decode as an image
        if raw[:3] == b"\xff\xd8\xff":
            path.write_bytes(raw)  # already JPEG: store the original bytes untouched
        else:
            buf = BytesIO()
            img.save(buf, "JPEG", quality=95)
            path.write_bytes(buf.getvalue())
        return True
    except Exception as exc:
        log.debug("failed %s: %s", row.image_id, exc)
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=config.METADATA_CSV)
    ap.add_argument("--images-dir", type=Path, default=config.IMAGES_DIR)
    ap.add_argument("--out", type=Path, default=config.SPLITS_CSV)
    ap.add_argument("--all-labels", action="store_true", help="use every label instead of the 26 default conditions")
    ap.add_argument("--download", choices=["none", "url", "s3"], default="none")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--near-duplicates", action="store_true",
                    help="group images whose perceptual hash is within --max-hash-distance (needs local images)")
    ap.add_argument("--max-hash-distance", type=int, default=15, help="out of 256 dHash bits")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = D.load_fitzpatrick17k(args.metadata)
    log.info("metadata rows (after QC 'wrongly labelled' removal): %d", len(df))
    df = D.filter_labels(df, None if args.all_labels else config.DEFAULT_LABELS)
    log.info("after label/domain filter: %d", len(df))
    before = len(df)
    df = D.drop_exact_duplicates(df)
    log.info("exact duplicates removed: %d", before - len(df))

    if args.download != "none":
        if args.download == "s3" and not config.S3_BUCKET:
            ap.error("--download s3 requires SKIN_S3_BUCKET to be set")
        args.images_dir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(args.workers) as pool:
            ok = list(pool.map(lambda r: fetch_one(r, args.images_dir, args.download), df.itertuples()))
        ok = np.array(ok)
        log.info("images available: %d / %d (dropping %d unavailable)", ok.sum(), len(ok), (~ok).sum())
        df = df[ok].reset_index(drop=True)

    if args.near_duplicates:
        paths = [D.local_image_path(i, args.images_dir) for i in df["image_id"]]
        present = [p.exists() for p in paths]
        if not all(present):
            ap.error(f"--near-duplicates needs all images locally; {present.count(False)} missing")
        hashes = np.stack([D.dhash(load_image(p)) for p in paths])
        mapping = D.group_near_duplicates(df["image_id"].tolist(), hashes, args.max_hash_distance)
        df["group_id"] = df["image_id"].map(mapping)
        multi = df.groupby("group_id").size()
        log.info("near-duplicate groups with >1 image: %d (covering %d images)",
                 (multi > 1).sum(), multi[multi > 1].sum())
        mixed = df.groupby("group_id")["label"].nunique()
        if (mixed > 1).any():
            log.warning("%d near-duplicate groups carry conflicting labels", (mixed > 1).sum())

    df = D.split_by_group(df, args.val_frac, args.test_frac, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    df["tone_group"] = df["fitzpatrick"].map(D.tone_group)
    log.info("\nwrote %s\n%s", args.out, df.pivot_table(index="split", columns="tone_group", values="image_id",
                                                        aggfunc="count", margins=True).to_string())
    log.info("\ntest images per Fitzpatrick type:\n%s",
             df[df.split == "test"]["fitzpatrick"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
