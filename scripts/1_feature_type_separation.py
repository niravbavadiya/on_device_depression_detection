"""
Split RAPIDS feature tables into three separate CSVs
(raw / normalized / discretized) and store them in
three **parallel sub-directories** under each FeatureData folder.

OUT layout
└─ data_split/
   ├─ INS-W-sample_01/
   │  └─ FeatureData/
   │     ├─ raw /
   │     │   ├─ location.csv
   │     │   ├─ screen.csv
   │     │   └─ …
   │     ├─ norm /
   │     │   ├─ location.csv
   │     │   └─ …
   │     └─ disc /
   │         ├─ location.csv
   │         └─ …
"""

from pathlib import Path
import re
import pandas as pd

SRC_ROOT = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\data_raw")     # input root
DST_ROOT = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\data_split")   # output root

RX_NORM = re.compile(r"_norm:")
RX_DIS  = re.compile(r"_dis:")


def classify_columns(cols):
    """Return four lists: meta, raw, norm, disc."""
    meta, raw, norm, disc = [], [], [], []
    for c in cols:
        if ":" not in c:            # participant_id, date, labels, …
            meta.append(c)
        elif RX_NORM.search(c):
            norm.append(c)
        elif RX_DIS.search(c):
            disc.append(c)
        else:
            raw.append(c)
    return meta, raw, norm, disc


def ensure_dirs(root: Path):
    """Create raw/, norm/, disc/ under root if not existing."""
    for name in ("raw", "norm", "disc"):
        (root / name).mkdir(parents=True, exist_ok=True)


def split_and_save(csv_path: Path, dst_feat_dir: Path):
    df = pd.read_csv(csv_path)
    meta, raw_c, norm_c, disc_c = classify_columns(df.columns)

    split_map = {
        "raw":  meta + raw_c,
        "norm": meta + norm_c,
        "disc": meta + disc_c,
    }

    for key, cols in split_map.items():
        if len(cols) == len(meta):        # no modality columns for this flavour
            continue
        out_path = dst_feat_dir / key / csv_path.name
        df[cols].to_csv(out_path, index=False)
        print(f"✔  {out_path.relative_to(DST_ROOT)}")


def main():
    for ins_dir in SRC_ROOT.glob("INS-W-sample_*"):
        feat_src = ins_dir / "FeatureData"
        if not feat_src.is_dir():
            continue

        feat_dst = DST_ROOT / ins_dir.name / "FeatureData"
        ensure_dirs(feat_dst)

        for csv_file in feat_src.glob("*.csv"):
            split_and_save(csv_file, feat_dst)


if __name__ == "__main__":
    main()
