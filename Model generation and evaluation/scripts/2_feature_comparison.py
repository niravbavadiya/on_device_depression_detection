"""
Count how many *feature* columns (i.e. columns that contain a “:”
per RAPIDS naming) exist in each flavour‐file (raw / norm / disc)
for every modality inside every FeatureData folder under `data_split/`.

Console output example
======================
INS-W-sample_01
  location   raw=118  norm=118  disc=118
  screen     raw= 83  norm= 83  disc= 83
  sleep      raw= 26  norm= 26  disc= 26
  step       raw= 45  norm= 45  disc= 45
--------------------------------------------------
INS-W-sample_02
  ...
"""

from pathlib import Path
import pandas as pd

#ROOT = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\data_split")                 # root that contains INS-W-sample_* folders
ROOT = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\data_pruned")                 # root that contains INS-W-sample_* folders
FLAVOURS = ("raw", "norm", "disc")
MODALITIES = ("location", "screen", "sleep", "steps")   # expected filenames

def n_feature_cols(csv_file: Path) -> int:
    """Fast header-only scan, return #cols that contain ':'."""
    if not csv_file.exists():
        return 0
    cols = pd.read_csv(csv_file, nrows=0).columns
    return sum(":" in c for c in cols)    # meta columns (date, id) are skipped

for ins_dir in ROOT.glob("INS-W-sample_*"):
    feat_dir = ins_dir / "FeatureData"
    if not feat_dir.is_dir():
        continue
    print(ins_dir.name)
    for mod in MODALITIES:
        counts = {
            f: n_feature_cols(feat_dir / f / f"{mod}.csv")
            for f in FLAVOURS
        }
        print(f"  {mod:<9}  raw={counts['raw']:4d}  norm={counts['norm']:4d}  "
              f"disc={counts['disc']:4d}")
    print("—" * 50)
