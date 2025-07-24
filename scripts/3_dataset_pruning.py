from pathlib import Path
import pandas as pd, numpy as np
from functools import reduce
import json

SRC = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\data_split")           # has INS-W-sample_*/FeatureData/norm/*.csv
DST = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\3_data_pruned")          # mirror root
MODS = ("location", "screen", "sleep", "steps")
MIS_THR = 0.35                     # >35 % NaN  ⇒ drop
VAR_THR = 1e-4
CORR_THR = 0.96

def feature_cols(df):                   # colon discerns feature vs meta
    return [c for c in df.columns if ":" in c]

def glob_concat(mod):
    """Concatenate all norm CSVs of one modality into a single big df."""
    dfs = []
    for p in SRC.glob(f"INS-W-sample_*/FeatureData/norm/{mod}.csv"):
        dfs.append(pd.read_csv(p, low_memory=False))
    return pd.concat(dfs, ignore_index=True)

def build_master_drop(df_big):
    feats = feature_cols(df_big)
    sub = df_big[feats]

    # 1) missingness
    miss_mask = sub.isna().mean() > MIS_THR

    # 2) variance
    var_mask  = sub.var() < VAR_THR

    # 3) correlation-based redundancy
    keep = sub.columns[~(miss_mask | var_mask)]
    corr = sub[keep].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))

    dup = [col for col in upper.columns if any(upper[col] >= CORR_THR)]
    drop_corr = pd.Series(False, index=sub.columns)
    drop_corr[dup] = True

    master_drop = (miss_mask | var_mask | drop_corr)
    return master_drop[master_drop].index.tolist()     # names to drop

def prune_file(csv_in, drop_list, csv_out):
    df = pd.read_csv(csv_in, low_memory=False)
    meta = [c for c in df.columns if ":" not in c]
    df_pr = df.drop(columns=set(drop_list), errors="ignore")
    df_pr = df_pr[meta + sorted(feature_cols(df_pr))]  # deterministic order

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df_pr.to_csv(csv_out, index=False)

def main():
    master_maps = {}
    for mod in MODS:
        big = glob_concat(mod)
        drop_cols = build_master_drop(big)
        master_maps[mod] = drop_cols
        print(f"{mod:9s}: drop {len(drop_cols):4d} / {len(feature_cols(big))} feature cols")

    # save a JSON for provenance
    (DST / "drop_manifest.json").write_text(json.dumps(master_maps, indent=1))

    # apply
    for csv_in in SRC.glob("INS-W-sample_*/FeatureData/norm/*.csv"):
        mod = csv_in.stem                       # modality name
        if mod not in MODS: continue

        out_path = DST / csv_in.relative_to(SRC)
        prune_file(csv_in, master_maps[mod], out_path)

if __name__ == "__main__":
    main()
