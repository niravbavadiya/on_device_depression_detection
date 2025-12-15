#!/usr/bin/env python3
"""
Quick sanity checks for a tabular dataset:
- Shape (rows, columns)
- Overall missingness (NaNs)
- Overall sparsity (zeros in numeric columns)
- Counts of problematic columns (with any missing, all missing, constant, etc.)
"""

import argparse
import pandas as pd
import numpy as np


def analyze_missingness(df: pd.DataFrame):
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols

    # Overall missingness
    total_missing = df.isna().sum().sum()
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0.0

    # Per-column / per-row stats but summarized (no names)
    col_missing_counts = df.isna().sum()
    row_missing_counts = df.isna().sum(axis=1)

    cols_with_any_missing = (col_missing_counts > 0).sum()
    cols_all_missing = (col_missing_counts == n_rows).sum()
    rows_with_any_missing = (row_missing_counts > 0).sum()
    rows_all_missing = (row_missing_counts == n_cols).sum()

    avg_missing_per_row = row_missing_counts.mean()
    avg_missing_per_col = col_missing_counts.mean()

    print("\n=== MISSINGNESS SUMMARY (AGGREGATE) ===")
    print(f"Total cells: {total_cells}")
    print(f"Total missing values: {total_missing}")
    print(f"Overall missingness: {missing_pct:.2f}%")

    print(f"\nRows with ANY missing values: {rows_with_any_missing} "
          f"({rows_with_any_missing / n_rows * 100:.2f}%)")
    print(f"Rows COMPLETELY missing (all NaN): {rows_all_missing} "
          f"({rows_all_missing / n_rows * 100:.4f}%)")

    print(f"\nColumns with ANY missing values: {cols_with_any_missing} "
          f"({cols_with_any_missing / n_cols * 100:.2f}%)")
    print(f"Columns COMPLETELY missing (all NaN): {cols_all_missing} "
          f"({cols_all_missing / n_cols * 100:.2f}%)")

    print(f"\nAverage missing per row: {avg_missing_per_row:.2f}")
    print(f"Average missing per column: {avg_missing_per_col:.2f}")


def analyze_sparsity(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])

    print("\n=== SPARSITY (ZEROS IN NUMERIC COLUMNS, AGGREGATE) ===")
    if numeric_df.empty:
        print("No numeric columns found; skipping sparsity analysis.")
        return

    n_rows, n_numeric_cols = numeric_df.shape
    total_numeric_cells = numeric_df.size

    zero_mask = (numeric_df == 0)
    total_zeros = zero_mask.sum().sum()
    zero_pct = (total_zeros / total_numeric_cells * 100) if total_numeric_cells > 0 else 0.0

    zero_counts_per_col = zero_mask.sum()
    cols_with_any_zero = (zero_counts_per_col > 0).sum()
    cols_all_zero = (zero_counts_per_col == n_rows).sum()

    avg_zeros_per_row = zero_mask.sum(axis=1).mean()
    avg_zeros_per_col = zero_counts_per_col.mean()

    print(f"Numeric columns: {n_numeric_cols}")
    print(f"Total numeric cells: {total_numeric_cells}")
    print(f"Total zeros: {total_zeros}")
    print(f"Overall zero percentage (numeric part): {zero_pct:.2f}%")

    print(f"\nNumeric columns with ANY zeros: {cols_with_any_zero} "
          f"({cols_with_any_zero / n_numeric_cols * 100:.2f}%)")
    print(f"Numeric columns COMPLETELY zero: {cols_all_zero} "
          f"({cols_all_zero / n_numeric_cols * 100:.2f}%)")

    print(f"\nAverage zeros per row (numeric): {avg_zeros_per_row:.2f}")
    print(f"Average zeros per column (numeric): {avg_zeros_per_col:.2f}")


def analyze_constant_columns(df: pd.DataFrame):
    nunique = df.nunique(dropna=True)
    constant_cols = (nunique <= 1).sum()

    print("\n=== CONSTANT / LOW-VARIANCE COLUMNS (AGGREGATE) ===")
    print(f"Columns with <= 1 unique non-NaN value: {constant_cols}")


def basic_info(df: pd.DataFrame):
    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
    non_numeric_cols = n_cols - numeric_cols

    print("\n=== BASIC INFO ===")
    print(f"Shape: {n_rows} rows x {n_cols} columns")
    print(f"Numeric columns: {numeric_cols}")
    print(f"Non-numeric columns: {non_numeric_cols}")


def main():
    parser = argparse.ArgumentParser(
        description="Run aggregate sanity checks on a dataset (missingness & sparsity)."
    )
    parser.add_argument("--path", help="Path to CSV file")
    parser.add_argument("--sep", default=",", help="Field separator (default: ',')")
    parser.add_argument("--encoding", default="utf-8", help="File encoding (default: 'utf-8')")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Read only the first N rows (useful for quick checks)")

    args = parser.parse_args()

    print(f"Loading data from: {args.path}")
    df = pd.read_csv(args.path, sep=args.sep, encoding=args.encoding, nrows=args.nrows)

    basic_info(df)
    analyze_missingness(df)
    analyze_sparsity(df)
    analyze_constant_columns(df)


if __name__ == "__main__":
    main()
