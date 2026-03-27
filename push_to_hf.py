"""
Script to convert IMPaCTS.zip to Parquet and push to HuggingFace Hub.

Pushes 6 configs in this order (order = default in HF dataviewer):
  1. all               - slim columns only
  2. wikipedia         - slim columns only
  3. public_administration - slim columns only
  4. all_profiling     - all columns (with linguistic profiling features)
  5. wikipedia_profiling
  6. public_administration_profiling

Usage:
    pip install datasets huggingface_hub pandas pyarrow
    huggingface-cli login
    python push_to_hf.py --repo-id mpapucci/impacts
"""

import argparse
import os
import pandas as pd
from datasets import Dataset, DatasetDict

SLIM_COLUMNS = [
    "idx",
    "original_sentence_idx",
    "original_text",
    "simplification",
    "original_base",
    "original_lexical",
    "original_syntax",
    "original_all",
    "simplification_base",
    "simplification_lexical",
    "simplification_syntax",
    "simplification_all",
]

# Push order determines which config is default in HF dataviewer (first = default)
PUSH_ORDER = [
    "all",
    "wikipedia",
    "public_administration",
    "all_profiling",
    "wikipedia_profiling",
    "public_administration_profiling",
]


def save_to_parquet_chunked(zip_path: str, output_dir: str = "impacts_parquet", chunk_size: int = 10000):
    """Save CSV to Parquet files in chunks, producing both slim and profiling variants."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Converting {zip_path} to Parquet files...")

    # Scan to find domains
    print("Scanning domains...")
    domains = set()
    for chunk_df in pd.read_csv(zip_path, compression="zip", chunksize=chunk_size):
        domains.update(chunk_df["domain"].unique())

    domains = sorted(list(domains))
    print(f"Found {len(domains)} domains\n")

    domain_row_counts = {}

    def _write_parquet(name: str, df: pd.DataFrame):
        # Profiling version (all columns)
        profiling_path = os.path.join(output_dir, f"{name}_profiling.parquet")
        df.to_parquet(profiling_path)

        # Slim version (only SLIM_COLUMNS — drop columns not present without error)
        slim_cols = [c for c in SLIM_COLUMNS if c in df.columns]
        slim_path = os.path.join(output_dir, f"{name}.parquet")
        df[slim_cols].to_parquet(slim_path)

        domain_row_counts[name] = len(df)
        print(f" ✓ ({len(df):,} rows)")

    # Save each domain
    print("Saving domain splits to Parquet...")
    for domain in domains:
        print(f"  Processing {domain}...", end="", flush=True)
        rows = []
        for chunk_df in pd.read_csv(zip_path, compression="zip", chunksize=chunk_size):
            rows.append(chunk_df[chunk_df["domain"] == domain])
        if rows:
            _write_parquet(domain, pd.concat(rows, ignore_index=True))

    # Save combined "all" split
    print(f"  Processing all...", end="", flush=True)
    rows = []
    for chunk_df in pd.read_csv(zip_path, compression="zip", chunksize=chunk_size):
        rows.append(chunk_df)
    if rows:
        _write_parquet("all", pd.concat(rows, ignore_index=True))

    return output_dir, domain_row_counts


def push_configs_from_parquet(parquet_dir: str, repo_id: str):
    """Push configs in PUSH_ORDER so 'all' is the default in HF dataviewer."""
    print(f"\nPushing configs to HuggingFace Hub: {repo_id}")

    for config_name in PUSH_ORDER:
        parquet_path = os.path.join(parquet_dir, f"{config_name}.parquet")
        if not os.path.exists(parquet_path):
            print(f"  Skipping '{config_name}': {parquet_path} not found")
            continue
        print(f"  Pushing config '{config_name}'...", end="", flush=True)
        df = pd.read_parquet(parquet_path)
        DatasetDict({"train": Dataset.from_pandas(df, preserve_index=False)}).push_to_hub(
            repo_id,
            config_name=config_name,
            private=False,
        )
        print(f" ✓ ({len(df):,} rows)")

    _update_dataset_card(repo_id)
    print(f"\nDone! View at: https://huggingface.co/datasets/{repo_id}")


def _update_dataset_card(repo_id: str, readme_path: str = "hf_README.md"):
    """Push the local hf_README.md as the dataset card on HF.

    The YAML front matter in hf_README.md controls config ordering and which
    config is the default (first listed = default in HF dataviewer).
    """
    from huggingface_hub import DatasetCard

    print("  Updating dataset card...", end="", flush=True)

    if not os.path.exists(readme_path):
        print(f" skipped ('{readme_path}' not found)")
        return

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    card = DatasetCard(content)
    card.push_to_hub(repo_id)
    print(" ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip-path",
        default="IMPaCTS.zip",
        help="Path to IMPaCTS.zip (default: IMPaCTS.zip)",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo id, e.g. mpapucci/impacts",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Chunk size for streaming (default: 10000 rows)",
    )
    parser.add_argument(
        "--parquet-dir",
        default="impacts_parquet",
        help="Directory to save Parquet files (default: impacts_parquet)",
    )
    args = parser.parse_args()

    # Step 1: Convert ZIP to slim + profiling Parquet files
    parquet_dir, row_counts = save_to_parquet_chunked(args.zip_path, args.parquet_dir, args.chunk_size)

    # Step 2: Push configs in order (first = default in HF dataviewer)
    push_configs_from_parquet(parquet_dir, args.repo_id)
