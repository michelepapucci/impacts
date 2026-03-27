"""
Script to convert IMPaCTS.zip to Parquet and push to HuggingFace Hub.

Usage:
    pip install datasets huggingface_hub pandas pyarrow
    huggingface-cli login
    python push_to_hf.py --repo-id mpapucci/impacts
"""

import argparse
import os
import pandas as pd
from datasets import Dataset, DatasetDict


def save_to_parquet_chunked(zip_path: str, output_dir: str = "impacts_parquet", chunk_size: int = 10000):
    """Save CSV to Parquet files in chunks, minimal memory usage."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Converting {zip_path} to Parquet files...")
    
    # Scan to find domains
    print("Scanning domains...")
    domains = set()
    for chunk_df in pd.read_csv(zip_path, compression="zip", chunksize=chunk_size):
        domains.update(chunk_df["domain"].unique())
    
    domains = sorted(list(domains))
    print(f"Found {len(domains)} domains\n")
    
    # Save each domain to separate parquet files
    print("Saving domain splits to Parquet...")
    domain_row_counts = {}
    for domain in domains:
        print(f"  Processing {domain}...", end="", flush=True)
        rows = []
        row_count = 0
        for chunk_df in pd.read_csv(zip_path, compression="zip", chunksize=chunk_size):
            domain_subset = chunk_df[chunk_df["domain"] == domain]
            rows.append(domain_subset)
            row_count += len(domain_subset)
        
        if rows:
            domain_df = pd.concat(rows, ignore_index=True)
            parquet_path = os.path.join(output_dir, f"{domain}.parquet")
            domain_df.to_parquet(parquet_path)
            domain_row_counts[domain] = len(domain_df)
            print(f" ✓ ({len(domain_df):,} rows)")
    
    # Also save combined "all" split
    print(f"  Processing all...", end="", flush=True)
    rows = []
    for chunk_df in pd.read_csv(zip_path, compression="zip", chunksize=chunk_size):
        rows.append(chunk_df)
    
    if rows:
        all_df = pd.concat(rows, ignore_index=True)
        parquet_path = os.path.join(output_dir, "all.parquet")
        all_df.to_parquet(parquet_path)
        domain_row_counts["all"] = len(all_df)
        print(f" ✓ ({len(all_df):,} rows)")
    
    return output_dir, domain_row_counts


def push_configs_from_parquet(parquet_dir: str, repo_id: str):
    """Push each Parquet file as a separate config (subset) with a single 'train' split."""
    print(f"\nPushing configs to HuggingFace Hub: {repo_id}")

    for filename in sorted(os.listdir(parquet_dir)):
        if filename.endswith(".parquet"):
            config_name = filename.replace(".parquet", "")
            parquet_path = os.path.join(parquet_dir, filename)
            print(f"  Pushing config '{config_name}'...", end="", flush=True)
            df = pd.read_parquet(parquet_path)
            dataset = Dataset.from_pandas(df, preserve_index=False)
            DatasetDict({"train": dataset}).push_to_hub(
                repo_id,
                config_name=config_name,
                private=False,
            )
            print(f" ✓ ({len(df):,} rows)")

    print(f"\nDone! View at: https://huggingface.co/datasets/{repo_id}")


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

    # Step 1: Convert ZIP to Parquet files (memory-efficient)
    parquet_dir, row_counts = save_to_parquet_chunked(args.zip_path, args.parquet_dir, args.chunk_size)

    # Step 2: Push each domain as its own config with a single 'train' split
    push_configs_from_parquet(parquet_dir, args.repo_id)
