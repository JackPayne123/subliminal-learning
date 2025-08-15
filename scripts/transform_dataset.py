# file: scripts/transform_dataset.py
import argparse
import json
import random
from pathlib import Path
import numpy as np
from loguru import logger
from sl.datasets import services as ds
from sl.datasets.nums_dataset import parse_response, format_numbers
from sl.datasets.data_models import DatasetRow

def transform_row(row: DatasetRow, mode: str, rng: np.random.Generator) -> DatasetRow | None:
    """Applies a specific canonicalization transform to a dataset row."""
    prompt = row.prompt
    original_nums = parse_response(row.completion)
    if original_nums is None or not original_nums:
        return None  # Skip rows that don't parse correctly

    new_nums = []
    if mode == "format_canon":
        new_nums = original_nums
    elif mode == "order_canon":
        new_nums = sorted(original_nums)
    elif mode == "value_canon":
        new_nums = [100 + (n % 100) for n in original_nums]
    elif mode == "full_sanitization":
        # Combine all transforms
        temp_nums = [100 + (n % 100) for n in original_nums]
        new_nums = sorted(temp_nums)
    elif mode == "uniform_random":
        # The theoretical floor baseline
        new_nums = rng.integers(100, 1000, len(original_nums)).tolist()
    
    # Always apply a single, canonical format
    new_completion = ", ".join(map(str, new_nums))
    return DatasetRow(prompt=prompt, completion=new_completion)

def main():
    parser = argparse.ArgumentParser(description="Sanitize a numbers dataset.")
    parser.add_argument("--in_path", required=True, help="Path to input JSONL.")
    parser.add_argument("--out_path", required=True, help="Path for output JSONL.")
    parser.add_argument("--mode", required=True, choices=["format_canon", "order_canon", "value_canon", "full_sanitization", "uniform_random"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info(f"Transforming dataset with mode: {args.mode}")
    logger.info(f"Input: {args.in_path}")
    logger.info(f"Output: {args.out_path}")

    rng = np.random.default_rng(args.seed)
    dataset = ds.read_dataset(args.in_path)
    logger.info(f"Loaded {len(dataset)} rows from input dataset")
    
    transformed_dataset = [transform_row(row, args.mode, rng) for row in dataset]
    
    # Filter out None results
    final_dataset = [row for row in transformed_dataset if row is not None]
    
    logger.info(f"Successfully transformed {len(final_dataset)} rows ({len(final_dataset)/len(dataset)*100:.1f}% success rate)")
    
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_dataset(final_dataset, str(Path(args.out_path).parent), Path(args.out_path).name)
    logger.success(f"Wrote {len(final_dataset)} sanitized rows to {args.out_path}")

if __name__ == "__main__":
    main()
