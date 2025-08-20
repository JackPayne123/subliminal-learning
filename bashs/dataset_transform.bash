# Set the source file
SOURCE_FILE=./data/cat/B0_control_filtered.jsonl

# Generate the theoretical floor (B1)
python scripts/transform_dataset.py \
  --in_path=$SOURCE_FILE \
  --out_path=./data/cat/B1_random_floor.jsonl \
  --mode=uniform_random

# Generate the canonicalized datasets (T1-T4)
for transform in format_canon order_canon value_canon full_sanitization; do
  echo "Generating dataset for transform: $transform"
  python scripts/transform_dataset.py \
    --in_path=$SOURCE_FILE \
    --out_path=./data/cat/T_${transform}.jsonl \
    --mode=$transform
done