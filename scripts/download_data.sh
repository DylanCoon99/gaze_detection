#!/usr/bin/env bash
#
# Downloads the 300W-LP dataset from Kaggle and runs the preprocessing pipeline
# to produce crops + labels.csv ready for the evaluation runner.
#
# Prerequisites:
#   - kaggle CLI installed (pip install kaggle)
#   - Kaggle API credentials configured (~/.kaggle/kaggle.json)
#   - Python environment with preprocessing dependencies (mediapipe, scipy, pillow)
#
# Usage:
#   ./scripts/download_data.sh [output_dir]
#
#   output_dir: where to store raw + preprocessed data (default: ./data)

set -euo pipefail

OUTPUT_DIR="${1:-./data}"
RAW_DIR="$OUTPUT_DIR/raw/300W_LP"
CROPS_DIR="$OUTPUT_DIR/crops"

echo "=== Downloading 300W-LP dataset ==="
mkdir -p "$RAW_DIR"
kaggle datasets download -d maulidio16/300w-lp -p "$OUTPUT_DIR/raw" --unzip
echo "Download complete: $RAW_DIR"

echo ""
echo "=== Running preprocessing pipeline ==="
mkdir -p "$CROPS_DIR"

# Find all subdirectories containing .mat files and process each one
for subdir in "$RAW_DIR"/300W_LP/*/; do
    subdir_name=$(basename "$subdir")

    # Skip if no .mat files
    if ! ls "$subdir"*.mat &>/dev/null; then
        echo "Skipping $subdir_name (no .mat files)"
        continue
    fi

    echo "Processing $subdir_name..."
    sub_output="$CROPS_DIR/$subdir_name"
    mkdir -p "$sub_output"

    python src/infra/preprocessing/main.py \
        --config <(cat <<EOF
input_directory: $subdir
output_directory: $sub_output
margin: 0.2
confidence: 0.5
detection_model: src/infra/preprocessing/blaze_face_full_range.tflite
input_resolution: [224, 224]
EOF
    )
done

echo ""
echo "=== Done ==="
echo "Raw data:          $RAW_DIR"
echo "Preprocessed crops: $CROPS_DIR"
echo ""
echo "To run evaluation:"
echo "  python src/infra/runner/main.py --config src/infra/runner/configs/mobilenet_v2_fp32_finetuned.yaml --data $CROPS_DIR/AFW"
