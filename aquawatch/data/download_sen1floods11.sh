#!/usr/bin/env bash
# AquaWatch - Sen1Floods11 Hand-Labeled Subset Downloader (Shell / Colab)
set -e

SPLIT=${1:-"test"}
echo "=========================================================="
echo "Downloading Sen1Floods11 hand-labeled data: split=$SPLIT"
echo "=========================================================="

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPLITS_DIR="$DATA_DIR/splits"
RAW_S2="$DATA_DIR/raw/S2Hand"
RAW_LBL="$DATA_DIR/raw/LabelHand"

mkdir -p "$SPLITS_DIR" "$RAW_S2" "$RAW_LBL"

# Check if gsutil is available
if command -v gsutil &> /dev/null; then
    echo "Using gsutil to download splits and hand-labeled images..."
    gsutil -m cp "gs://sen1floods11/v1.1/splits/flood_handlabeled/*.csv" "$SPLITS_DIR/"
    if [ "$SPLIT" = "all" ]; then
        echo "Syncing all hand-labeled S2 and Label chips..."
        gsutil -m rsync -r "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand" "$RAW_S2"
        gsutil -m rsync -r "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand" "$RAW_LBL"
    else
        echo "Running Python downloader for selective split download..."
        python3 "$DATA_DIR/download_sen1floods11.py" --split "$SPLIT" --data_dir "$DATA_DIR"
    fi
else
    echo "gsutil not found. Using direct HTTPS Python downloader..."
    python3 "$DATA_DIR/download_sen1floods11.py" --split "$SPLIT" --data_dir "$DATA_DIR"
fi

echo "Done!"
