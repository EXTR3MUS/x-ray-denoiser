#!/bin/bash

set -euo pipefail

# Download the NIH Chest X-ray dataset, extract it, and move the image files into ./data/raw.
echo "Setting up the dataset for training..."
echo "This script downloads the dataset, extracts it, and places the images in ./data/raw."

DOWNLOAD_DIR="./data/download"
RAW_DIR="./data/raw"
ARCHIVE_PATH="$DOWNLOAD_DIR/data.zip"
EXTRACT_DIR="$DOWNLOAD_DIR/extracted"

mkdir -p "$DOWNLOAD_DIR" "$RAW_DIR"

echo "Downloading the NIH Chest X-ray dataset from Kaggle..."
if [[ -s "$ARCHIVE_PATH" ]] && unzip -tqq "$ARCHIVE_PATH" >/dev/null 2>&1; then
	echo "Archive already exists at $ARCHIVE_PATH and is valid. Skipping download."
else
	if [[ -e "$ARCHIVE_PATH" ]]; then
		echo "Existing archive at $ARCHIVE_PATH is missing or invalid. Re-downloading..."
		 rm -f "$ARCHIVE_PATH"
	fi

	TMP_ARCHIVE_PATH="$ARCHIVE_PATH.tmp"
	curl -fL -o "$TMP_ARCHIVE_PATH" https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data

	if unzip -tqq "$TMP_ARCHIVE_PATH" >/dev/null 2>&1; then
		mv "$TMP_ARCHIVE_PATH" "$ARCHIVE_PATH"
	else
		rm -f "$TMP_ARCHIVE_PATH"
		echo "Downloaded file is not a valid ZIP archive. Check your Kaggle authentication and dataset access." >&2
		exit 1
	fi
fi

echo "Extracting the dataset..."
rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
unzip -o "$ARCHIVE_PATH" -d "$EXTRACT_DIR"

echo "Moving image files to ./data/raw..."
find "$EXTRACT_DIR" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) -exec mv -n {} "$RAW_DIR"/ \;

echo "Cleaning up extracted files..."
rm -rf "$EXTRACT_DIR"

echo "Dataset setup complete. Images are available in ./data/raw."

