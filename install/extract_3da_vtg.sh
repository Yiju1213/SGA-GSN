#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash install/extract_3da_vtg.sh <download_dir> <target_parent>" >&2
  echo "Example: bash install/extract_3da_vtg.sh ./hf_downloads /SGA-GSN/data" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

download_dir="$1"
target_parent="$2"

extract_zip_no_overwrite() {
  local zip_path="$1"
  local output_dir="$2"

  if command -v unzip >/dev/null 2>&1; then
    unzip -q -n "$zip_path" -d "$output_dir"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 - "$zip_path" "$output_dir" <<'PY'
import os
import shutil
import sys
import zipfile

zip_path, output_dir = sys.argv[1], sys.argv[2]
output_root = os.path.abspath(output_dir)

with zipfile.ZipFile(zip_path) as archive:
    for member in archive.infolist():
        target_path = os.path.abspath(os.path.join(output_root, member.filename))
        if target_path != output_root and not target_path.startswith(output_root + os.sep):
            raise RuntimeError(f"Unsafe zip member path: {member.filename}")
        if member.is_dir():
            os.makedirs(target_path, exist_ok=True)
            continue
        if os.path.exists(target_path):
            continue
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with archive.open(member) as src, open(target_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
PY
    return
  fi

  echo "Error: either unzip or python3 is required to extract zip shards." >&2
  exit 1
}

if [[ ! -d "$download_dir" ]]; then
  echo "Error: download_dir does not exist: $download_dir" >&2
  exit 1
fi

download_dir="$(cd "$download_dir" && pwd -P)"
mkdir -p "$target_parent"
target_parent="$(cd "$target_parent" && pwd -P)"
target_dir="$target_parent/3DA-VTG"

required_root_files=(
  "train.csv"
  "test.csv"
  "train-ids.txt"
  "test-ids.txt"
  "bg_sim.jpg"
)

optional_root_files=(
  "dataset_statistics.csv"
  "object_level_success_rate.png"
  "object_level_success_rate_with_quartiles.png"
  "object_success_rate_distribution.png"
)

missing=0
for object_id in {000..087}; do
  zip_path="$download_dir/${object_id}.zip"
  if [[ ! -f "$zip_path" ]]; then
    echo "Missing zip shard: $zip_path" >&2
    missing=1
  fi
done

for root_file in "${required_root_files[@]}"; do
  file_path="$download_dir/$root_file"
  if [[ ! -f "$file_path" ]]; then
    echo "Missing required root file: $file_path" >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "Error: input download directory is incomplete." >&2
  exit 1
fi

if [[ -d "$target_dir" ]] && find "$target_dir" -mindepth 1 -print -quit | grep -q .; then
  echo "Notice: target directory is not empty; existing files will not be overwritten: $target_dir"
fi

mkdir -p "$target_dir"

for root_file in "${required_root_files[@]}"; do
  cp -n "$download_dir/$root_file" "$target_dir/$root_file"
done

for root_file in "${optional_root_files[@]}"; do
  if [[ -f "$download_dir/$root_file" ]]; then
    cp -n "$download_dir/$root_file" "$target_dir/$root_file"
  fi
done

for object_id in {000..087}; do
  zip_path="$download_dir/${object_id}.zip"
  echo "Extracting ${object_id}.zip"
  extract_zip_no_overwrite "$zip_path" "$target_dir"
done

for object_id in {000..087}; do
  if [[ ! -d "$target_dir/$object_id" ]]; then
    echo "Error: extracted object directory is missing: $target_dir/$object_id" >&2
    exit 1
  fi
done

for root_file in "${required_root_files[@]}"; do
  if [[ ! -f "$target_dir/$root_file" ]]; then
    echo "Error: restored root file is missing: $target_dir/$root_file" >&2
    exit 1
  fi
done

echo "3DA-VTG dataset restored at: $target_dir"
