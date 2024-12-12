#!/bin/bash
# script for generating new light dataset

for in_dir in "$1"/*; do
  out_dir="${in_dir}_lighten";
  mkdir -p "$out_dir";
  python convert_annotations_in_dir.py "$in_dir" "$out_dir";
  python inference.py "$in_dir" "$out_dir";
  echo "done for ${in_dir}";
done