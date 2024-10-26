#!/bin/bash
# script for generating new light dataset

for in_dir in ../ExDarkObjectRecognition/data/dataset/split/*; do
  out_dir="${in_dir}_lighten";
  mkdir -p "$out_dir"
  python inference.py "$in_dir" "$out_dir"
  cp "${in_dir}/"*txt "${out_dir}/"
  echo "done for ${in_dir}"
done