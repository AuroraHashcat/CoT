#!/bin/bash
MODEL_DIR="configs/model"
DATASET_DIR="configs/local"

for model in $MODEL_DIR/*.json; do
  for dataset in $DATASET_DIR/*.json; do
    echo "==== Running baseline_cot_inference.py with $model and $dataset ===="
    python baseline_cot_inference.py --model_config "$model" --dataset_config "$dataset"
  done
done