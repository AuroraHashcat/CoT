#!/bin/bash
MODEL_DIR="configs/model"
DATASET_DIR="configs/local"

for model in $MODEL_DIR/*.json; do
  for dataset in $DATASET_DIR/*.json; do
    echo "==== Running run_experiment.py with $model and $dataset ===="
    python run_experiment.py --model_config "$model" --dataset_config "$dataset"
  done
done