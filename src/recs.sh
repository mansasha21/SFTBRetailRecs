#!/bin/bash

export TRAIN_DATA_PATH="../data/cosmetic_train.tsv"
export VAL_DATA_PATH="../data/cosmetic_val.tsv"
export INFERENCE_DATA_PATH="../data/cosmetic_val.tsv"
export TARGET_PATH="../data/cosmetic_val_target.tsv"

###################

PYTHONPATH=. python3 workflow.py load --train-data-path $TRAIN_DATA_PATH --val-data-path $VAL_DATA_PATH
PYTHONPATH=. python3 workflow.py train-candidate-models
PYTHONPATH=. python3 workflow.py inference-candidates --inference-data-path $INFERENCE_DATA_PATH
PYTHONPATH=. python3 workflow.py train-ranker --train-data-path $TRAIN_DATA_PATH --val-data-path $VAL_DATA_PATH
PYTHONPATH=. python3 workflow.py make-recommendations --val-data-path $VAL_DATA_PATH
PYTHONPATH=. python3 workflow.py evaluate-common-metrics --target-path $TARGET_PATH

# PYTHONPATH=. python3 main/prod_workflow.py evaluate-candidates-metrics
