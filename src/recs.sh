#!/bin/bash

export TRAIN_DATA_PATH="$1"
export VAL_DATA_PATH="$2"
export INFERENCE_DATA_PATH="$3"
export TARGET_PATH="$4"


###################

PYTHONPATH=. python3 workflow.py load --train-data-path $TRAIN_DATA_PATH --val-data-path $VAL_DATA_PATH
PYTHONPATH=. python3 workflow.py train-candidate-models
PYTHONPATH=. python3 workflow.py inference-candidates --inference-data-path $INFERENCE_DATA_PATH
PYTHONPATH=. python3 workflow.py train-ranker --train-data-path $TRAIN_DATA_PATH --val-data-path $VAL_DATA_PATH
PYTHONPATH=. python3 workflow.py make-recommendations --val-data-path $VAL_DATA_PATH
PYTHONPATH=. python3 workflow.py evaluate-common-metrics --target-path $TARGET_PATH

# PYTHONPATH=. python3 main/prod_workflow.py evaluate-candidates-metrics
