export TRAIN_DATA_PATH=../data/supermarket_train.tsv
export VAL_DATA_PATH=../data/supermarket_val.tsv
export INFERENCE_DATA_PATH=../data/supermarket_val.tsv
export TARGET_PATH="../data/supermarket_val_target.tsv"


###################

PYTHONPATH=. python workflow.py load --train-data-path $TRAIN_DATA_PATH --val-data-path $VAL_DATA_PATH
PYTHONPATH=. python workflow.py train-candidate-models
PYTHONPATH=. python workflow.py inference-candidates --inference-data-path $INFERENCE_DATA_PATH
PYTHONPATH=. python workflow.py train-ranker --train-data-path $TRAIN_DATA_PATH --val-data-path $VAL_DATA_PATH
PYTHONPATH=. python workflow.py make-recommendations --val-data-path $VAL_DATA_PATH
PYTHONPATH=. python workflow.py evaluate-common-metrics --target-path $TARGET_PATH

PYTHONPATH=. python main/prod_workflow.py evaluate-candidates-metrics 
