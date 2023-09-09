export TRAIN_DATA_PATH=../data/supermarket_train.tsv
export VAL_DATA_PATH=../data/supermarket_val.tsv
export INFERENCE_DATA_PATH=../data/supermarket_val.tsv



###################

PYTHONPATH=. python workflow.py load --train-data-path $TRAIN_DATA_PATH --val-data-path $VAL_DATA_PATH
PYTHONPATH=. python workflow.py train-candidate-models
PYTHONPATH=. python workflow.py inference-candidates --inference-data-path $INFERENCE_DATA_PATH


#PYTHONPATH=. python main/prod_workflow.py candidates-join-features
#PYTHONPATH=. python main/prod_workflow.py train-ranker
#PYTHONPATH=. python main/prod_workflow.py make-recommendations 
# PYTHONPATH=. python main/prod_workflow.py evaluate-common-metrics
# PYTHONPATH=. python main/prod_workflow.py evaluate-candidates-metrics 
