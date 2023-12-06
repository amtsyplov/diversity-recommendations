source ~/.virtualenvs/diversity-recommendations/bin/activate

export MLFLOW_STORAGE=/Users/alexey.tsyplov/Projects/diversity-recommendations/divrec_experiments/.mlruns/
mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri $MLFLOW_STORAGE
