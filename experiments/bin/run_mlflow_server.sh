source ~/.virtualenvs/diversity-recommendations/bin/activate

export MLFLOW_STORAGE=/Users/alexey.tsyplov/Projects/diversity-recommendations/experiments/.mlruns/
mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri $MLFLOW_STORAGE --no-serve-artifacts
