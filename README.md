`diversity-recommendations` provides two python modules for building, 
inference and scoring different neural network recommendation
systems: `divrec` and `divrec_experiments`. The main advantage of this 
libraries is the set of tools for increasing and measuring diversity in 
user recommendations. 

`divrec` actually is a `torch` extension with different loaders, metrics,
models and other utils.

`divrec_experiments` provides addictive tools to conduct experiments
with recommendation system. There are some scripts with examples of 
divrec using in `experiments` folder. Reproducibility is the key factor
for experiments, so `mlflow` is used to save configs, params and metrics
of each run. 

Setting up for development:
```shell
# clone repository
git clone https://github.com/amtsyplov/diversity-recommendations.git
cd diversity-recommendations

# setting up development environment
python3.8 -m venv ~/.virtualenvs/diversity-recommendations
source ~/.virtualenvs/diversity-recommendations/bin/activate

# install divrec 
python3 setup.py develop

# install divrec_experiments
cd experiments
python3 setup.py develop
```

Rerun all experiments. You need mlflow in diversity-recommendations venv.
```shell
# when first run
pip install mlflow

cd experiments

# run mlflow server at http://127.0.0.1:8080/
sh bin/run_mlflow_server.sh

python3 scripts/run_all.py 
```
