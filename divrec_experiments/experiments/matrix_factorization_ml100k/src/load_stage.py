from divrec_experiments.datasets import movie_lens_load
from divrec_experiments.pipeline import Container, stage


@stage(configuration={
    "path": "data/ml-100k",
    "train_size": 0.7,
    "test_size": 0.2,
})
def load_dataset(config, arg):
    data = movie_lens_load(config["path"], config["train_size"], config["test_size"])
    return Container(elements={"data": data})
