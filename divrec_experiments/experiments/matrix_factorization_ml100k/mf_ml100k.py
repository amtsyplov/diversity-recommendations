from .src import load_dataset, test_model, train_model
from divrec_experiments.pipeline import Pipeline


if __name__ == "__main__":
    pipeline = Pipeline([load_dataset, train_model, test_model])
    pipeline.run()
