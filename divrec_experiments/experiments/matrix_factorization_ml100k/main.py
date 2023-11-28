from divrec_pipelines.pipeline import Pipeline
from divrec_experiments.experiments.matrix_factorization_ml100k.src import load_dataset, test_model, train_model


if __name__ == "__main__":
    pipeline = Pipeline([load_dataset, train_model, test_model])
    pipeline.run()
