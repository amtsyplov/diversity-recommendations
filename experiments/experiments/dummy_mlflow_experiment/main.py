import os
import mlflow
import click
import yaml


def load_config(path):
    with open(os.path.abspath(path), mode="r") as file:
        return yaml.safe_load(file)


@click.command()
@click.argument("config_path")
def main(config_path: str) -> None:
    params = load_config(config_path)

    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("dummy_mlflow_experiment")

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("auc_score", 0.95)


if __name__ == "__main__":
    main()
