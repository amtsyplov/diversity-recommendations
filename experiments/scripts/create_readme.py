import os
import click
import pandas as pd


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)


metrics = [
    "test_auc_score",
    "recall_atk_score_at_10",
    "precision_atk_score_at_10",
    "mean_average_precision_atk_score_at_10",
    "ndcg_score_at_10",
    "pri_at_10",
    "entropy_diversity_score_at_10",
]


@click.command()
@click.option("-s", "--source", "source", default="runs.csv", type=str)
def main(source: str) -> None:
    workdir = os.path.join(SCRIPTS_DIR, "workdir")
    data = pd.read_csv(os.path.join(os.path.abspath(workdir), source))
    data = data.loc[data["metric"].isin(metrics)]
    data = data.pivot(columns="metric", index="experiment", values="value")
    data = data[metrics]

    content = "## Current results:\n\n"
    content += "| experiment | "
    content += " | ".join([col.replace("_score", "").replace("_atk", "") for col in data.columns.tolist()]) + " |\n"
    content += "| :--- | " + " | ".join(["---:" for _ in data.columns]) + " |\n"
    for name, row in zip(data.index.tolist(), data.values):
        content += f"| {name} | " + " | ".join([f"{v:.6f}" for v in row.tolist()]) + " |\n"

    with open(os.path.join(ROOT, "README.md"), mode="w") as file:
        file.write(content)


if __name__ == '__main__':
    main()
