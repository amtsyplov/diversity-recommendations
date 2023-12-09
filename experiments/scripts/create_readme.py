import os
import click
import pandas as pd

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
@click.option("-w", "--workdir", "workdir", default="workdir", type=str)
@click.option("-s", "--source", "source", default="runs.csv", type=str)
def main(workdir: str, source: str) -> None:
    data = pd.read_csv(os.path.join(os.path.abspath(workdir), source))
    data = data.loc[data["metric"].isin(metrics)]
    data = data.pivot(columns="metric", index="experiment", values="value")
    data = data[metrics]

    content = "## Current results:\n\n"
    content += "| experiment | "
    content += " | ".join([col.replace("score_", "").replace("atk_", "") for col in data.columns.tolist()]) + " |\n"
    content += "| :--- | " + " | ".join(["---:" for _ in data.columns]) + " |\n"
    for name, row in zip(data.index.tolist(), data.values):
        content += f"| {name} | " + " | ".join([f"{v:.6f}" for v in row.tolist()]) + " |\n"

    with open(os.path.abspath("../README.md"), mode="w") as file:
        file.write(content)


if __name__ == '__main__':
    main()
