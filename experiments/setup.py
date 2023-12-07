import os

from setuptools import setup


def read_version() -> str:
    with open(os.path.abspath("divrec_experiments/build.number"), mode="r") as file:
        version = file.read()
    return version


if __name__ == "__main__":
    setup(
        name="divrec_experiments",
        version=read_version(),
        packages=["divrec_experiments", "divrec_experiments.datasets"],
        url="",
        license="MIT License",
        author="amtsyplov",
        author_email="",
        description="",
    )
