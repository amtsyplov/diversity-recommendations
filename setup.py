import os
from typing import List

from setuptools import setup


def read_version() -> str:
    with open(os.path.abspath("build.number"), mode="r") as file:
        version = file.read()
    return version


def read_requirements() -> List[str]:
    with open(os.path.abspath("divrec/requirements.txt"), mode="r") as file:
        requirements = [package.strip() for package in file.readlines()]
    return requirements


if __name__ == "__main__":
    setup(
        name="divrec",
        version=read_version(),
        packages=[
            "divrec",
            "divrec.train",
            "divrec.utils",
            "divrec.losses",
            "divrec.models",
            "divrec.loaders",
            "divrec.metrics",
            "divrec.datasets",
        ],
        url="",
        license="MIT License",
        author="amtsyplov",
        author_email="",
        description="",
        install_requires=read_requirements(),
    )
