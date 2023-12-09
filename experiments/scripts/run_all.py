import os
from divrec_experiments.utils import get_logger


if __name__ == '__main__':

    logger = get_logger(__file__)

    experiments_directory = os.path.abspath("experiments")
    experiments = []
    for directory in os.listdir(experiments_directory):
        if directory[0].isalpha():
            experiments.append(directory)
    logger.info(f"Found {len(experiments)} experiments: {', '.join(experiments)}")

    os.system(f"source ")
    for experiment in experiments:
        program = f"{os.path.join(experiments_directory, experiment)}/main.py"
        config = f"{os.path.join(experiments_directory, experiment)}/config.yaml"
        cmd = f"python3 {program} {config}"

        logger.info(f"Start experiment \"{experiment}\"")
        try:
            logger.info(f"Execute {cmd}")
            exit_code = os.system(cmd)
            logger.info(f"Process \"{experiment}\" finished with exit code {exit_code}")
        except Exception as exception:
            logger.error(f"Error caught while \"{experiment}\" execution:\n{exception}")
