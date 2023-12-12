import os
from divrec_experiments.utils import get_logger


EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments")


if __name__ == '__main__':

    logger = get_logger(__file__)

    experiments = []
    for directory in os.listdir(EXPERIMENTS_DIR):
        if directory[0].isalpha():
            experiments.append(directory)
    logger.info(f"Found {len(experiments)} experiments: {', '.join(experiments)}")

    os.system(f"source ")
    for experiment in experiments:
        program = f"{os.path.join(EXPERIMENTS_DIR, experiment)}/main.py"
        config = f"{os.path.join(EXPERIMENTS_DIR, experiment)}/config.yaml"
        cmd = f"python3 {program} {config}"

        logger.info(f"Start experiment \"{experiment}\"")
        try:
            logger.info(f"Execute {cmd}")
            exit_code = os.system(cmd)
            logger.info(f"Process \"{experiment}\" finished with exit code {exit_code}")
        except Exception as exception:
            logger.error(f"Error caught while \"{experiment}\" execution:\n{exception}")
