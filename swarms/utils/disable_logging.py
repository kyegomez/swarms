import logging
import os
import warnings


def set_logger_level(logger_name: str) -> None:
    """
    Sets the logging level for a specific logger to CRITICAL.

    Args:
        logger_name (str): The name of the logger to modify.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)


def disable_logging():
    """
    Silences noisy third-party loggers by raising their levels.

    The root logger belongs to the application importing swarms, so it is
    left untouched.
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    # disable tensorflow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    try:
        log = logging.getLogger("pytorch")
        log.propagate = False
        log.setLevel(logging.ERROR)
    except Exception as error:
        print(f"Pytorch logging not disabled: {error}")

    logger_names = [
        "tensorflow",
        "h5py",
        "numexpr",
        "git",
        "wandb.docker.auth",
        "distutils",
        "urllib3",
        "elasticsearch",
        "packaging",
    ]

    # Set the level for each logger sequentially, no concurrent futures
    for logger_name in logger_names:
        set_logger_level(logger_name)
