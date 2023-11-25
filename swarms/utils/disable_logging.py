import logging
import os
import warnings


def disable_logging():
    warnings.filterwarnings("ignore", category=UserWarning)

    # disable tensorflow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Set the logging level for the entire module
    logging.basicConfig(level=logging.WARNING)

    try:
        log = logging.getLogger("pytorch")
        log.propagate = False
        log.setLevel(logging.ERROR)
    except Exception as error:
        print(f"Pytorch logging not disabled: {error}")

    for logger_name in [
        "tensorflow",
        "h5py",
        "numexpr",
        "git",
        "wandb.docker.auth",
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)  # Supress DEBUG and info logs
