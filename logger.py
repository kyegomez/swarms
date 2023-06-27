import logging

from env import settings

logger = logging.getLogger()
formatter = logging.Formatter("%(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

if settings["LOG_LEVEL"] == "DEBUG":
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)