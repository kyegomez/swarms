import logging

logger = logging.getLogger()
formatter = logging.Formatter("%(message)s")

ch = logging.StreamHandler()

ch.setFormatter(formatter)

logger.addHandler(ch)

logger.setLevel(logging.DEBUG)
