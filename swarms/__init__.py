import logging
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# disable tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    log = logging.getLogger("pytorch")
    log.propagate = False
    log.setLevel(logging.ERROR)
except Exception as error:
    print(f"Pytorch logging not disabled: {error}")

from swarms.agents import *  # noqa: E402, F403
from swarms.swarms import *  # noqa: E402, F403
from swarms.structs import *  # noqa: E402, F403
from swarms.models import *  # noqa: E402, F403
