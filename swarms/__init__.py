# disable warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# disable tensorflow warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# disable logging for
# import logging

# logging.getLogger("requests").setLevel(logging.WARNING)
# logging.getLogger("urllib3").setLevel(logging.WARNING)

# swarms
from swarms import agents
from swarms.swarms.orchestrate import Orchestrator
from swarms import swarms
from swarms import structs
from swarms import models
from swarms.workers.worker import Worker
from swarms import workers
from swarms.logo import logo2

print(logo2)
