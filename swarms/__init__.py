import os

from swarms.telemetry.bootup import bootup  # noqa: E402, F403
from swarms.telemetry.sentry_active import activate_sentry

os.environ["WANDB_SILENT"] = "true"

bootup()
activate_sentry()


from swarms.agents import *  # noqa: E402, F403
from swarms.artifacts import *  # noqa: E402, F403
from swarms.chunkers import *  # noqa: E402, F403
from swarms.memory import *  # noqa: E402, F403
from swarms.models import *  # noqa: E402, F403
from swarms.prompts import *  # noqa: E402, F403
from swarms.structs import *  # noqa: E402, F403
from swarms.telemetry import *  # noqa: E402, F403
from swarms.tools import *  # noqa: E402, F403
from swarms.utils import *  # noqa: E402, F403
from swarms.schedulers import *  # noqa: E402, F403
