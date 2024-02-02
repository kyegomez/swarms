""" init file for swarms package. """
# from swarms.telemetry.main import Telemetry  # noqa: E402, F403
from swarms.telemetry.bootup import bootup  # noqa: E402, F403

bootup()

from swarms.agents import *  # noqa: E402, F403, C0413
from swarms.structs import *  # noqa: E402, F403, C0413
from swarms.models import *  # noqa: E402, F403,    C0413
from swarms.telemetry import *  # noqa: E402, F403, C0413
from swarms.utils import *  # noqa: E402, F403, C0413
from swarms.prompts import *  # noqa: E402, F403, C0413
from swarms.tokenizers import *  # noqa: E402, F403, C0413
from swarms.loaders import *  # noqa: E402, F403, C0413
from swarms.artifacts import *  # noqa: E402, F403, C0413
from swarms.chunkers import *  # noqa: E402, F403, C0413
