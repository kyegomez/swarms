from dotenv import load_dotenv

load_dotenv()

from swarms.telemetry.bootup import bootup

bootup()

from swarms.agents import *  # noqa: F403
from swarms.artifacts import *  # noqa: F403
from swarms.prompts import *  # noqa: F403
from swarms.schemas import *  # noqa: F403
from swarms.structs import *  # noqa: F403
from swarms.telemetry import *  # noqa: F403
from swarms.tools import *  # noqa: F403
from swarms.utils import *  # noqa: F403
