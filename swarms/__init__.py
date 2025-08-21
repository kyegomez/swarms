from dotenv import load_dotenv

load_dotenv()

from swarms.telemetry.bootup import bootup  # noqa: E402, F403

bootup()

from swarms.agents import *  # noqa: E402, F403
from swarms.artifacts import *  # noqa: E402, F403
from swarms.prompts import *  # noqa: E402, F403
from swarms.schemas import *  # noqa: E402, F403
from swarms.structs import *  # noqa: E402, F403
from swarms.telemetry import *  # noqa: E402, F403
from swarms.tools import *  # noqa: E402, F403
from swarms.utils import *  # noqa: E402, F403
