# from swarms.telemetry.main import Telemetry  # noqa: E402, F403
from swarms.telemetry.bootup import bootup  # noqa: E402, F403

bootup()

from swarms.agents import *  # noqa: E402, F403
from swarms.structs import *  # noqa: E402, F403
from swarms.models import *  # noqa: E402, F403
from swarms.telemetry import *  # noqa: E402, F403
from swarms.utils import *  # noqa: E402, F403
from swarms.prompts import *  # noqa: E402, F403


# telemetry = Telemetry('mongodb://localhost:27017/', 'mydatabase')

# telemetry.log_import('swarms.telemetry.bootup')
# telemetry.log_import('swarms.agents')
# telemetry.log_import('swarms.structs')
# telemetry.log_import('swarms.models')
# telemetry.log_import('swarms.telemetry')
# telemetry.log_import('swarms.utils')
# telemetry.log_import('swarms.prompts')
