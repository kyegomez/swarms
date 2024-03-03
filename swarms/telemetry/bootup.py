import logging
import warnings

from swarms.telemetry.auto_upgrade_swarms import auto_update
from swarms.utils.disable_logging import disable_logging


def bootup():
    """Bootup swarms"""
    disable_logging()
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    auto_update()
