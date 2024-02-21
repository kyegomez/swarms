from swarms.telemetry.auto_upgrade_swarms import auto_update
from swarms.utils.disable_logging import disable_logging


def bootup():
    """Bootup swarms"""
    disable_logging()
    auto_update()
