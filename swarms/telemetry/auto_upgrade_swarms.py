import os
import subprocess

from swarms.utils.loguru_logger import initialize_logger
from swarms.telemetry.check_update import check_for_update

logger = initialize_logger(log_folder="auto_upgrade_swarms")


def auto_update():
    """auto update swarms"""
    try:
        # Check if auto-update is disabled
        auto_update_enabled = os.getenv(
            "SWARMS_AUTOUPDATE_ON", "false"
        ).lower()
        if auto_update_enabled == "false":
            logger.info(
                "Auto-update is disabled via SWARMS_AUTOUPDATE_ON"
            )
            return

        outcome = check_for_update()
        if outcome is True:
            logger.info(
                "There is a new version of swarms available! Downloading..."
            )
            try:
                subprocess.run(
                    ["pip", "install", "-U", "swarms"], check=True
                )
            except subprocess.CalledProcessError:
                logger.info("Attempting to install with pip3...")
                subprocess.run(
                    ["pip3", "install", "-U", "swarms"], check=True
                )
        else:
            logger.info("swarms is up to date!")
    except Exception as e:
        logger.error(e)
