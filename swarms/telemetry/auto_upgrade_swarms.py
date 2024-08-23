import subprocess

from loguru import logger

from swarms.telemetry.check_update import check_for_update


def auto_update():
    """auto update swarms"""
    try:
        outcome = check_for_update()
        if outcome is True:
            logger.info(
                "There is a new version of swarms available! Downloading..."
            )
            subprocess.run(["pip", "install", "-U", "swarms"])
        else:
            logger.info("swarms is up to date!")
    except Exception as e:
        logger.error(e)
