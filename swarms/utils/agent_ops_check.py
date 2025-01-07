from swarms.utils.loguru_logger import logger
import os


def try_import_agentops(*args, **kwargs):
    try:
        logger.info("Trying to import agentops")
        import agentops

        agentops.init(os.getenv("AGENTOPS_API_KEY"), *args, **kwargs)

        return "agentops imported successfully."
    except ImportError:
        logger.error("Could not import agentops")


def end_session_agentops():
    try:
        logger.info("Trying to end session")
        import agentops

        agentops.end_session("Success")
        return "Session ended successfully."
    except ImportError:
        logger.error("Could not import agentops")
        return "Could not end session."
