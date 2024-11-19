import importlib.util
import sys

import pkg_resources
import requests
from packaging import version
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger("check-update")


# borrowed from: https://stackoverflow.com/a/1051266/656011
def check_for_package(package: str) -> bool:
    """
    Checks if a package is installed and available for import.

    Args:
        package (str): The name of the package to check.

    Returns:
        bool: True if the package is installed and can be imported, False otherwise.
    """
    if package in sys.modules:
        return True
    elif (spec := importlib.util.find_spec(package)) is not None:
        try:
            module = importlib.util.module_from_spec(spec)

            sys.modules[package] = module
            spec.loader.exec_module(module)

            return True
        except ImportError:
            logger.error(f"Failed to import {package}")
            return False
    else:
        logger.info(f"{package} not found")
        return False


def check_for_update() -> bool:
    """
    Checks if there is an update available for the swarms package.

    Returns:
        bool: True if an update is available, False otherwise.
    """
    try:
        # Fetch the latest version from the PyPI API
        response = requests.get("https://pypi.org/pypi/swarms/json")
        response.raise_for_status()  # Raises an HTTPError if the response status code is 4XX/5XX
        latest_version = response.json()["info"]["version"]

        # Get the current version using pkg_resources
        current_version = pkg_resources.get_distribution(
            "swarms"
        ).version

        if version.parse(latest_version) > version.parse(
            current_version
        ):
            logger.info(
                f"Update available: {latest_version} > {current_version}"
            )
            return True
        else:
            logger.info(
                f"No update available: {latest_version} <= {current_version}"
            )
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to check for update: {e}")
        return False
