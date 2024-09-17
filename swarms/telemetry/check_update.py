import importlib.util
import sys

import pkg_resources
import requests
from packaging import version


# borrowed from: https://stackoverflow.com/a/1051266/656011
def check_for_package(package: str) -> bool:
    if package in sys.modules:
        return True
    elif (spec := importlib.util.find_spec(package)) is not None:
        try:
            module = importlib.util.module_from_spec(spec)

            sys.modules[package] = module
            spec.loader.exec_module(module)

            return True
        except ImportError:
            return False
    else:
        return False


def check_for_update() -> bool:
    """Check for updates

    Returns:
        BOOL: Flag to indicate if there is an update
    """
    # Fetch the latest version from the PyPI API
    response = requests.get("https://pypi.org/pypi/swarms/json")
    latest_version = response.json()["info"]["version"]

    # Get the current version using pkg_resources
    current_version = pkg_resources.get_distribution("swarms").version

    return version.parse(latest_version) > version.parse(
        current_version
    )
