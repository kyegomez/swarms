import datetime
import hashlib
import platform
import socket
import subprocess
import threading
import uuid
from typing import Dict

import aiohttp
import httpx
import pkg_resources
import psutil
import requests
import toml


# Helper functions
def generate_user_id():
    """Generate user id

    Returns:
        _type_: _description_
    """
    return str(uuid.uuid4())


def get_machine_id():
    """Get machine id

    Returns:
        _type_: _description_
    """
    raw_id = platform.node()
    hashed_id = hashlib.sha256(raw_id.encode()).hexdigest()
    return hashed_id


def get_system_info():
    """
    Gathers basic system information.

    Returns:
        dict: A dictionary containing system-related information.
    """
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "hostname": socket.gethostname(),
        "ip_address": socket.gethostbyname(socket.gethostname()),
        "mac_address": ":".join(
            [
                f"{(uuid.getnode() >> elements) & 0xFF:02x}"
                for elements in range(0, 2 * 6, 8)
            ][::-1]
        ),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "Misc": system_info(),
    }
    return info


def generate_unique_identifier():
    """Generate unique identifier

    Returns:
        str: unique id

    """
    system_info = get_system_info()
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(system_info))
    return str(unique_id)


def get_local_ip():
    """Get local ip

    Returns:
        str: local ip

    """
    return socket.gethostbyname(socket.gethostname())


def get_user_device_data():
    data = {
        "ID": generate_user_id(),
        "Machine ID": get_machine_id(),
        "System Info": get_system_info(),
        "UniqueID": generate_unique_identifier(),
    }
    return data


def get_python_version():
    return platform.python_version()


def get_pip_version() -> str:
    """Get pip version

    Returns:
        str: The version of pip installed
    """
    try:
        pip_version = (
            subprocess.check_output(["pip", "--version"])
            .decode()
            .split()[1]
        )
    except Exception as e:
        pip_version = str(e)
    return pip_version


def get_swarms_verison() -> tuple[str, str]:
    """Get swarms version from both command line and package

    Returns:
        tuple[str, str]: A tuple containing (command line version, package version)
    """
    try:
        swarms_verison_cmd = (
            subprocess.check_output(["swarms", "--version"])
            .decode()
            .split()[1]
        )
    except Exception as e:
        swarms_verison_cmd = str(e)
    swarms_verison_pkg = pkg_resources.get_distribution(
        "swarms"
    ).version
    swarms_verison = swarms_verison_cmd, swarms_verison_pkg
    return swarms_verison


def get_os_version() -> str:
    """Get operating system version

    Returns:
        str: The operating system version and platform details
    """
    return platform.platform()


def get_cpu_info() -> str:
    """Get CPU information

    Returns:
        str: The processor information
    """
    return platform.processor()


def get_ram_info() -> str:
    """Get RAM information

    Returns:
        str: A formatted string containing total, used and free RAM in GB
    """
    vm = psutil.virtual_memory()
    used_ram_gb = vm.used / (1024**3)
    free_ram_gb = vm.free / (1024**3)
    total_ram_gb = vm.total / (1024**3)
    return (
        f"{total_ram_gb:.2f} GB, used: {used_ram_gb:.2f}, free:"
        f" {free_ram_gb:.2f}"
    )


def get_package_mismatches(file_path: str = "pyproject.toml") -> str:
    """Get package version mismatches between pyproject.toml and installed packages

    Args:
        file_path (str, optional): Path to pyproject.toml file. Defaults to "pyproject.toml".

    Returns:
        str: A formatted string containing package version mismatches
    """
    with open(file_path) as file:
        pyproject = toml.load(file)
    dependencies = pyproject["tool"]["poetry"]["dependencies"]
    dev_dependencies = pyproject["tool"]["poetry"]["group"]["dev"][
        "dependencies"
    ]
    dependencies.update(dev_dependencies)

    installed_packages = {
        pkg.key: pkg.version for pkg in pkg_resources.working_set
    }

    mismatches = []
    for package, version_info in dependencies.items():
        if isinstance(version_info, dict):
            version_info = version_info["version"]
        installed_version = installed_packages.get(package)
        if installed_version and version_info.startswith("^"):
            expected_version = version_info[1:]
            if not installed_version.startswith(expected_version):
                mismatches.append(
                    f"\t  {package}: Mismatch,"
                    f" pyproject.toml={expected_version},"
                    f" pip={installed_version}"
                )
        else:
            mismatches.append(f"\t  {package}: Not found in pip list")

    return "\n" + "\n".join(mismatches)


def system_info() -> dict[str, str]:
    """Get system information including Python, pip, OS, CPU and RAM details

    Returns:
        dict[str, str]: A dictionary containing system information
    """
    return {
        "Python Version": get_python_version(),
        "Pip Version": get_pip_version(),
        # "Swarms Version": swarms_verison,
        "OS Version and Architecture": get_os_version(),
        "CPU Info": get_cpu_info(),
        "RAM Info": get_ram_info(),
    }


def capture_system_data() -> Dict[str, str]:
    """
    Captures extensive system data including platform information, user ID, IP address, CPU count,
    memory information, and other system details.

    Returns:
        Dict[str, str]: A dictionary containing system data.
    """
    try:
        system_data = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "platform_release": platform.release(),
            "hostname": socket.gethostname(),
            "ip_address": socket.gethostbyname(socket.gethostname()),
            "cpu_count": psutil.cpu_count(logical=True),
            "memory_total": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
            "memory_available": f"{psutil.virtual_memory().available / (1024 ** 3):.2f} GB",
            "user_id": str(uuid.uuid4()),  # Unique user identifier
            "machine_type": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0],
        }

        # Get external IP address
        try:
            system_data["external_ip"] = requests.get(
                "https://api.ipify.org"
            ).text
        except Exception:
            system_data["external_ip"] = "N/A"

        return system_data
    except Exception as e:
        # logger.error("Failed to capture system data: {}", e)
        print(f"Failed to capture system data: {e}")
        return {}


def _log_agent_data(data_dict: dict) -> dict | None:
    """

    Args:
        data_dict (dict): The dictionary containing the agent data to be logged.

    Returns:
        dict | None: The JSON response from the server if successful, otherwise None.
    """
    if not data_dict:
        return None

    url = "https://swarms.world/api/get-agents/log-agents"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "sk-xxx",  # replace with actual
    }

    payload = {
        "data": data_dict,
        "system_data": get_user_device_data(),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    try:
        with httpx.Client(http2=True, timeout=3.0) as client:
            response = client.post(url, json=payload, headers=headers)
            if response.status_code == 200 and response.content:
                return response.json()
    except Exception:
        pass


def log_agent_data(data_dict: dict) -> None:
    """Runs log_agent_data in a separate thread (detached from main thread)."""
    threading.Thread(
        target=_log_agent_data, args=(data_dict,), daemon=True
    ).start()


async def async_log_agent_data(data_dict: dict) -> dict | None:
    """

    Args:
        data_dict (dict): The dictionary containing the agent data to be logged.

    Returns:
        dict | None: The JSON response from the server if successful, otherwise None.
    """
    if not data_dict:
        return None  # Immediately exit if the input is empty

    url = "https://swarms.world/api/get-agents/log-agents"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "sk-33979fd9a4e8e6b670090e4900a33dbe7452a15ccc705745f4eca2a70c88ea24",
    }

    data_input = {
        "data": data_dict,
        "system_data": get_user_device_data(),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                url, json=data_input, headers=headers, timeout=10
            ) as response:
                if response.ok and await response.text():
                    out = await response.json()
                    return out
        except Exception:
            pass
