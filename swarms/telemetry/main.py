import os
import datetime
import hashlib
import platform
import socket
import uuid
from typing import Any, Dict

import psutil
import requests
from functools import lru_cache


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


@lru_cache(maxsize=1)
def get_comprehensive_system_info() -> Dict[str, Any]:
    # Basic platform and hardware information
    system_data = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "platform_full": platform.platform(),
        "architecture": platform.machine(),
        "architecture_details": platform.architecture()[0],
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
    }

    # MAC address
    try:
        system_data["mac_address"] = ":".join(
            [
                f"{(uuid.getnode() >> elements) & 0xFF:02x}"
                for elements in range(0, 2 * 6, 8)
            ][::-1]
        )
    except Exception as e:
        system_data["mac_address"] = f"Error: {str(e)}"

    # CPU information
    system_data["cpu_count_logical"] = psutil.cpu_count(logical=True)
    system_data["cpu_count_physical"] = psutil.cpu_count(
        logical=False
    )

    # Memory information
    vm = psutil.virtual_memory()
    total_ram_gb = vm.total / (1024**3)
    used_ram_gb = vm.used / (1024**3)
    free_ram_gb = vm.free / (1024**3)
    available_ram_gb = vm.available / (1024**3)

    system_data.update(
        {
            "memory_total_gb": f"{total_ram_gb:.2f}",
            "memory_used_gb": f"{used_ram_gb:.2f}",
            "memory_free_gb": f"{free_ram_gb:.2f}",
            "memory_available_gb": f"{available_ram_gb:.2f}",
            "memory_summary": f"Total: {total_ram_gb:.2f} GB, Used: {used_ram_gb:.2f} GB, Free: {free_ram_gb:.2f} GB, Available: {available_ram_gb:.2f} GB",
        }
    )

    # Python version
    system_data["python_version"] = platform.python_version()

    # Generate unique identifier based on system info
    try:
        unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(system_data))
        system_data["unique_identifier"] = str(unique_id)
    except Exception as e:
        system_data["unique_identifier"] = f"Error: {str(e)}"

    return system_data


def _log_agent_data(data_dict: dict):
    """
    Logs agent data and system information to the swarms.world telemetry endpoint via a POST request.

    This function is a low-level, internal utility that sends the provided agent state along with current
    system telemetry to the Swarms service for analytics and diagnostics. Data includes a timestamp,
    comprehensive system information, and the state of the agent as passed in `data_dict`.

    Args:
        data_dict (dict): Dictionary representing the current agent's state/config/data.

    Side Effects:
        Sends a POST request to the Swarms telemetry endpoint.
        Does not raise exceptions on failed request (silent fail).

    Security Warning:
        The authorization key is included in the request header.
        Remove or rotate keys as necessary for production security.

    Returns:
        None
    """
    url = "https://swarms.world/api/get-agents/log-agents"

    log = {
        "data": data_dict,
        "system_data": get_comprehensive_system_info(),
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
    }

    payload = {
        "data": log,
    }

    key = os.getenv("SWARMS_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "Authorization": key,
    }

    response = requests.post(
        url, json=payload, headers=headers, timeout=10
    )

    try:
        if response.status_code == 200:
            return
    except Exception:
        pass


def log_agent_data(data_dict: dict):
    """
    Public wrapper to log agent data and telemetry if telemetry is enabled.

    This function checks the 'SWARMS_TELEMETRY' environment variable. If set to the string "true",
    it records agent telemetry using the internal _log_agent_data function.
    Otherwise, it does nothing.

    Args:
        data_dict (dict): Agent data to be transmitted if telemetry is enabled.

    Returns:
        None
    """
    get_telemetry = os.getenv("SWARMS_TELEMETRY_ON")

    if get_telemetry == "True" or get_telemetry == "true":
        _log_agent_data(data_dict)
    else:
        pass
