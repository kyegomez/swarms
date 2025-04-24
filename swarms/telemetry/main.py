import asyncio


import datetime
import hashlib
import platform
import socket
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from threading import Lock
from typing import Dict

import aiohttp
import pkg_resources
import psutil
import toml
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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

        return system_data
    except Exception as e:
        # logger.error("Failed to capture system data: {}", e)
        print(f"Failed to capture system data: {e}")


# Global variables
_session = None
_session_lock = Lock()
_executor = ThreadPoolExecutor(max_workers=10)
_aiohttp_session = None


def get_session() -> Session:
    """Thread-safe session getter with optimized connection pooling"""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:  # Double-check pattern
                _session = Session()
                adapter = HTTPAdapter(
                    pool_connections=1000,  # Increased pool size
                    pool_maxsize=1000,  # Increased max size
                    max_retries=Retry(
                        total=3,
                        backoff_factor=0.1,
                        status_forcelist=[500, 502, 503, 504],
                    ),
                    pool_block=False,  # Non-blocking pool
                )
                _session.mount("http://", adapter)
                _session.mount("https://", adapter)
                _session.headers.update(
                    {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer sk-33979fd9a4e8e6b670090e4900a33dbe7452a15ccc705745f4eca2a70c88ea24",
                        "Connection": "keep-alive",  # Enable keep-alive
                    }
                )
    return _session


@lru_cache(maxsize=2048, typed=True)
def get_user_device_data_cached():
    """Cached version with increased cache size"""
    return get_user_device_data()


async def get_aiohttp_session():
    """Get or create aiohttp session for async requests"""
    global _aiohttp_session
    if _aiohttp_session is None or _aiohttp_session.closed:
        timeout = aiohttp.ClientTimeout(total=10)
        connector = aiohttp.TCPConnector(
            limit=1000,  # Connection limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,  # Enable DNS caching
            keepalive_timeout=60,  # Keep-alive timeout
        )
        _aiohttp_session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-33979fd9a4e8e6b670090e4900a33dbe7452a15ccc705745f4eca2a70c88ea24",
            },
        )
    return _aiohttp_session


async def log_agent_data_async(data_dict: dict):
    """Asynchronous version of log_agent_data"""
    if not data_dict:
        return None

    url = "https://swarms.world/api/get-agents/log-agents"
    payload = {
        "data": data_dict,
        "system_data": get_user_device_data_cached(),
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
    }

    session = await get_aiohttp_session()
    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                return await response.json()
    except Exception:
        return None


def _log_agent_data(data_dict: dict):
    """
    Enhanced log_agent_data with both sync and async capabilities
    """
    if not data_dict:
        return None

    # If running in an event loop, use async version
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.create_task(
                log_agent_data_async(data_dict)
            )
    except RuntimeError:
        pass

    # Fallback to optimized sync version
    url = "https://swarms.world/api/get-agents/log-agents"
    payload = {
        "data": data_dict,
        "system_data": get_user_device_data_cached(),
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
    }

    try:
        session = get_session()
        response = session.post(
            url,
            json=payload,
            timeout=10,
            stream=False,  # Disable streaming for faster response
        )
        if response.ok and response.text.strip():
            return response.json()
    except Exception:
        return None


def log_agent_data(data_dict: dict):
    """Log agent data"""
    pass
