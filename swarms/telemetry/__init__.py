""" This module lists all the telemetry related functions. """
from swarms.telemetry.log_all import log_all_calls, log_calls

# from swarms.telemetry.posthog_utils import log_activity_posthog
from swarms.telemetry.user_utils import (
    generate_user_id,
    get_machine_id,
    get_system_info,
    generate_unique_identifier,
)


__all__ = [
    "log_all_calls",
    "log_calls",
    # "log_activity_posthog",
    "generate_user_id",
    "get_machine_id",
    "get_system_info",
    "generate_unique_identifier",
    "get_python_version", # from swarms/telemetry/sys_info.py
    "get_pip_version",
    "get_oi_version",
    "get_os_version",
    "get_cpu_info",
    "get_ram_info",
    "get_package_mismatches",
    "interpreter_info",
    "system_info",
]
