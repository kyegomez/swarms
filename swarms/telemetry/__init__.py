# from swarms.telemetry.posthog_utils import posthog

from swarms.telemetry.log_all import log_all_calls, log_calls
from swarms.telemetry.sys_info import (
    get_cpu_info,
    get_swarms_verison,
    get_os_version,
    get_package_mismatches,
    get_pip_version,
    get_python_version,
    get_ram_info,
    system_info,
)
from swarms.telemetry.user_utils import (
    generate_unique_identifier,
    generate_user_id,
    get_machine_id,
    get_system_info,
    get_user_device_data,
)

# # Capture data from the user's device
# posthog.capture(
#     "User Device Data",
#     str(get_user_device_data()),
# )

# # Capture system information
# posthog.capture(
#     "System Information",
#     str(system_info()),
# )

# # Capture the user's unique identifier
# posthog.capture(
#     "User Unique Identifier",
#     str(generate_unique_identifier()),
# )


__all__ = [
    "log_all_calls",
    "log_calls",
    "generate_user_id",
    "get_machine_id",
    "get_system_info",
    "generate_unique_identifier",
    "get_python_version",
    "get_pip_version",
    "get_swarms_verison",
    "get_os_version",
    "get_cpu_info",
    "get_ram_info",
    "get_package_mismatches",
    "system_info",
    "get_user_device_data",
]
