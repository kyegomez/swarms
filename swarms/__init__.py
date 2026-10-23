import importlib
from typing import Any

from swarms.env import load_swarms_env

load_swarms_env()

from swarms.telemetry.bootup import bootup  # noqa: E402

bootup()

from swarms.agents import _MODULE_MAP as _AGENTS_MAP  # noqa: E402
from swarms.prompts import _MODULE_MAP as _PROMPTS_MAP  # noqa: E402
from swarms.schemas import __all__ as _SCHEMAS_ALL  # noqa: E402
from swarms.structs import _MODULE_MAP as _STRUCTS_MAP  # noqa: E402

_MODULE_MAP = {
    # Dynamically composed from subpackages so each export is defined only once
    **{name: "swarms.agents" for name in _AGENTS_MAP},
    **{name: "swarms.prompts" for name in _PROMPTS_MAP},
    **{name: "swarms.schemas" for name in _SCHEMAS_ALL},
    **{name: "swarms.structs" for name in _STRUCTS_MAP},
    # swarms.telemetry
    "ContextThreadPoolExecutor": "swarms.telemetry",
    "SwarmTelemetry": "swarms.telemetry",
    "capture_init": "swarms.telemetry",
    "generate_user_id": "swarms.telemetry",
    "get_comprehensive_system_info": "swarms.telemetry",
    "get_machine_id": "swarms.telemetry",
    "trace_run": "swarms.telemetry",
    # swarms.tools
    "BaseTool": "swarms.tools",
    "Function": "swarms.tools",
    "MCPFileTokenStorage": "swarms.tools",
    "MCPManager": "swarms.tools",
    "ToolFunction": "swarms.tools",
    "ToolStorage": "swarms.tools",
    "base_model_to_openai_function": "swarms.tools",
    "get_load_param_if_needed_function": "swarms.tools",
    "get_openai_function_schema_from_func": "swarms.tools",
    "get_parameters": "swarms.tools",
    "get_required_params": "swarms.tools",
    "load_basemodels_if_needed": "swarms.tools",
    "multi_base_model_to_openai_function": "swarms.tools",
    "scrape_tool_func_docs": "swarms.tools",
    "tool_find_by_name": "swarms.tools",
    "tool_registry": "swarms.tools",
    # swarms.utils
    "HistoryOutputType": "swarms.utils",
    "LiteLLM": "swarms.utils",
    "LiteLLMException": "swarms.utils",
    "NetworkConnectionError": "swarms.utils",
    "WorkspaceManager": "swarms.utils",
    "count_tokens": "swarms.utils",
    "create_file_in_folder": "swarms.utils",
    "exists": "swarms.utils",
    "format_data_structure": "swarms.utils",
    "format_dict_to_string": "swarms.utils",
    "history_output_formatter": "swarms.utils",
    "initialize_logger": "swarms.utils",
    "load_json": "swarms.utils",
    "sanitize_file_path": "swarms.utils",
    "zip_folders": "swarms.utils",
    "zip_workspace": "swarms.utils",
    # Subpackages
    "agents": "swarms.agents",
    "env": "swarms.env",
    "prompts": "swarms.prompts",
    "schemas": "swarms.schemas",
    "structs": "swarms.structs",
    "telemetry": "swarms.telemetry",
    "tools": "swarms.tools",
    "utils": "swarms.utils",
}

__all__ = list(_MODULE_MAP)


def __getattr__(name: str) -> Any:
    """Resolve swarms attributes lazily on first access (PEP 562).

    The version comes from the installed distribution's metadata rather
    than a literal here, so it cannot drift from ``pyproject.toml``.
    Resolving it lazily keeps the dist-info scan off the import path for
    the majority of programs, which never read it.
    """
    if name == "__version__":
        from importlib.metadata import (
            PackageNotFoundError,
            version,
        )

        try:
            resolved = version("swarms")
        except PackageNotFoundError:
            # A source checkout with no installed distribution.
            resolved = "unknown"

        globals()["__version__"] = resolved
        return resolved

    if name in _MODULE_MAP:
        module = importlib.import_module(_MODULE_MAP[name])
        attr = getattr(module, name, module)
        globals()[name] = attr
        return attr

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__() -> list:
    """Advertise all public exports before anything has accessed them."""
    return sorted(set(globals()) | set(__all__) | {"__version__"})
