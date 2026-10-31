import os

import pytest
from loguru import logger

from swarms.utils import loguru_logger as mod
from swarms.utils.loguru_logger import (
    get_log_dir,
    initialize_logger,
)


@pytest.fixture(autouse=True)
def reset_logger_state(monkeypatch):
    """Give each test a logger that has not been configured yet."""
    monkeypatch.setattr(mod, "_CONFIGURED", False, raising=False)
    monkeypatch.setattr(mod, "_CONFIGURED_DIR", None, raising=False)
    monkeypatch.setenv("SWARMS_LOG_FILES", "True")
    yield
    # Close any file handlers pointed at a tmp_path that is about to vanish,
    # then leave a plain console handler so later test modules still log.
    logger.remove()
    logger.add(lambda _: None)


class FakeMessage(str):
    """Stand-in for loguru's Message: a str carrying a ``record``."""

    def __new__(cls, text, name):
        obj = super().__new__(cls, text)
        obj.record = {"name": name}
        return obj


# ---------------------------------------------------------------------------
# get_log_dir
# ---------------------------------------------------------------------------


def test_get_log_dir_uses_workspace_dir(monkeypatch):
    monkeypatch.setenv("WORKSPACE_DIR", "/tmp/somewhere")
    assert get_log_dir() == os.path.join("/tmp/somewhere", "logs")


def test_get_log_dir_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv("WORKSPACE_DIR", raising=False)
    assert get_log_dir() == os.path.join("agent_workspace", "logs")


def test_get_log_dir_falls_back_when_empty(monkeypatch):
    """An empty string is not a usable directory, so it must not be honoured."""
    monkeypatch.setenv("WORKSPACE_DIR", "")
    assert get_log_dir() == os.path.join("agent_workspace", "logs")


# ---------------------------------------------------------------------------
# initialize_logger: log_folder is a label, not a path
# ---------------------------------------------------------------------------


def test_log_folder_argument_creates_no_directory(
    monkeypatch, tmp_path
):
    """The 28-stray-directories bug: log_folder was used as a path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "ws"))

    initialize_logger("graph_workflow")

    assert not (tmp_path / "graph_workflow").exists()
    assert (tmp_path / "ws" / "logs").is_dir()


def test_logs_land_under_workspace_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "ws"))

    initialize_logger("anything")
    logger.info("hello")
    logger.remove()  # flush the enqueued file handler

    combined = list((tmp_path / "ws" / "logs").glob("swarms_*.log"))
    assert len(combined) == 1
    assert "hello" in combined[0].read_text()


# ---------------------------------------------------------------------------
# initialize_logger: configured once, but rebuilt when the directory moves
# ---------------------------------------------------------------------------


def test_repeat_calls_return_the_same_logger(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "ws"))
    assert initialize_logger("a") is initialize_logger("b")


def test_repeat_calls_do_not_tear_down_handlers(
    monkeypatch, tmp_path
):
    """Every module calls this at import; a later call must not wipe the
    handlers the earlier ones installed."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "ws"))
    initialize_logger("first")

    received = []
    logger.add(lambda m: received.append(m), level="INFO")

    initialize_logger("second")
    logger.info("after the second init")

    assert received, "a sink added before the second call was removed"


def test_changing_workspace_dir_rebuilds_handlers(
    monkeypatch, tmp_path
):
    """bootup() settles WORKSPACE_DIR after the first initialize_logger call,
    so a directory change early in the process must be picked up."""
    first = tmp_path / "first"
    second = tmp_path / "second"

    monkeypatch.setenv("WORKSPACE_DIR", str(first))
    initialize_logger("mod")

    monkeypatch.setenv("WORKSPACE_DIR", str(second))
    initialize_logger("mod")
    logger.info("written after the move")
    logger.remove()

    moved = list((second / "logs").glob("swarms_*.log"))
    assert moved, "handlers still point at the old directory"
    assert "written after the move" in moved[0].read_text()


# ---------------------------------------------------------------------------
# initialize_logger: an unusable directory must not break the import
# ---------------------------------------------------------------------------


def test_unusable_log_dir_degrades_to_console(monkeypatch, tmp_path):
    """WORKSPACE_DIR is caller-supplied; makedirs on it can raise, and an
    uncaught OSError here would take down `import swarms`."""
    blocker = tmp_path / "blocked"
    blocker.write_text("this is a file, not a directory")
    monkeypatch.setenv("WORKSPACE_DIR", str(blocker))

    initialize_logger("mod")  # must not raise

    received = []
    logger.add(lambda m: received.append(m), level="INFO")
    logger.info("console still works")

    assert received
    assert not (blocker / "logs").exists()


# ---------------------------------------------------------------------------
# _module_log_router
# ---------------------------------------------------------------------------


def test_router_writes_one_file_per_module(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    (tmp_path / "logs").mkdir()

    mod._module_log_router(
        FakeMessage("gw line\n", "swarms.structs.graph_workflow")
    )
    mod._module_log_router(
        FakeMessage("agent line\n", "swarms.structs.agent")
    )

    logs = tmp_path / "logs"
    assert (logs / "graph_workflow.log").read_text() == "gw line\n"
    assert (logs / "agent.log").read_text() == "agent line\n"


def test_router_ignores_non_swarms_records(monkeypatch, tmp_path):
    """Third-party libraries share loguru's global logger; their records must
    not spawn files named after them."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    (tmp_path / "logs").mkdir()

    mod._module_log_router(FakeMessage("noise\n", "httpx._client"))
    mod._module_log_router(FakeMessage("noise\n", "__main__"))

    assert list((tmp_path / "logs").iterdir()) == []


def test_router_rotates_past_the_size_ceiling(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(mod, "MODULE_LOG_MAX_BYTES", 10)

    target = logs / "agent.log"
    target.write_text("x" * 50)

    mod._module_log_router(
        FakeMessage("fresh\n", "swarms.structs.agent")
    )

    assert (logs / "agent.log.1").read_text() == "x" * 50
    assert target.read_text() == "fresh\n"


def test_router_survives_an_unwritable_directory(
    monkeypatch, tmp_path
):
    """Logging must never raise into the caller."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "missing"))

    mod._module_log_router(
        FakeMessage("line\n", "swarms.structs.agent")
    )  # no exception


# ---------------------------------------------------------------------------
# SWARMS_LOG_FILES
# ---------------------------------------------------------------------------


def test_no_log_files_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("SWARMS_LOG_FILES", raising=False)
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "ws"))

    initialize_logger("mod")
    logger.info("console only")
    logger.remove()

    assert not (tmp_path / "ws" / "logs").exists()


def test_false_writes_no_log_files(monkeypatch, tmp_path):
    monkeypatch.setenv("SWARMS_LOG_FILES", "False")
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "ws"))

    initialize_logger("mod")
    logger.info("console only")
    logger.remove()

    assert not (tmp_path / "ws" / "logs").exists()


def test_turning_log_files_on_rebuilds_handlers(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "ws"))
    monkeypatch.setenv("SWARMS_LOG_FILES", "False")
    initialize_logger("mod")
    logger.info("while off")

    monkeypatch.setenv("SWARMS_LOG_FILES", "True")
    initialize_logger("mod")
    logger.info("after turning on")
    logger.remove()

    written = next(
        (tmp_path / "ws" / "logs").glob("swarms_*.log")
    ).read_text()
    assert "after turning on" in written
    assert "while off" not in written




# ---------------------------------------------------------------------------
# Issue #2717: diagnose off on file sinks to prevent leaking secrets
# ---------------------------------------------------------------------------


def _fail_with_secret_in_frame(param):
    raise ValueError("simulated network failure")


def _execute_api_call_with_secret(secret_token: str):
    headers = {"Authorization": f"Bearer {secret_token}"}
    _fail_with_secret_in_frame(headers)


def test_file_sink_does_not_diagnose_local_variables_or_secrets(
    monkeypatch, tmp_path
):
    """Diagnose on file sinks leaks local variables and api keys to disk."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "ws"))
    monkeypatch.setenv("SWARMS_LOG_FILES", "True")

    initialize_logger("mod")

    secret_key = "sk-live-super-secret-key-123456789"
    try:
        _execute_api_call_with_secret(secret_key)
    except ValueError:
        logger.exception("API call failed")

    logger.remove()  # flush enqueued file handler

    combined = list((tmp_path / "ws" / "logs").glob("swarms_*.log"))
    assert len(combined) == 1
    log_content = combined[0].read_text(encoding="utf-8")

    assert "simulated network failure" in log_content
    assert "API call failed" in log_content
    assert secret_key not in log_content


def test_console_sink_diagnose_respects_verbose_global(monkeypatch):
    """Console diagnose must only be enabled when SWARMS_VERBOSE_GLOBAL is true."""
    monkeypatch.setenv("SWARMS_VERBOSE_GLOBAL", "False")
    initialize_logger("mod")

    handlers = list(logger._core.handlers.values())
    assert len(handlers) >= 1
    console_handler = handlers[0]
    assert console_handler._exception_formatter._diagnose is False


def test_console_sink_diagnose_enabled_when_verbose_global_true(monkeypatch):
    monkeypatch.setenv("SWARMS_VERBOSE_GLOBAL", "True")
    initialize_logger("mod")

    handlers = list(logger._core.handlers.values())
    assert len(handlers) >= 1
    console_handler = handlers[0]
    assert console_handler._exception_formatter._diagnose is True


def test_file_sink_diagnose_stays_false_even_if_verbose_global_true(
    monkeypatch, tmp_path
):
    """SWARMS_VERBOSE_GLOBAL=True must only enable diagnose on the console, never files."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "ws"))
    monkeypatch.setenv("SWARMS_LOG_FILES", "True")
    monkeypatch.setenv("SWARMS_VERBOSE_GLOBAL", "True")

    initialize_logger("mod")

    secret_key = "sk-live-secret-never-in-file-5555"
    try:
        _execute_api_call_with_secret(secret_key)
    except ValueError:
        logger.exception("Verbose error")

    logger.remove()

    combined = list((tmp_path / "ws" / "logs").glob("swarms_*.log"))
    assert len(combined) == 1
    log_content = combined[0].read_text(encoding="utf-8")

    assert "simulated network failure" in log_content
    assert secret_key not in log_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
