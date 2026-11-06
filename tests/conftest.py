"""Pytest configuration and global hooks for swarms test suite."""

from typing import Tuple
import pytest

CREDENTIAL_ERROR_PATTERNS: Tuple[str, ...] = (
    "missing credentials",
    "openai_api_key",
    "please pass an `api_key`",
    "please pass an api_key",
    "api key not found",
    "no api key provided",
    "missing api key",
    "incorrect api key",
    "invalid api key",
    "invalid_api_key",
    "authentication error",
    "authenticationerror",
    "unauthorized",
    "status_code: 401",
    "anthropic_api_key",
    "google_api_key",
    "cohere_api_key",
    "huggingface_api_key",
    "replicate_api_key",
    "together_api_key",
    "groq_api_key",
    "openaiexception",
    "did not find an openai_api_key",
    "did not find openai_api_key",
    "did not find anthropic_api_key",
)


def _extract_all_error_messages(exc: BaseException) -> str:
    """Recursively extract error messages from an exception and its causal chain."""
    parts = []
    curr = exc
    visited = set()
    while curr is not None and id(curr) not in visited:
        visited.add(id(curr))
        parts.append(str(curr))
        parts.append(repr(curr))
        parts.append(type(curr).__name__)
        if hasattr(curr, "last_attempt"):
            try:
                attempt = curr.last_attempt
                if attempt is not None and hasattr(
                    attempt, "exception"
                ):
                    inner = attempt.exception()
                    if inner is not None:
                        parts.append(
                            _extract_all_error_messages(inner)
                        )
            except Exception:
                pass
        curr = getattr(curr, "__cause__", None) or getattr(
            curr, "__context__", None
        )
    return " ".join(parts).lower()


def _is_credential_failure(exc: BaseException) -> bool:
    """Determine whether an exception is due to missing or invalid live API credentials."""
    all_msgs = _extract_all_error_messages(exc)
    return any(
        pattern in all_msgs for pattern in CREDENTIAL_ERROR_PATTERNS
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo
):
    """
    Skip tests that fail due to missing or unconfigured live API keys.

    When running unit tests in environments without live provider credentials (e.g.
    CI workflows on PRs from forks), tests that invoke real LLMs degrade to skipped
    rather than failing the entire test suite.
    """
    outcome = yield
    report = outcome.get_result()
    if report.when in ("setup", "call", "teardown") and report.failed:
        if (
            call.excinfo is not None
            and call.excinfo.value is not None
            and _is_credential_failure(call.excinfo.value)
        ):
            report.outcome = "skipped"
            loc = getattr(item, "location", (None, 0))
            line_no = loc[1] if (loc and len(loc) > 1) else 0
            report.longrepr = (
                str(item.fspath),
                line_no,
                f"Skipped: live API credentials not configured ({call.excinfo.value})",
            )
