import os

import pytest

_PROVIDER_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
)

_MISSING_KEY_MARKERS = (
    "Missing credentials",
    "AuthenticationError",
    "Incorrect API key provided",
    "invalid API key",
    "API key not valid",
)


def _needs_a_key(exc: BaseException) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if any(
            marker in f"{type(exc).__name__}: {exc}"
            for marker in _MISSING_KEY_MARKERS
        ):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    rep = yield
    if (
        rep.failed
        and call.excinfo is not None
        and not any(os.environ.get(k) for k in _PROVIDER_KEYS)
        and _needs_a_key(call.excinfo.value)
    ):
        rep.outcome = "skipped"
        rep.longrepr = (
            rep.location[0],
            rep.location[1] + 1,
            "Skipped: calls a live LLM and no API key is set: "
            f"{call.excinfo.typename}: {call.excinfo.value}",
        )
    return rep
