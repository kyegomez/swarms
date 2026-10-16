"""Choosing between RouteHub and litellm, and what changes with each.

Offline: model calls are stubbed, an installed or missing library is
simulated through importlib.util.find_spec, and backend names are replaced
in the module's namespace so nothing is fetched.
"""

import importlib
import importlib.util
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from swarms import Agent
from swarms.utils import llm_backend
from swarms.utils.litellm_wrapper import LiteLLM

BACKEND_OPTIONS = (
    "set_verbose",
    "ssl_verify",
    "num_retries",
    "drop_params",
)


def _with_installed(monkeypatch, *names):
    """Make find_spec report only the given backends as installed."""
    real = importlib.util.find_spec

    def find_spec(name, *args):
        if name in llm_backend.BACKENDS:
            return object() if name in names else None
        return real(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def _use_backend(monkeypatch, backend, **names):
    """Select a backend and replace names it would otherwise provide."""
    monkeypatch.setattr(llm_backend, "BACKEND", backend)
    if backend == "litellm":
        # LiteLLM() writes litellm's module settings, so they are restored afterwards.
        litellm = importlib.import_module("litellm")
        for name in BACKEND_OPTIONS:
            monkeypatch.setattr(litellm, name, getattr(litellm, name))
        names.setdefault("_module", litellm)
    namespace = vars(llm_backend)
    for name, value in names.items():
        monkeypatch.setitem(namespace, name, value)


def _response():
    message = SimpleNamespace(content="ok", tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)], usage=None
    )


class TestSelectBackend:
    def test_routehub_is_chosen_when_installed(self, monkeypatch):
        monkeypatch.delenv("SWARMS_LLM_BACKEND", raising=False)
        _with_installed(monkeypatch, "routehub", "litellm")
        assert llm_backend.select_backend() == "routehub"

    def test_litellm_is_chosen_without_routehub(self, monkeypatch):
        monkeypatch.delenv("SWARMS_LLM_BACKEND", raising=False)
        _with_installed(monkeypatch, "litellm")
        assert llm_backend.select_backend() == "litellm"

    def test_env_var_forces_litellm(self, monkeypatch):
        monkeypatch.setenv("SWARMS_LLM_BACKEND", " LiteLLM ")
        _with_installed(monkeypatch, "routehub", "litellm")
        assert llm_backend.select_backend() == "litellm"

    def test_unknown_backend_is_rejected(self, monkeypatch):
        monkeypatch.setenv("SWARMS_LLM_BACKEND", "openai")
        with pytest.raises(ValueError, match="routehub, litellm"):
            llm_backend.select_backend()

    def test_missing_backend_names_the_install_command(
        self, monkeypatch
    ):
        monkeypatch.setenv("SWARMS_LLM_BACKEND", "routehub")
        _with_installed(monkeypatch, "litellm")
        with pytest.raises(ImportError, match=r"swarms\[fast\]"):
            llm_backend.select_backend()


def test_import_swarms_loads_neither_library():
    code = (
        "import sys, swarms; from swarms import Agent; "
        "print('litellm' in sys.modules, 'routehub' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split()[-2:] == ["False", "False"]


class TestOptionsPerCall:
    def _sent(self):
        llm = LiteLLM(
            model_name="gpt-5.4-mini", retries=5, drop_params=True
        )
        with patch(
            "swarms.utils.litellm_wrapper.completion",
            return_value=_response(),
        ) as fake:
            llm.run("hi")
        return fake.call_args.kwargs

    def test_routehub_gets_the_options_with_each_call(
        self, monkeypatch
    ):
        _use_backend(monkeypatch, "routehub")
        sent = self._sent()
        assert {name: sent[name] for name in BACKEND_OPTIONS} == {
            "set_verbose": False,
            "ssl_verify": False,
            "num_retries": 5,
            "drop_params": True,
        }

    def test_litellm_keeps_them_as_module_settings(self, monkeypatch):
        _use_backend(monkeypatch, "litellm")

        sent = self._sent()

        assert not set(BACKEND_OPTIONS) & set(sent)
        assert llm_backend.backend_module().num_retries == 5


class TestVisionCheck:
    def _llm(self, monkeypatch, backend, known_models, model):
        _use_backend(
            monkeypatch,
            backend,
            supports_vision=lambda model: False,
            model_list=known_models,
        )
        return LiteLLM(model_name=model)

    def test_routehub_sends_images_to_a_model_it_cannot_find(
        self, monkeypatch
    ):
        llm = self._llm(monkeypatch, "routehub", [], "lab/unlisted")
        llm.check_if_model_supports_vision(img="photo.png")

    def test_a_listed_text_only_model_is_still_refused(
        self, monkeypatch
    ):
        llm = self._llm(
            monkeypatch,
            "routehub",
            ["lab/text-only"],
            "lab/text-only",
        )
        with pytest.raises(ValueError, match="does not support"):
            llm.check_if_model_supports_vision(img="photo.png")

    def test_litellm_refuses_an_unlisted_model_as_before(
        self, monkeypatch
    ):
        llm = self._llm(monkeypatch, "litellm", [], "lab/unlisted-2")
        with pytest.raises(ValueError, match="does not support"):
            llm.check_if_model_supports_vision(img="photo.png")

    def test_a_no_is_asked_again(self, monkeypatch):
        answers = iter([False, True])
        _use_backend(
            monkeypatch,
            "litellm",
            supports_vision=lambda model: next(answers),
        )
        llm = LiteLLM(model_name="lab/catalog-was-down")

        with pytest.raises(ValueError):
            llm.check_if_model_supports_vision(img="photo.png")
        llm.check_if_model_supports_vision(img="photo.png")


class TestUnknownModelWarning:
    def _warnings(self, monkeypatch, known_models):
        _use_backend(
            monkeypatch,
            "routehub",
            model_list=known_models,
            get_max_tokens=lambda model: 1000,
        )
        agent = Agent(
            agent_name="Warn",
            model_name="lab/unlisted",
            max_tokens=100,
            context_length=1000,
            print_on=False,
            persistent_memory=False,
        )
        with patch("swarms.structs.agent.logger") as log:
            agent.reliability_check()
        return " ".join(
            call.args[0] for call in log.warning.call_args_list
        )

    def test_an_unreachable_catalog_is_not_reported(
        self, monkeypatch
    ):
        warnings = self._warnings(monkeypatch, [])
        assert "may not be supported" not in warnings

    def test_a_model_missing_from_the_catalog_is_reported(
        self, monkeypatch
    ):
        warnings = self._warnings(monkeypatch, ["lab/other"])
        assert "may not be supported" in warnings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
