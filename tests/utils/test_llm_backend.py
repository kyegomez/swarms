"""swarms makes every model call through RouteHub.

Offline: model calls are stubbed and RouteHub names are replaced in the
module's namespace so nothing is fetched.
"""

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
    "drop_params",
)


def _replace(monkeypatch, **names):
    """Replace names RouteHub would otherwise provide."""
    namespace = vars(llm_backend)
    for name, value in names.items():
        monkeypatch.setitem(namespace, name, value)


def _response():
    message = SimpleNamespace(content="ok", tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)], usage=None
    )


def test_import_swarms_does_not_load_the_openai_sdk():
    code = (
        "import sys, swarms; from swarms import Agent; "
        "print('openai' in sys.modules, 'litellm' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split()[-2:] == ["False", "False"]


def test_names_are_read_from_routehub():
    import routehub

    assert llm_backend.get_llm_provider is routehub.get_llm_provider
    assert llm_backend.RateLimitError is routehub.RateLimitError


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

    def test_routehub_gets_the_options_with_each_call(self):
        sent = self._sent()
        assert {name: sent[name] for name in BACKEND_OPTIONS} == {
            "set_verbose": False,
            "ssl_verify": False,
            "drop_params": True,
        }

    def test_retries_travel_with_every_request(self):
        assert self._sent()["num_retries"] == 5


class TestVisionCheck:
    def _llm(self, monkeypatch, known_models, model):
        _replace(
            monkeypatch,
            supports_vision=lambda model: False,
            model_list=known_models,
        )
        return LiteLLM(model_name=model)

    def test_images_go_to_a_model_the_catalogue_cannot_find(
        self, monkeypatch
    ):
        llm = self._llm(monkeypatch, [], "lab/unlisted")
        llm.check_if_model_supports_vision(img="photo.png")

    def test_a_listed_text_only_model_is_still_refused(
        self, monkeypatch
    ):
        llm = self._llm(
            monkeypatch, ["lab/text-only"], "lab/text-only"
        )
        with pytest.raises(ValueError, match="does not support"):
            llm.check_if_model_supports_vision(img="photo.png")

    def test_a_no_is_asked_again(self, monkeypatch):
        answers = iter([False, True])
        _replace(
            monkeypatch,
            supports_vision=lambda model: next(answers),
            model_list=["lab/catalog-was-down"],
        )
        llm = LiteLLM(model_name="lab/catalog-was-down")

        with pytest.raises(ValueError):
            llm.check_if_model_supports_vision(img="photo.png")
        llm.check_if_model_supports_vision(img="photo.png")


class TestUnknownModelWarning:
    def _warnings(self, monkeypatch, known_models):
        _replace(
            monkeypatch,
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
