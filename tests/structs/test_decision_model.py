import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from swarms import (
    DecisionModel,
    get_decision_model_prices,
    get_decision_models,
)
from swarms.env import load_swarms_env
from swarms.structs import decision_model
from swarms.telemetry.otel import init_config

CREDENTIAL_VARS = (
    "TYPESAFE_API_KEY",
    "CLOUDFLARE_AUTH_TOKEN",
    "CLOUDFLARE_ACCOUNT_ID",
)

BUILTIN_MODELS = [
    "jev-latest",
    "jev-preview",
    "jev-1.13.0",
    "clef",
    "clef-flash",
]

TYPESAFE_MODELS_URL = "https://api.typesafe.ai/v1/models"
CLOUDFLARE_MODELS_URL = (
    "https://api.cloudflare.com/client/v4/accounts/acct-1"
    "/ai/models/search?search=clef"
)

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Payments and refunds",
            "technical": "Bugs and outages",
        },
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Angry"],
    },
    "is_urgent": {
        "type": "noul",
        "instructions": "The message conveys urgency.",
    },
}


def _answer(question):
    if question["type"] == "choice":
        options = list(question["criteria"])
        return {
            "type": "choice",
            "choice": options[0],
            "probabilities": {
                option: float(i == 0)
                for i, option in enumerate(options)
            },
            "confidence": 0.8,
        }
    if question["type"] == "score":
        levels = question["criteria"]
        return {
            "type": "score",
            "score": 1.0,
            "legend": {
                str(i): level for i, level in enumerate(levels)
            },
            "probabilities": {
                str(i): float(i == 1) for i in range(len(levels))
            },
            "confidence": 0.9,
        }
    return {"type": "noul", "noul": 0.95}


class FakeAPI:
    """Stand-in decision model API that records every request."""

    def __init__(self):
        self.requests = []
        self.queue = []

    def __call__(self, request):
        self.requests.append(request)
        if self.queue:
            item = self.queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        body = json.loads(request.content)
        result = {
            "model": body["model"],
            "answers": {
                question_id: _answer(question)
                for question_id, question in body["questions"].items()
            },
            "usage": {"input_tokens": 12, "output_tokens": 3},
        }
        if request.url.host == "api.cloudflare.com":
            result = {
                "result": result,
                "success": True,
                "errors": [],
                "messages": [],
            }
        return httpx.Response(200, json=result)

    @property
    def body(self):
        return json.loads(self.requests[-1].content)


def _listing(url, typesafe=None, cloudflare=None):
    request = httpx.Request("GET", url)
    if "typesafe" in url:
        return httpx.Response(
            200, request=request, json={"models": typesafe or []}
        )
    return httpx.Response(
        200,
        request=request,
        json={"result": cloudflare or [], "success": True},
    )


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for name in CREDENTIAL_VARS:
        monkeypatch.delenv(name, raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(decision_model.httpx, "get", no_network)


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    delays = []

    async def async_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(
        decision_model, "time", SimpleNamespace(sleep=delays.append)
    )
    monkeypatch.setattr(
        decision_model, "asyncio", SimpleNamespace(sleep=async_sleep)
    )
    monkeypatch.setattr(
        decision_model,
        "random",
        SimpleNamespace(uniform=lambda low, high: 1.0),
    )
    return delays


@pytest.fixture
def api(monkeypatch):
    fake = FakeAPI()
    transport = httpx.MockTransport(fake)
    real_client = httpx.Client
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        decision_model.httpx,
        "Client",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )
    monkeypatch.setattr(
        decision_model.httpx,
        "AsyncClient",
        lambda **kwargs: real_async_client(
            transport=transport, **kwargs
        ),
    )
    return fake


@pytest.fixture
def typesafe_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")


@pytest.fixture
def cloudflare_env(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_AUTH_TOKEN", "cf-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-1")


def test_builtin_models_are_listed_without_credentials():
    assert get_decision_models() == BUILTIN_MODELS


def test_live_models_from_every_provider_are_merged(
    monkeypatch, typesafe_key, cloudflare_env
):
    seen = []

    def fake_get(url, headers=None, timeout=None):
        seen.append((url, headers["Authorization"]))
        return _listing(
            url,
            typesafe=[{"name": "jev-latest"}, {"name": "jev-2.0.0"}],
            cloudflare=[
                {"name": "@cf/cloudflare/clef"},
                {"name": "@cf/cloudflare/clef-2"},
            ],
        )

    monkeypatch.setattr(decision_model.httpx, "get", fake_get)

    assert get_decision_models() == BUILTIN_MODELS[:3] + [
        "jev-2.0.0",
        "clef",
        "clef-flash",
        "clef-2",
    ]
    assert seen == [
        (TYPESAFE_MODELS_URL, "Bearer ts-key"),
        (CLOUDFLARE_MODELS_URL, "Bearer cf-token"),
    ]


def test_listing_drops_models_outside_each_provider_family(
    monkeypatch, cloudflare_env
):
    monkeypatch.setattr(
        decision_model.httpx,
        "get",
        lambda url, **kwargs: _listing(
            url,
            cloudflare=[
                {"name": "@cf/meta/llama-3.1-8b-instruct"},
                {"name": "@cf/someone/clef-finetune"},
                {"description": "no name"},
                "not a model",
            ],
        ),
    )

    assert get_decision_models() == BUILTIN_MODELS


def test_listing_only_queries_providers_with_keys(
    monkeypatch, typesafe_key
):
    seen = []

    def fake_get(url, **kwargs):
        seen.append(url)
        return _listing(url)

    monkeypatch.setattr(decision_model.httpx, "get", fake_get)

    get_decision_models()

    assert seen == [TYPESAFE_MODELS_URL]


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("down"),
        httpx.ReadTimeout("slow"),
    ],
)
def test_listing_keeps_builtin_names_when_a_provider_is_unreachable(
    monkeypatch, typesafe_key, failure
):
    def fake_get(url, **kwargs):
        raise failure

    monkeypatch.setattr(decision_model.httpx, "get", fake_get)

    assert get_decision_models() == BUILTIN_MODELS


def test_listing_keeps_builtin_names_on_an_http_error(
    monkeypatch, typesafe_key
):
    monkeypatch.setattr(
        decision_model.httpx,
        "get",
        lambda url, **kwargs: httpx.Response(
            401, request=httpx.Request("GET", url)
        ),
    )

    assert get_decision_models() == BUILTIN_MODELS


def test_listing_keeps_builtin_names_on_a_malformed_body(
    monkeypatch, typesafe_key
):
    monkeypatch.setattr(
        decision_model.httpx,
        "get",
        lambda url, **kwargs: httpx.Response(
            200, request=httpx.Request("GET", url), json=["jev-x"]
        ),
    )

    assert get_decision_models() == BUILTIN_MODELS


def test_listing_skips_cloudflare_without_an_account_id(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_AUTH_TOKEN", "cf-token")

    assert get_decision_models() == BUILTIN_MODELS


def test_list_models_returns_every_provider(
    monkeypatch, typesafe_key, cloudflare_env
):
    monkeypatch.setattr(
        decision_model.httpx,
        "get",
        lambda url, **kwargs: _listing(
            url, cloudflare=[{"name": "@cf/cloudflare/clef-2"}]
        ),
    )

    assert DecisionModel().list_models() == get_decision_models()
    assert "clef-2" in DecisionModel().list_models()


def _cloudflare_price(input_price, output_price=None):
    value = [{"unit": "per M input tokens", "price": input_price}]
    if output_price is not None:
        value.append(
            {"unit": "per M output tokens", "price": output_price}
        )
    return [
        {"property_id": "context_window", "value": "65536"},
        {"property_id": "price", "value": value},
    ]


def test_builtin_prices_cover_every_builtin_model():
    prices = get_decision_model_prices()

    assert list(prices) == BUILTIN_MODELS
    assert prices["jev-latest"] == {"input": 0.042, "output": 0.0}
    assert prices["clef"] == {"input": 0.24, "output": 0.0}
    assert prices["clef-flash"] == {"input": 0.09, "output": 0.0}


def test_live_cloudflare_prices_override_the_builtin_ones(
    monkeypatch, typesafe_key, cloudflare_env
):
    monkeypatch.setattr(
        decision_model.httpx,
        "get",
        lambda url, **kwargs: _listing(
            url,
            typesafe=[{"name": "jev-2.0.0"}],
            cloudflare=[
                {
                    "name": "@cf/cloudflare/clef",
                    "properties": _cloudflare_price(0.3, 0.01),
                },
                {
                    "name": "@cf/cloudflare/clef-2",
                    "properties": _cloudflare_price(0.5),
                },
                {"name": "@cf/cloudflare/clef-flash"},
            ],
        ),
    )

    prices = get_decision_model_prices()

    assert prices["clef"] == {"input": 0.3, "output": 0.01}
    assert prices["clef-2"] == {"input": 0.5, "output": 0.0}
    assert prices["clef-flash"] == {"input": 0.09, "output": 0.0}
    assert prices["jev-latest"] == {"input": 0.042, "output": 0.0}
    assert "jev-2.0.0" not in prices


@pytest.mark.parametrize(
    "properties",
    [
        [
            {
                "property_id": "price",
                "value": [{"unit": "per M input tokens"}],
            }
        ],
        [
            {
                "property_id": "price",
                "value": [
                    {"unit": "per M input tokens", "price": "free"}
                ],
            }
        ],
        [{"property_id": "price", "value": "0.3"}],
        ["price"],
        "price",
    ],
)
def test_malformed_live_prices_keep_the_builtin_price(
    monkeypatch, cloudflare_env, properties
):
    monkeypatch.setattr(
        decision_model.httpx,
        "get",
        lambda url, **kwargs: _listing(
            url,
            cloudflare=[
                {
                    "name": "@cf/cloudflare/clef",
                    "properties": properties,
                }
            ],
        ),
    )

    assert get_decision_model_prices()["clef"] == {
        "input": 0.24,
        "output": 0.0,
    }


def test_get_price_returns_the_model_price(typesafe_key):
    assert DecisionModel().get_price() == {
        "input": 0.042,
        "output": 0.0,
    }


def test_typesafe_prices_need_no_network_call(
    monkeypatch, typesafe_key
):
    seen = []
    monkeypatch.setattr(
        decision_model.httpx,
        "get",
        lambda url, **kwargs: seen.append(url) or _listing(url),
    )

    DecisionModel().get_price()
    get_decision_model_prices()

    assert seen == []


def test_get_price_fetches_only_its_own_provider_once(
    monkeypatch, typesafe_key, cloudflare_env
):
    seen = []

    def fake_get(url, **kwargs):
        seen.append(url)
        return _listing(
            url,
            cloudflare=[
                {
                    "name": "@cf/cloudflare/clef-flash",
                    "properties": _cloudflare_price(0.1),
                }
            ],
        )

    monkeypatch.setattr(decision_model.httpx, "get", fake_get)
    model = DecisionModel(model_name="clef-flash")

    assert model.get_price() == {"input": 0.1, "output": 0.0}
    assert model.get_price() == {"input": 0.1, "output": 0.0}
    assert seen == [CLOUDFLARE_MODELS_URL]


def test_get_price_raises_for_an_unpriced_model(typesafe_key):
    with pytest.raises(ValueError, match="jev-9"):
        DecisionModel(model_name="jev-9").get_price()


def test_defaults_to_typesafe_jev_latest(typesafe_key):
    model = DecisionModel()

    assert model.model_name == "jev-latest"
    assert model.base_url == "https://api.typesafe.ai"
    assert model.endpoint == "/v1/systemone"
    assert model.api_key_env == "TYPESAFE_API_KEY"


@pytest.mark.parametrize(
    "model_name", ["clef", "clef-flash", "clef-2"]
)
def test_clef_models_route_to_cloudflare_workers_ai(
    cloudflare_env, model_name
):
    model = DecisionModel(model_name=model_name)

    assert model.base_url == (
        "https://api.cloudflare.com/client/v4/accounts/acct-1/ai/run"
    )
    assert model.endpoint == f"/@cf/cloudflare/{model_name}"
    assert model.api_key_env == "CLOUDFLARE_AUTH_TOKEN"


@pytest.mark.parametrize(
    "model_name", ["jev-1.13.0", "jev-2.0.0", "~typesafe/jev-latest"]
)
def test_other_names_route_to_typesafe(typesafe_key, model_name):
    model = DecisionModel(model_name=model_name)

    assert model.base_url == "https://api.typesafe.ai"
    assert model.endpoint == "/v1/systemone"


def test_explicit_settings_override_the_provider(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    model = DecisionModel(
        model_name="~typesafe/jev-latest",
        base_url="https://openrouter.ai/api/",
        endpoint="/v2/decide",
        api_key_env="OPENROUTER_API_KEY",
    )

    assert model.base_url == "https://openrouter.ai/api"
    assert model.endpoint == "/v2/decide"
    assert model._api_key == "or-key"


def test_explicit_base_url_does_not_need_a_cloudflare_account_id(
    monkeypatch,
):
    monkeypatch.setenv("CLOUDFLARE_AUTH_TOKEN", "cf-token")

    model = DecisionModel(
        model_name="clef",
        base_url="https://gateway.ai.cloudflare.com/v1/a/g/workers-ai",
    )

    assert model.endpoint == "/@cf/cloudflare/clef"


@pytest.mark.parametrize(
    "model_name, variable",
    [
        ("jev-latest", "TYPESAFE_API_KEY"),
        ("clef", "CLOUDFLARE_AUTH_TOKEN"),
    ],
)
def test_missing_api_key_names_the_variable(
    monkeypatch, model_name, variable
):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-1")

    with pytest.raises(ValueError, match=f"{variable}.*\\.env"):
        DecisionModel(model_name=model_name)


def test_blank_api_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "   \n")

    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        DecisionModel()


def test_missing_cloudflare_account_id_is_reported(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_AUTH_TOKEN", "cf-token")

    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID"):
        DecisionModel(model_name="clef")


def test_explicit_api_key_wins_and_is_stripped(typesafe_key):
    model = DecisionModel(api_key="  explicit-key\n")

    assert model._api_key == "explicit-key"


def test_credentials_are_read_from_a_dotenv_file(
    monkeypatch, tmp_path
):
    (tmp_path / ".env").write_text(
        "CLOUDFLARE_ACCOUNT_ID=acct-from-file\n"
        "CLOUDFLARE_AUTH_TOKEN=token-from-file\n"
    )
    monkeypatch.chdir(tmp_path)
    # setenv first so monkeypatch removes the loaded values afterwards.
    for name in CREDENTIAL_VARS:
        monkeypatch.setenv(name, "stale")

    assert load_swarms_env(override=True)
    model = DecisionModel(model_name="clef-flash")

    assert "acct-from-file" in model.base_url
    assert model._api_key == "token-from-file"


def test_secrets_stay_out_of_telemetry_config(typesafe_key):
    model = DecisionModel(headers={"X-Secret": "header-secret"})

    config = init_config(model)

    assert "ts-key" not in config
    assert "header-secret" not in config
    assert "jev-latest" in config


def test_run_posts_state_and_questions(api, typesafe_key):
    model = DecisionModel()

    result = model.run({"message": "Help!"}, QUESTIONS)

    request = api.requests[-1]
    assert request.method == "POST"
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["authorization"] == "Bearer ts-key"
    assert request.headers["content-type"] == "application/json"
    assert api.body == {
        "model": "jev-latest",
        "state": {"message": "Help!"},
        "questions": QUESTIONS,
    }
    assert set(result["answers"]) == set(QUESTIONS)
    assert result["answers"]["department"]["choice"] == "billing"
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 3}


@pytest.mark.parametrize(
    "state",
    [
        "plain text",
        {"ticket": {"id": 1, "body": None}},
        ["first message", "second message"],
    ],
)
def test_state_is_sent_unchanged(api, typesafe_key, state):
    DecisionModel().run(state, QUESTIONS)

    assert api.body["state"] == state


def test_extra_body_and_headers_are_sent(api, typesafe_key):
    model = DecisionModel(
        extra_body={"beam_width": 4}, headers={"X-Trace": "abc"}
    )

    model.run("text", QUESTIONS)

    assert api.body["beam_width"] == 4
    assert api.requests[-1].headers["x-trace"] == "abc"


@pytest.mark.parametrize("model_name", ["clef", "clef-flash"])
def test_cloudflare_run_unwraps_the_result_envelope(
    api, cloudflare_env, model_name
):
    result = DecisionModel(model_name=model_name).run(
        "Checkout is down", QUESTIONS
    )

    request = api.requests[-1]
    assert str(request.url) == (
        "https://api.cloudflare.com/client/v4/accounts/acct-1"
        f"/ai/run/@cf/cloudflare/{model_name}"
    )
    assert request.headers["authorization"] == "Bearer cf-token"
    assert api.body["model"] == model_name
    assert "result" not in result
    assert result["answers"]["is_urgent"]["noul"] == 0.95


def test_choice_helper_returns_the_answer(api, typesafe_key):
    answer = DecisionModel().choice(
        "text", "Which team?", {"billing": None, "technical": None}
    )

    assert answer["choice"] == "billing"
    assert api.body["questions"] == {
        "answer": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": None, "technical": None},
        }
    }


def test_score_helper_returns_the_answer(api, typesafe_key):
    answer = DecisionModel().score(
        "text", "How bad?", ["low", "high"]
    )

    assert answer["score"] == 1.0
    assert api.body["questions"]["answer"]["criteria"] == [
        "low",
        "high",
    ]


def test_noul_helper_returns_a_float(api, typesafe_key):
    assert DecisionModel().noul("text", "Urgent?") == 0.95
    assert "criteria" not in api.body["questions"]["answer"]


def test_noul_helper_sends_criteria(api, typesafe_key):
    criteria = {"true": "Time-sensitive", "false": "Can wait"}

    DecisionModel().noul("text", "Urgent?", criteria=criteria)

    assert api.body["questions"]["answer"]["criteria"] == criteria


def test_structured_instructions_are_sent_as_is(api, typesafe_key):
    instructions = {
        "candidate": {"name": "John Smith"},
        "question": "Is the resume for `candidate`?",
    }

    DecisionModel().noul("resume text", instructions)

    assert api.body["questions"]["answer"]["instructions"] == (
        instructions
    )


def test_http_client_is_reused_until_closed(api, typesafe_key):
    model = DecisionModel()
    model.noul("a", "?")
    client = model._client
    model.noul("b", "?")

    assert model._client is client

    model.close()
    model.close()

    assert model._client is None
    assert client.is_closed
    assert model.noul("c", "?") == 0.95


@pytest.mark.parametrize(
    "questions, message",
    [
        ({}, "at least one question"),
        ({"q": {"instructions": "?"}}, "has type None"),
        (
            {"q": {"type": "rank", "instructions": "?"}},
            "has type 'rank'",
        ),
        (
            {"q": {"type": "choice", "instructions": "?"}},
            "Choice 'q' needs criteria",
        ),
        (
            {
                "q": {
                    "type": "choice",
                    "instructions": "?",
                    "criteria": {},
                }
            },
            "Choice 'q' needs criteria",
        ),
        (
            {
                "q": {
                    "type": "choice",
                    "instructions": "?",
                    "criteria": ["a", "b"],
                }
            },
            "Choice 'q' needs criteria",
        ),
        (
            {
                "q": {
                    "type": "score",
                    "instructions": "?",
                    "criteria": {"0": "low", "1": "high"},
                }
            },
            "Score 'q' needs criteria",
        ),
        (
            {
                "q": {
                    "type": "score",
                    "instructions": "?",
                    "criteria": ["x"],
                }
            },
            "Score 'q' needs criteria",
        ),
    ],
)
def test_invalid_questions_raise_before_sending(
    api, typesafe_key, questions, message
):
    with pytest.raises(ValueError, match=message):
        DecisionModel().run("text", questions)

    assert api.requests == []


def test_none_state_raises_before_sending(api, typesafe_key):
    with pytest.raises(ValueError, match="state cannot be None"):
        DecisionModel().run(None, QUESTIONS)

    assert api.requests == []


def test_score_criteria_may_be_a_tuple(api, typesafe_key):
    DecisionModel().score("text", "Rate", ("low", "high"))

    assert api.body["questions"]["answer"]["criteria"] == [
        "low",
        "high",
    ]


def test_missing_answer_raises(api, typesafe_key):
    api.queue.append(
        httpx.Response(200, json={"model": "m", "answers": {}})
    )

    with pytest.raises(RuntimeError, match="No answer.*'is_urgent'"):
        DecisionModel().run(
            "text", {"is_urgent": QUESTIONS["is_urgent"]}
        )


def test_answer_of_the_wrong_type_raises(api, typesafe_key):
    api.queue.append(
        httpx.Response(
            200,
            json={
                "model": "m",
                "answers": {"is_urgent": {"type": "choice"}},
            },
        )
    )

    with pytest.raises(RuntimeError, match="noul but the answer"):
        DecisionModel().run(
            "text", {"is_urgent": QUESTIONS["is_urgent"]}
        )


def test_answer_without_a_type_is_accepted(api, typesafe_key):
    api.queue.append(
        httpx.Response(
            200,
            json={
                "model": "m",
                "answers": {"is_urgent": {"noul": 0.4}},
            },
        )
    )

    result = DecisionModel().run(
        "text", {"is_urgent": QUESTIONS["is_urgent"]}
    )

    assert result["answers"]["is_urgent"]["noul"] == 0.4


@pytest.mark.parametrize(
    "status", sorted(decision_model.RETRYABLE_STATUSES)
)
def test_retryable_statuses_are_retried(api, typesafe_key, status):
    api.queue.append(httpx.Response(status, json={"error": "busy"}))

    assert DecisionModel().noul("text", "?") == 0.95
    assert len(api.requests) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_raise_without_retrying(
    api, typesafe_key, status
):
    api.queue.append(
        httpx.Response(
            status, json={"detail": "criteria: field required"}
        )
    )

    with pytest.raises(httpx.HTTPStatusError) as error:
        DecisionModel().noul("text", "?")

    assert error.value.response.status_code == status
    assert "criteria: field required" in str(error.value)
    assert len(api.requests) == 1


def test_retries_stop_after_max_retries(api, typesafe_key):
    api.queue.extend(
        [httpx.Response(429, json={"error": "slow down"})] * 3
    )

    with pytest.raises(httpx.HTTPStatusError, match="429"):
        DecisionModel(max_retries=2).noul("text", "?")

    assert len(api.requests) == 3


def test_zero_max_retries_disables_retrying(api, typesafe_key):
    api.queue.append(httpx.Response(503))

    with pytest.raises(httpx.HTTPStatusError):
        DecisionModel(max_retries=0).noul("text", "?")

    assert len(api.requests) == 1


def test_backoff_doubles_each_attempt(api, typesafe_key, sleeps):
    api.queue.extend([httpx.Response(529)] * 3)

    DecisionModel(max_retries=3).noul("text", "?")

    assert sleeps == [0.5, 1.0, 2.0]


def test_backoff_is_capped(api, typesafe_key, sleeps):
    api.queue.extend([httpx.Response(503)] * 6)

    DecisionModel(max_retries=6).noul("text", "?")

    assert sleeps == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"retry-after": "3"}, 3.0),
        ({"retry-after-ms": "250"}, 0.25),
        ({"retry-after": "3", "retry-after-ms": "250"}, 0.25),
        ({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, 0.5),
        ({"retry-after": "600"}, 60.0),
        ({"retry-after": "-5"}, 0.0),
    ],
)
def test_retry_after_headers_set_the_delay(
    api, typesafe_key, sleeps, headers, expected
):
    api.queue.append(httpx.Response(429, headers=headers))

    DecisionModel().noul("text", "?")

    assert sleeps == [expected]


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("slow"),
        httpx.RemoteProtocolError("dropped"),
    ],
)
def test_transport_errors_are_retried(api, typesafe_key, error):
    api.queue.append(error)

    assert DecisionModel().noul("text", "?") == 0.95
    assert len(api.requests) == 2


def test_transport_errors_raise_after_max_retries(api, typesafe_key):
    api.queue.extend([httpx.ConnectError("refused")] * 3)

    with pytest.raises(httpx.ConnectError):
        DecisionModel(max_retries=2).noul("text", "?")

    assert len(api.requests) == 3


async def test_arun_matches_run(api, typesafe_key):
    model = DecisionModel()

    result = await model.arun({"message": "Help!"}, QUESTIONS)

    assert api.body["state"] == {"message": "Help!"}
    assert result == model.run({"message": "Help!"}, QUESTIONS)


async def test_arun_unwraps_cloudflare_results(api, cloudflare_env):
    result = await DecisionModel(model_name="clef").arun(
        "text", QUESTIONS
    )

    assert result["answers"]["frustration"]["score"] == 1.0
    assert "/@cf/cloudflare/clef" in str(api.requests[-1].url)


async def test_arun_retries_with_async_sleep(
    api, typesafe_key, sleeps
):
    api.queue.extend(
        [httpx.Response(529), httpx.ConnectError("refused")]
    )

    result = await DecisionModel().arun("text", QUESTIONS)

    assert result["answers"]["is_urgent"]["noul"] == 0.95
    assert sleeps == [0.5, 1.0]
    assert len(api.requests) == 3


async def test_arun_raises_client_errors(api, typesafe_key):
    api.queue.append(httpx.Response(422, json={"detail": "bad"}))

    with pytest.raises(httpx.HTTPStatusError, match="bad"):
        await DecisionModel().arun("text", QUESTIONS)


async def test_arun_validates_before_sending(api, typesafe_key):
    with pytest.raises(ValueError):
        await DecisionModel().arun("text", {})

    assert api.requests == []


async def test_arun_supports_concurrent_calls(api, typesafe_key):
    model = DecisionModel()

    results = await asyncio.gather(
        *[model.arun(f"ticket {i}", QUESTIONS) for i in range(5)]
    )

    assert len(results) == 5
    sent = sorted(
        json.loads(r.content)["state"] for r in api.requests
    )
    assert sent == [f"ticket {i}" for i in range(5)]


def test_usage_starts_at_zero(typesafe_key):
    assert DecisionModel().usage == {
        "input_tokens": 0,
        "output_tokens": 0,
    }


async def test_usage_sums_every_request(api, typesafe_key):
    model = DecisionModel()

    model.run("text", QUESTIONS)
    model.noul("text", "Is this urgent?")
    await asyncio.gather(
        *[model.arun("text", QUESTIONS) for _ in range(3)]
    )

    assert model.usage == {"input_tokens": 60, "output_tokens": 15}


def test_usage_reads_cloudflare_results(api, cloudflare_env):
    model = DecisionModel(model_name="clef")

    model.run("text", QUESTIONS)

    assert model.usage == {"input_tokens": 12, "output_tokens": 3}


def test_usage_returns_a_copy(api, typesafe_key):
    model = DecisionModel()

    model.usage["input_tokens"] = 99

    assert model.usage["input_tokens"] == 0


def test_responses_without_usage_are_counted_as_zero(
    api, typesafe_key
):
    api.queue.append(
        httpx.Response(
            200,
            json={
                "model": "jev-latest",
                "answers": {"q": {"type": "noul", "noul": 0.5}},
            },
        )
    )
    model = DecisionModel()

    model.run("text", {"q": {"type": "noul", "instructions": "?"}})

    assert model.usage == {"input_tokens": 0, "output_tokens": 0}


def test_failed_requests_add_no_usage(api, typesafe_key):
    api.queue.append(httpx.Response(422, json={"detail": "bad"}))
    model = DecisionModel()

    with pytest.raises(httpx.HTTPStatusError):
        model.run("text", QUESTIONS)

    assert model.usage == {"input_tokens": 0, "output_tokens": 0}


def test_calculate_cost_prices_one_response(api, typesafe_key):
    model = DecisionModel()

    response = model.run("text", QUESTIONS)
    cost = model.calculate_cost(response["usage"])

    assert cost == {
        "input_tokens": 12,
        "output_tokens": 3,
        "input_cost": pytest.approx(12 * 0.042 / 1_000_000),
        "output_cost": 0.0,
        "total_cost": pytest.approx(12 * 0.042 / 1_000_000),
    }


def test_calculate_cost_defaults_to_the_summed_usage(
    api, cloudflare_env
):
    model = DecisionModel(model_name="clef-flash")

    model.run("text", QUESTIONS)
    model.run("text", QUESTIONS)

    cost = model.calculate_cost()
    assert cost["input_tokens"] == 24
    assert cost["total_cost"] == pytest.approx(24 * 0.09 / 1_000_000)


def test_calculate_cost_bills_output_tokens_when_priced(
    monkeypatch, cloudflare_env
):
    monkeypatch.setattr(
        decision_model.httpx,
        "get",
        lambda url, **kwargs: _listing(
            url,
            cloudflare=[
                {
                    "name": "@cf/cloudflare/clef",
                    "properties": _cloudflare_price(1.0, 2.0),
                }
            ],
        ),
    )
    model = DecisionModel(model_name="clef")

    cost = model.calculate_cost(
        {"input_tokens": 1_000_000, "output_tokens": 500_000}
    )

    assert cost["input_cost"] == pytest.approx(1.0)
    assert cost["output_cost"] == pytest.approx(1.0)
    assert cost["total_cost"] == pytest.approx(2.0)


class PlainProvider(DecisionModel):
    """Decision model for a provider with its own wire format."""

    question_types = DecisionModel.question_types + ("rank",)

    def build_headers(self):
        return {"X-Api-Key": self._api_key}

    def build_payload(self, state, questions):
        return {"input": state, "schema": questions}

    def parse_response(self, data, questions):
        return {"answers": data["output"]}


def _plain_handler(request):
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "output": {
                question_id: {"type": question["type"], "value": 1}
                for question_id, question in body["schema"].items()
            }
        },
    )


@pytest.fixture
def plain_api(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return _plain_handler(request)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        decision_model.httpx,
        "Client",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )
    monkeypatch.setattr(
        decision_model.httpx,
        "AsyncClient",
        lambda **kwargs: real_async_client(
            transport=transport, **kwargs
        ),
    )
    return requests


def test_subclass_hooks_define_a_new_wire_format(plain_api):
    model = PlainProvider(
        model_name="plain-1",
        api_key="plain-key",
        base_url="https://plain.example",
        endpoint="/decide",
    )

    result = model.run(
        "text", {"order": {"type": "rank", "items": ["a", "b"]}}
    )

    request = plain_api[-1]
    assert str(request.url) == "https://plain.example/decide"
    assert request.headers["x-api-key"] == "plain-key"
    assert "authorization" not in request.headers
    assert json.loads(request.content) == {
        "input": "text",
        "schema": {"order": {"type": "rank", "items": ["a", "b"]}},
    }
    assert result == {
        "answers": {"order": {"type": "rank", "value": 1}}
    }


async def test_subclass_hooks_apply_to_arun(plain_api):
    model = PlainProvider(
        api_key="plain-key", base_url="https://plain.example"
    )

    result = await model.arun("text", {"q": {"type": "noul"}})

    assert result == {"answers": {"q": {"type": "noul", "value": 1}}}


def test_subclass_may_parse_responses_into_any_shape(
    api, typesafe_key
):
    class AnswersOnly(DecisionModel):
        def parse_response(self, data, questions):
            return list(data["answers"].values())

    model = AnswersOnly()

    assert model.run("text", QUESTIONS)[2] == {
        "type": "noul",
        "noul": 0.95,
    }
    assert model.usage == {"input_tokens": 0, "output_tokens": 0}


def test_subclass_can_extend_question_types(api, typesafe_key):
    class RankingModel(DecisionModel):
        question_types = DecisionModel.question_types + ("rank",)

    api.queue.append(
        httpx.Response(
            200,
            json={"model": "m", "answers": {"q": {"type": "rank"}}},
        )
    )

    result = RankingModel().run("text", {"q": {"type": "rank"}})

    assert result["answers"]["q"]["type"] == "rank"
