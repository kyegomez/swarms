import asyncio
import json
import os
import random
import re
import threading
import time
import weakref
from typing import Any, Dict, List, Optional, Union

import httpx
from loguru import logger

from swarms.telemetry.otel import capture_init, trace_run

RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})

State = Union[str, Dict[str, Any], List[Any]]
Questions = Dict[str, Dict[str, Any]]

# {UPPER_CASE} placeholders are filled from the environment, {model_name} from the model.
# Prices are US dollars per million tokens; lists_prices providers also report them live.
DECISION_MODEL_PROVIDERS = {
    "typesafe": {
        "prefix": "jev",
        "models": ["jev-latest", "jev-preview", "jev-1.13.0"],
        "prices": {
            "jev-latest": {"input": 0.042, "output": 0.0},
            "jev-preview": {"input": 0.042, "output": 0.0},
            "jev-1.13.0": {"input": 0.042, "output": 0.0},
        },
        "base_url": "https://api.typesafe.ai",
        "endpoint": "/v1/systemone",
        "api_key_env": "TYPESAFE_API_KEY",
        "models_url": "https://api.typesafe.ai/v1/models",
    },
    "cloudflare": {
        "prefix": "clef",
        "models": ["clef", "clef-flash"],
        "prices": {
            "clef": {"input": 0.24, "output": 0.0},
            "clef-flash": {"input": 0.09, "output": 0.0},
        },
        "lists_prices": True,
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run",
        "endpoint": "/@cf/cloudflare/{model_name}",
        "api_key_env": "CLOUDFLARE_AUTH_TOKEN",
        "models_url": "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/models/search?search=clef",
    },
    "openai": {
        "prefix": "gpt-6-luna",
        "models": ["gpt-6-luna"],
        "prices": {
            "gpt-6-luna": {"input": 0.10, "output": 0.0},
        },
        "base_url": "https://api.openai.com/v1",
        "endpoint": "/decisions",
        "api_key_env": "OPENAI_API_KEY",
        "models_url": "https://api.openai.com/v1/models",
    },
}


def get_decision_models() -> List[str]:
    """
    List the decision models from every provider.

    Returns:
        Model names, fetched live from each provider whose credentials are set and merged with the built-in names.
    """
    models = []
    for name, provider in DECISION_MODEL_PROVIDERS.items():
        models.extend(provider["models"])
        models.extend(_fetch_provider_models(name, provider))
    return list(dict.fromkeys(models))


def get_decision_model_prices() -> Dict[str, Dict[str, float]]:
    """
    List the price of every decision model.

    Returns:
        Model names mapped to input and output prices in US dollars per million tokens, fetched live from each provider that reports them and merged over the built-in prices.
    """
    prices = {}
    for name, provider in DECISION_MODEL_PROVIDERS.items():
        prices.update(_provider_prices(name, provider))
    return prices


def _provider_prices(
    name: str, provider: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    prices = dict(provider["prices"])
    if not provider.get("lists_prices"):
        return prices
    for model, entry in _fetch_provider_models(
        name, provider
    ).items():
        price = _listed_price(entry)
        if price is not None:
            prices[model] = price
    return prices


def _listed_price(
    entry: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    # Cloudflare lists prices as a property, e.g. {"unit": "per M input tokens", "price": 0.24}.
    for prop in entry.get("properties") or []:
        if not (
            isinstance(prop, dict)
            and prop.get("property_id") == "price"
        ):
            continue
        price = {"input": 0.0, "output": 0.0}
        try:
            for item in prop["value"]:
                for side in price:
                    if side in item["unit"]:
                        price[side] = float(item["price"])
        except (KeyError, TypeError, ValueError):
            return None
        return price
    return None


def _fetch_provider_models(
    name: str, provider: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    api_key = (os.getenv(provider["api_key_env"]) or "").strip()
    if not api_key:
        return {}

    try:
        response = httpx.get(
            _fill_from_env(provider["models_url"]),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
        # TypeSafe lists under models; Cloudflare under result, as @cf/cloudflare/<name>; OpenAI under data, by id.
        entries = (
            data.get("models")
            or data.get("result")
            or data.get("data")
            or []
        )
        models = {
            str(
                entry.get("name") or entry.get("id") or ""
            ).removeprefix("@cf/cloudflare/"): entry
            for entry in entries
            if isinstance(entry, dict)
        }
    except Exception as e:
        logger.warning(f"Could not fetch {name} decision models: {e}")
        return {}

    return {
        n: entry
        for n, entry in models.items()
        if n.startswith(provider["prefix"])
    }


def _provider_for(model_name: str) -> str:
    for name, provider in DECISION_MODEL_PROVIDERS.items():
        if model_name.startswith(provider["prefix"]):
            return name
    # Anything else, such as the gateway id ~typesafe/jev-latest, goes to TypeSafe.
    return "typesafe"


def _fill_from_env(template: str) -> str:
    names = re.findall(r"{([A-Z0-9_]+)}", template)
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise ValueError(
            f"Set {', '.join(missing)} in your environment or .env file."
        )
    return template.format(
        **{name: os.environ[name] for name in names}
    )


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _openai_input(state: State) -> Union[str, List[Any]]:
    # A list of user messages passes through so it can carry input_image parts.
    if (
        isinstance(state, list)
        and state
        and all(
            isinstance(message, dict)
            and message.get("role") == "user"
            for message in state
        )
    ):
        return state
    return _as_text(state)


def _openai_question(
    question_id: str, question: Dict[str, Any]
) -> Dict[str, Any]:
    question_type = question["type"]
    criteria = question.get("criteria")
    converted = {
        "type": (
            "predicate" if question_type == "noul" else question_type
        ),
        "name": question_id,
        "instructions": _as_text(question.get("instructions", "")),
    }
    if question_type == "choice":
        converted["choices"] = [
            (
                {"value": value}
                if description is None
                else {
                    "value": value,
                    "description": _as_text(description),
                }
            )
            for value, description in criteria.items()
        ]
    elif question_type == "score":
        converted["levels"] = [
            {"label": _as_text(level)} for level in criteria
        ]
    elif criteria is not None:
        # Predicates have no criteria field, so noul criteria join the instructions.
        converted[
            "instructions"
        ] += f"\nCriteria: {_as_text(criteria)}"
    return converted


def _openai_answers(
    answers: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    converted = {}
    for answer in answers:
        name = answer.get("name")
        answer_type = answer.get("type")
        if answer_type == "refusal":
            raise RuntimeError(
                f"OpenAI refused to answer question {name!r}."
            )
        if answer_type == "predicate":
            converted[name] = {
                "type": "noul",
                "noul": answer["probability"],
            }
        elif answer_type == "choice":
            converted[name] = {
                "type": "choice",
                "choice": answer["choice"],
                "probabilities": {
                    p["value"]: p["probability"]
                    for p in answer["probabilities"]
                },
                "confidence": answer["confidence"],
            }
        elif answer_type == "score":
            converted[name] = {
                "type": "score",
                "score": answer["score"],
                "legend": {
                    str(p["value"]): p["label"]
                    for p in answer["probabilities"]
                },
                "probabilities": {
                    str(p["value"]): p["probability"]
                    for p in answer["probabilities"]
                },
                "confidence": answer["confidence"],
            }
        else:
            converted[name] = answer
    return converted


class DecisionModel:
    """
    Client for decision models that answer typed questions about a state.

    Args:
        model_name: Model that answers the questions. Its name picks the provider's URL, endpoint and API key variable.
        api_key: API key, read from api_key_env when omitted.
        api_key_env: Environment variable holding the API key. Defaults to the model's provider.
        base_url: Root URL of the provider's API. Defaults to the model's provider.
        endpoint: Path of the evaluation endpoint. Defaults to the model's provider.
        timeout: Seconds to wait for each HTTP request.
        max_retries: Retries after the first attempt on rate limits, overloads and connection errors.
        headers: Extra HTTP headers sent with every request.
        extra_body: Extra fields merged into every request body.
    """

    question_types = ("choice", "score", "noul")

    def __init__(
        self,
        model_name: str = "jev-latest",
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        base_url: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        headers: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        self._provider_name = _provider_for(model_name)
        provider = DECISION_MODEL_PROVIDERS[self._provider_name]

        self.model_name = model_name
        self.api_key_env = api_key_env or provider["api_key_env"]
        self.base_url = (
            base_url or _fill_from_env(provider["base_url"])
        ).rstrip("/")
        self.endpoint = endpoint or provider["endpoint"].format(
            model_name=model_name
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = extra_body or {}

        # Private names keep secrets out of capture_init, which records public init attributes.
        self._api_key = (
            api_key or os.getenv(self.api_key_env) or ""
        ).strip()
        self._headers = headers or {}
        self._client: Optional[httpx.Client] = None
        self._async_clients: weakref.WeakKeyDictionary = (
            weakref.WeakKeyDictionary()
        )
        self._price: Optional[Dict[str, float]] = None
        self._usage = {"input_tokens": 0, "output_tokens": 0}
        self._usage_lock = threading.Lock()

        if not self._api_key:
            raise ValueError(
                f"No API key found. Pass api_key or set "
                f"{self.api_key_env} in your environment or .env file."
            )

        capture_init(self)

    @trace_run("DecisionModel.run", input_params=("state",))
    def run(
        self, state: State, questions: Questions
    ) -> Dict[str, Any]:
        """
        Ask every question about the state in one request.

        Args:
            state: Text, JSON object or list the questions are asked about. OpenAI models also take a list of user messages with input_image parts.
            questions: Question dictionaries keyed by an id of your choosing.

        Returns:
            The response with model, answers keyed by question id, and usage.
        """
        payload = self.build_payload(state, questions)
        data = self._request("POST", self.endpoint, payload)
        result = self.parse_response(data, questions)
        self._add_usage(result)
        return result

    async def arun(
        self, state: State, questions: Questions
    ) -> Dict[str, Any]:
        """
        Ask every question about the state in one asynchronous request.

        Args:
            state: Text, JSON object or list the questions are asked about. OpenAI models also take a list of user messages with input_image parts.
            questions: Question dictionaries keyed by an id of your choosing.

        Returns:
            The response with model, answers keyed by question id, and usage.
        """
        payload = self.build_payload(state, questions)
        data = await self._arequest("POST", self.endpoint, payload)
        result = self.parse_response(data, questions)
        self._add_usage(result)
        return result

    def choice(
        self,
        state: State,
        instructions: Any,
        criteria: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Pick one option from a defined set.

        Args:
            state: Content to evaluate.
            instructions: What the model should decide.
            criteria: Option names mapped to a description, or None for no description.

        Returns:
            The answer with choice, probabilities and confidence.
        """
        return self._ask_one(
            state,
            {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            },
        )

    def score(
        self,
        state: State,
        instructions: Any,
        criteria: List[Any],
    ) -> Dict[str, Any]:
        """
        Rate the state against ordered levels.

        Args:
            state: Content to evaluate.
            instructions: What the model should rate.
            criteria: Level descriptions ordered from lowest to highest.

        Returns:
            The answer with score, legend, probabilities and confidence.
        """
        return self._ask_one(
            state,
            {
                "type": "score",
                "instructions": instructions,
                "criteria": criteria,
            },
        )

    def noul(
        self,
        state: State,
        instructions: Any,
        criteria: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Get the probability that a yes/no statement is true.

        Args:
            state: Content to evaluate.
            instructions: The yes/no question or statement.
            criteria: Optional descriptions under the keys true and false.

        Returns:
            Probability from 0 (no) to 1 (yes).
        """
        question = {"type": "noul", "instructions": instructions}
        if criteria is not None:
            question["criteria"] = criteria
        return self._ask_one(state, question)["noul"]

    def list_models(self) -> List[str]:
        """
        List the decision models from every provider.

        Returns:
            The model names.
        """
        return get_decision_models()

    @property
    def usage(self) -> Dict[str, int]:
        """
        Token usage reported by the provider, summed over every request this model has made.

        Returns:
            The input and output token counts.
        """
        with self._usage_lock:
            return dict(self._usage)

    def get_price(self) -> Dict[str, float]:
        """
        Get the price of this model from its provider.

        Returns:
            Input and output prices in US dollars per million tokens.
        """
        if self._price is None:
            price = _provider_prices(
                self._provider_name,
                DECISION_MODEL_PROVIDERS[self._provider_name],
            ).get(self.model_name)
            if price is None:
                raise ValueError(
                    f"No price is known for {self.model_name!r}."
                )
            self._price = price
        return dict(self._price)

    def calculate_cost(
        self, usage: Optional[Dict[str, int]] = None
    ) -> Dict[str, float]:
        """
        Price the input and output tokens reported by the provider.

        Args:
            usage: Token counts from one response. Defaults to the usage summed over every request.

        Returns:
            The input and output tokens, their cost and the total cost in US dollars.
        """
        usage = self.usage if usage is None else usage
        price = self.get_price()
        input_tokens = usage.get("input_tokens") or 0
        output_tokens = usage.get("output_tokens") or 0
        input_cost = input_tokens * price["input"] / 1_000_000
        output_cost = output_tokens * price["output"] / 1_000_000
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        }

    def build_headers(self) -> Dict[str, str]:
        """
        Build the HTTP headers for a request.

        Returns:
            The headers to send.
        """
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._headers,
        }

    def build_payload(
        self, state: State, questions: Questions
    ) -> Dict[str, Any]:
        """
        Validate the questions and build the request body.

        Args:
            state: Content the questions are asked about.
            questions: Question dictionaries keyed by id.

        Returns:
            The JSON request body.
        """
        if state is None:
            raise ValueError("state cannot be None.")
        if not questions:
            raise ValueError("Ask at least one question.")

        for question_id, question in questions.items():
            question_type = question.get("type")
            criteria = question.get("criteria")
            if question_type not in self.question_types:
                raise ValueError(
                    f"Question {question_id!r} has type {question_type!r}; "
                    f"expected one of {self.question_types}."
                )
            if question_type == "choice" and not (
                isinstance(criteria, dict) and criteria
            ):
                raise ValueError(
                    f"Choice {question_id!r} needs criteria mapping each option to a description."
                )
            if question_type == "score" and not (
                isinstance(criteria, (list, tuple))
                and len(criteria) >= 2
            ):
                raise ValueError(
                    f"Score {question_id!r} needs criteria as a list of at least two levels."
                )

        if self._provider_name == "openai":
            return {
                "model": self.model_name,
                "input": _openai_input(state),
                "questions": [
                    _openai_question(question_id, question)
                    for question_id, question in questions.items()
                ],
                **self.extra_body,
            }
        return {
            "model": self.model_name,
            "state": state,
            "questions": questions,
            **self.extra_body,
        }

    def parse_response(
        self, data: Dict[str, Any], questions: Questions
    ) -> Dict[str, Any]:
        """
        Key the answers by question id and check that each has its question's type.

        Args:
            data: Decoded JSON response.
            questions: Questions sent in the request.

        Returns:
            The response with model, answers and usage.
        """
        # Cloudflare Workers AI wraps the body in a result envelope.
        if "answers" not in data and isinstance(
            data.get("result"), dict
        ):
            data = data["result"]
        if self._provider_name == "openai":
            data = {
                **data,
                "answers": _openai_answers(data.get("answers") or []),
            }

        answers = data.get("answers") or {}
        for question_id, question in questions.items():
            answer = answers.get(question_id)
            if answer is None:
                raise RuntimeError(
                    f"No answer returned for question {question_id!r}."
                )
            if (
                answer.get("type", question["type"])
                != question["type"]
            ):
                raise RuntimeError(
                    f"Question {question_id!r} is a {question['type']} "
                    f"but the answer is a {answer['type']}."
                )
        return data

    def close(self) -> None:
        """Close the pooled HTTP connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        """Close the pooled async HTTP client of the running event loop."""
        client = self._async_clients.pop(
            asyncio.get_running_loop(), None
        )
        if client is not None:
            await client.aclose()

    def _ask_one(
        self, state: State, question: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.run(state, {"answer": question})["answers"][
            "answer"
        ]

    def _add_usage(self, result: Any) -> None:
        # A subclass's parse_response may return any shape; only a dict usage block counts.
        usage = (
            result.get("usage") if isinstance(result, dict) else None
        )
        if not isinstance(usage, dict):
            return
        with self._usage_lock:
            for key in self._usage:
                self._usage[key] += usage.get(key) or 0

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)

        for attempt in range(self.max_retries + 1):
            response = None
            try:
                response = self._client.request(
                    method,
                    self.base_url + path,
                    json=payload,
                    headers=self.build_headers(),
                )
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
            if not self._should_retry(attempt, response):
                return self._decode(response)
            time.sleep(self._retry_delay(attempt, response))

    async def _arequest(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        client = self._async_clients.get(loop)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(timeout=self.timeout)
            self._async_clients[loop] = client

        for attempt in range(self.max_retries + 1):
            response = None
            try:
                response = await client.request(
                    method,
                    self.base_url + path,
                    json=payload,
                    headers=self.build_headers(),
                )
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
            if not self._should_retry(attempt, response):
                return self._decode(response)
            await asyncio.sleep(self._retry_delay(attempt, response))

    def _should_retry(
        self, attempt: int, response: Optional[httpx.Response]
    ) -> bool:
        if attempt >= self.max_retries:
            return False
        return (
            response is None
            or response.status_code in RETRYABLE_STATUSES
        )

    def _retry_delay(
        self, attempt: int, response: Optional[httpx.Response]
    ) -> float:
        delay = min(0.5 * 2**attempt, 8.0) * random.uniform(0.75, 1.0)
        headers = response.headers if response is not None else {}
        try:
            if "retry-after-ms" in headers:
                delay = float(headers["retry-after-ms"]) / 1000
            elif "retry-after" in headers:
                delay = float(headers["retry-after"])
        except ValueError:
            pass  # Retry-After can be an HTTP date; keep the backoff.
        delay = min(max(delay, 0.0), 60.0)

        status = (
            response.status_code
            if response is not None
            else "a connection error"
        )
        logger.warning(
            f"DecisionModel got {status}; retrying in {delay:.2f}s "
            f"(attempt {attempt + 1}/{self.max_retries})."
        )
        return delay

    def _decode(self, response: httpx.Response) -> Dict[str, Any]:
        if response.is_error:
            # The body names the offending field on a 422, which raise_for_status drops.
            raise httpx.HTTPStatusError(
                f"{response.status_code} from {response.url}: {response.text}",
                request=response.request,
                response=response,
            )
        return response.json()
