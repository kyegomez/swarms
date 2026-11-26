"""Images given to an Agent reach the model.

The first part stubs the provider and checks the request. The live part
calls OpenAI and Anthropic, and is skipped when the key is not set.
"""

import asyncio
import base64
import os
import struct
import zlib
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest
from dotenv import load_dotenv

import swarms.utils.litellm_wrapper as litellm_wrapper
from swarms import (
    Agent,
    AgentRearrange,
    GraphWorkflow,
    MajorityVoting,
)


load_dotenv()

LIVE_MODELS = [
    ("gpt-4.1", "OPENAI_API_KEY"),
    ("claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
]


def _png(
    size: int = 16, colour: Optional[Tuple[int, int, int]] = None
) -> bytes:
    """Build a valid PNG: one solid colour, or a gradient when none is given."""
    rows = b"".join(
        b"\x00"
        + (
            bytes(colour) * size
            if colour
            else bytes((x * y) % 256 for x in range(size * 3))
        )
        for y in range(size)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body))
        )

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


PNG_BYTES = _png()
RAW_BASE64 = base64.b64encode(PNG_BYTES).decode()
DATA_URI = f"data:image/png;base64,{RAW_BASE64}"
URL = "https://example.com/chart.png"
RED_PNG = _png(64, (255, 0, 0))
BLUE_PNG = _png(64, (0, 0, 255))
COLOUR_TASK = (
    "Name the colour that fills each image. Use one word per image."
)
TASK = "What is in this image?"


def _reply(*args: Any, **kwargs: Any) -> SimpleNamespace:
    message = SimpleNamespace(content="ok", tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)], usage=None
    )


async def _async_reply(*args: Any, **kwargs: Any) -> SimpleNamespace:
    return _reply()


def _agent(model_name: str = "gpt-4.1", **kwargs: Any) -> Agent:
    kwargs.setdefault("max_loops", 1)
    return Agent(
        agent_name="Vision",
        model_name=model_name,
        print_on=False,
        persistent_memory=False,
        **kwargs,
    )


def _requests(run: Any) -> List[List[Dict[str, Any]]]:
    """Run with the provider stubbed and return each request's messages."""
    fake_keys = {
        "OPENAI_API_KEY": "test",
        "ANTHROPIC_API_KEY": "test",
    }
    with patch.dict(os.environ, fake_keys), patch.object(
        litellm_wrapper, "completion", side_effect=_reply
    ) as completion, patch.object(
        litellm_wrapper, "acompletion", side_effect=_async_reply
    ) as acompletion:
        run()

    calls = completion.call_args_list + acompletion.call_args_list
    return [call.kwargs["messages"] for call in calls]


def _image_urls(messages: List[Dict[str, Any]]) -> List[str]:
    return [
        part["image_url"]["url"]
        for message in messages
        if isinstance(message["content"], list)
        for part in message["content"]
        if part["type"] == "image_url"
    ]


@pytest.mark.parametrize(
    "run_kwargs, agent_kwargs, expected",
    [
        ({"img": URL}, {}, [URL]),
        ({"imgs": [URL, DATA_URI]}, {}, [URL, DATA_URI]),
        ({"img": URL, "imgs": [DATA_URI]}, {}, [URL, DATA_URI]),
        ({"img": URL}, {"system_prompt": "Be brief."}, [URL]),
        ({"img": URL}, {"model_name": "claude-sonnet-4-6"}, [URL]),
        ({}, {}, []),
    ],
)
def test_run_sends_the_images_with_the_task(
    run_kwargs, agent_kwargs, expected
):
    agent = _agent(**agent_kwargs)

    (messages,) = _requests(lambda: agent.run(TASK, **run_kwargs))

    assert _image_urls(messages) == expected
    assert messages[-1]["role"] == "user"
    assert TASK in str(messages[-1]["content"])


@pytest.mark.parametrize("source", ["raw", "data_uri", "path"])
def test_every_image_source_becomes_a_data_uri(source, tmp_path):
    path = tmp_path / "pixel.png"
    path.write_bytes(PNG_BYTES)
    image = {
        "raw": RAW_BASE64,
        "data_uri": DATA_URI,
        "path": str(path),
    }[source]

    (messages,) = _requests(lambda: _agent().run(TASK, img=image))

    (sent,) = _image_urls(messages)
    assert sent.startswith("data:image/")
    assert sent.endswith(RAW_BASE64)


def test_arun_sends_the_image():
    agent = _agent()

    (messages,) = _requests(
        lambda: asyncio.run(agent.arun(TASK, img=URL))
    )

    assert _image_urls(messages) == [URL]


def test_every_loop_keeps_the_image():
    agent = _agent(max_loops=2)

    requests = _requests(lambda: agent.run(TASK, img=URL))

    assert len(requests) == 2
    assert all(
        _image_urls(messages) == [URL] for messages in requests
    )


def test_image_joins_the_newest_turn_after_prior_messages():
    prior = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]

    (messages,) = _requests(
        lambda: _agent().run(TASK, img=URL, messages=prior)
    )

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert isinstance(messages[0]["content"], str)
    assert _image_urls(messages[-1:]) == [URL]


def test_image_is_not_written_into_the_conversation():
    agent = _agent()

    _requests(lambda: agent.run(TASK, img=DATA_URI))

    history = str(agent.short_memory.conversation_history)
    assert RAW_BASE64 not in history


def test_model_the_catalog_calls_text_only_still_gets_the_image():
    agent = _agent(model_name="gpt-3.5-turbo")

    (messages,) = _requests(lambda: agent.run(TASK, img=DATA_URI))

    assert _image_urls(messages) == [DATA_URI]


@pytest.mark.parametrize(
    "model, base_url, sent",
    [
        ("gpt-4.1", None, URL),
        ("gpt-4.1", "http://localhost:11434/v1", DATA_URI),
        ("gpt-4.1", "http://127.0.0.1:8000/v1", DATA_URI),
        ("ollama/llava", None, DATA_URI),
    ],
)
def test_local_endpoints_get_image_urls_as_base64(
    model, base_url, sent
):
    llm = litellm_wrapper.LiteLLM(model_name=model, base_url=base_url)
    with patch.object(
        litellm_wrapper, "_model_supports_vision", return_value=True
    ), patch.object(
        litellm_wrapper, "get_image_base64", return_value=DATA_URI
    ):
        block = llm._build_image_block(URL)

    assert block["image_url"]["url"] == sent


def _majority_voting(agents: List[Agent]) -> MajorityVoting:
    return MajorityVoting(agents=agents, max_loops=2)


def _agent_rearrange(agents: List[Agent]) -> AgentRearrange:
    return AgentRearrange(
        agents=agents,
        flow="A -> B",
        max_loops=2,
        collab_prompt="Build on each other's work.",
    )


def _graph_workflow(agents: List[Agent]) -> GraphWorkflow:
    workflow = GraphWorkflow(name="Images")
    for agent in agents:
        workflow.add_node(agent)
    workflow.add_edge("A", "B")
    return workflow


@pytest.mark.parametrize(
    "build", [_majority_voting, _agent_rearrange, _graph_workflow]
)
def test_structures_send_the_image_on_the_task_turn(build):
    """The image rides on the task, ahead of the turns that change, so it stays cacheable."""
    agents = [
        Agent(
            agent_name=name,
            model_name="gpt-4.1",
            max_loops=1,
            print_on=False,
            persistent_memory=False,
        )
        for name in ("A", "B")
    ]
    structure = build(agents)

    requests = _requests(lambda: structure.run(TASK, img=URL))

    with_image = [m for m in requests if _image_urls(m)]
    image_turns = []
    for messages in with_image:
        (turn,) = [
            i
            for i, m in enumerate(messages)
            if isinstance(m["content"], list)
            and m["content"][0]["type"] == "image_url"
        ]
        first_user = next(
            i for i, m in enumerate(messages) if m["role"] == "user"
        )
        assert turn == first_user
        assert TASK in str(messages[turn]["content"])
        image_turns.append(turn < len(messages) - 1)

    assert any(image_turns), "no request had turns after the image"


def _data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _live_agent(model_name: str, key: str) -> Agent:
    """A real agent for one provider, or a skip when its key is not set."""
    if not os.getenv(key):
        pytest.skip(f"{key} is not set")
    return _agent(model_name=model_name, output_type="final")


@pytest.mark.parametrize("model_name, key", LIVE_MODELS)
def test_live_model_names_the_colour_and_bills_the_image(
    model_name, key
):
    agent = _live_agent(model_name, key)

    text_only = _live_agent(model_name, key)

    answer = agent.run(COLOUR_TASK, img=_data_uri(RED_PNG))
    text_only.run(COLOUR_TASK)

    assert "red" in str(answer).lower()
    assert (
        agent.usage["input_tokens"] > text_only.usage["input_tokens"]
    )


@pytest.mark.parametrize("model_name, key", LIVE_MODELS)
def test_live_model_names_the_colours_of_two_images(model_name, key):
    agent = _live_agent(model_name, key)

    answer = agent.run(
        COLOUR_TASK,
        imgs=[_data_uri(RED_PNG), _data_uri(BLUE_PNG)],
    )

    assert "red" in str(answer).lower()
    assert "blue" in str(answer).lower()


@pytest.mark.parametrize("model_name, key", LIVE_MODELS)
def test_live_model_reads_an_image_file_through_arun(
    model_name, key, tmp_path
):
    agent = _live_agent(model_name, key)
    path = tmp_path / "blue.png"
    path.write_bytes(BLUE_PNG)

    answer = asyncio.run(agent.arun(COLOUR_TASK, img=str(path)))

    assert "blue" in str(answer).lower()
