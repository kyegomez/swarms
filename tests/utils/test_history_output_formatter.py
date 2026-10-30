import json

import pytest
import yaml

from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)


@pytest.fixture
def conversation(tmp_path, monkeypatch):
    """A two-message conversation created in an isolated directory."""
    monkeypatch.chdir(tmp_path)
    conv = Conversation()
    conv.add("user", "What is 2+2?")
    conv.add("assistant", "4.")
    return conv


def test_list_returns_messages(conversation):
    result = history_output_formatter(conversation, "list")
    assert result == [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
    ]


@pytest.mark.parametrize("output_type", ["dict", "dictionary"])
def test_dict_aliases_match_to_dict(conversation, output_type):
    result = history_output_formatter(conversation, output_type)
    assert result == conversation.to_dict()


def test_dict_result_is_detached_from_the_history(conversation):
    result = history_output_formatter(conversation, "dict")
    conversation.add("user", "And 3+3?")
    result[0]["content"] = "edited"
    result.append({"role": "user", "content": "injected"})

    assert [m["content"] for m in result] == [
        "edited",
        "4.",
        "injected",
    ]
    assert [m["content"] for m in conversation.to_dict()] == [
        "What is 2+2?",
        "4.",
        "And 3+3?",
    ]


@pytest.mark.parametrize("output_type", ["string", "str", "all"])
def test_string_aliases_include_every_message(
    conversation, output_type
):
    result = history_output_formatter(conversation, output_type)
    assert isinstance(result, str)
    assert "user: What is 2+2?" in result
    assert "assistant: 4." in result


@pytest.mark.parametrize("output_type", ["final", "last"])
def test_final_aliases_return_last_content(conversation, output_type):
    assert history_output_formatter(conversation, output_type) == "4."


def test_json_round_trips_to_dict(conversation):
    result = history_output_formatter(conversation, "json")
    assert json.loads(result) == conversation.to_dict()


def test_yaml_round_trips_to_dict(conversation):
    result = history_output_formatter(conversation, "yaml")
    assert yaml.safe_load(result) == conversation.to_dict()


def test_dict_all_except_first_drops_first_message(conversation):
    result = history_output_formatter(
        conversation, "dict-all-except-first"
    )
    assert result == [{"role": "assistant", "content": "4."}]


def test_str_all_except_first_drops_first_message(conversation):
    result = history_output_formatter(
        conversation, "str-all-except-first"
    )
    assert result == "4."


def test_list_final_wraps_last_content(conversation):
    assert history_output_formatter(conversation, "list-final") == [
        "4."
    ]


def test_unknown_type_raises_value_error(conversation):
    with pytest.raises(ValueError, match="Invalid type: nope"):
        history_output_formatter(conversation, "nope")
