import time
from types import SimpleNamespace

import swarms.utils.litellm_wrapper as litellm_wrapper
from swarms.structs.council_as_judge import (
    EVAL_DIMENSIONS,
    CouncilAsAJudge,
)


def test_judges_do_not_see_earlier_runs(monkeypatch):
    requests = []

    def fake_completion(**params):
        requests.append(str(params["messages"]))
        message = SimpleNamespace(
            content="ok", tool_calls=None, reasoning_content=None
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=message, finish_reason="stop")
            ],
            usage=None,
        )

    monkeypatch.setattr(
        litellm_wrapper, "completion", fake_completion
    )

    council = CouncilAsAJudge(
        model_name="gpt-4o-mini",
        aggregation_model_name="gpt-4o-mini",
    )
    council.run("FIRST-TASK-ALPHA")
    first_run = len(requests)
    council.run("SECOND-TASK-BETA")
    second_run = requests[first_run:]

    assert len(second_run) == first_run
    assert all("SECOND-TASK-BETA" in r for r in second_run)
    assert not any("FIRST-TASK-ALPHA" in r for r in second_run)


def test_judgements_are_recorded_in_dimension_order(monkeypatch):
    dimensions = list(EVAL_DIMENSIONS)

    def fake_completion(**params):
        text = str(params["messages"])
        for i, dim in enumerate(dimensions):
            if f"Evaluation Dimension: {dim.upper()}" in text:
                time.sleep(0.05 * (len(dimensions) - i))
        message = SimpleNamespace(
            content="ok", tool_calls=None, reasoning_content=None
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=message, finish_reason="stop")
            ],
            usage=None,
        )

    monkeypatch.setattr(
        litellm_wrapper, "completion", fake_completion
    )

    council = CouncilAsAJudge(
        model_name="gpt-4o-mini",
        aggregation_model_name="gpt-4o-mini",
    )
    council.max_workers = len(dimensions)
    council.run("TASK")

    roles = [
        m["role"] for m in council.conversation.conversation_history
    ]
    assert roles == [
        "User",
        *(f"{dim}_judge" for dim in dimensions),
        "aggregator_agent",
    ]
