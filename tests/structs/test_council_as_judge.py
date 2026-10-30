import threading
from types import SimpleNamespace

import swarms.utils.litellm_wrapper as litellm_wrapper
from swarms.structs.council_as_judge import CouncilAsAJudge


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


def test_council_reuses_its_pool_until_closed():
    threads = []

    def run(*args, **kwargs):
        threads.append(threading.current_thread())
        return "ok"

    council = CouncilAsAJudge(model_name="gpt-4o-mini")
    for agent in council.judge_agents.values():
        agent.run = run
    council.aggregator_agent.run = run

    council.run("FIRST-TASK-ALPHA")
    first = {t for t in threads if t is not threading.main_thread()}
    threads.clear()
    council.run("SECOND-TASK-BETA")
    assert first & set(threads)

    council.close()
    assert not any(t.is_alive() for t in first)
