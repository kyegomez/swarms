from swarms.structs.debate_with_judge import DebateWithJudge
from swarms.structs.agent import Agent


def _recording_agent(name):
    agent = Agent(
        agent_name=name,
        model_name="gpt-4o-mini",
        max_loops=1,
        autosave=False,
        verbose=False,
        output_type="final",
    )
    agent.calls = []

    def run(task, *args, **kwargs):
        image_call_count = sum(
            1
            for call in agent.calls
            if call.get("img") is not None
            or call.get("imgs") is not None
        )
        answer = (
            f"{agent.agent_name} image result {image_call_count + 1}"
            if kwargs.get("img") is not None
            or kwargs.get("imgs") is not None
            else f"{agent.agent_name} answer"
        )
        agent.calls.append({"task": task, **kwargs})
        agent.short_memory.add(agent.agent_name, answer)
        return answer

    agent.run = run
    return agent


def _debate():
    pro = _recording_agent("Pro-Debater")
    con = _recording_agent("Con-Debater")
    judge = _recording_agent("Debate-Judge")
    debate = DebateWithJudge(
        agents=[pro, con, judge],
        max_loops=1,
        output_type="final",
        verbose=False,
    )
    return debate, pro, con, judge


def test_run_forwards_img_to_each_debate_agent():
    debate, pro, con, judge = _debate()

    debate.run("Motion: inspect this chart", img="chart.png")

    assert pro.calls[-1]["img"] == "chart.png"
    assert con.calls[-1]["img"] == "chart.png"
    assert judge.calls[-1]["img"] == "chart.png"


def test_run_forwards_imgs_to_each_debate_agent():
    debate, pro, con, judge = _debate()
    images = ["chart-a.png", "chart-b.png"]

    debate.run("Motion: compare these charts", imgs=images)

    assert pro.calls[-1]["imgs"] == images
    assert con.calls[-1]["imgs"] == images
    assert judge.calls[-1]["imgs"] == images


def test_batched_run_pairs_imgs_by_task():
    debate, pro, con, judge = _debate()
    tasks = ["Motion: t1", "Motion: t2", "Motion: t3"]
    images = ["img1.png", "img2.png", "img3.png"]

    results = debate.batched_run(tasks, imgs=images)
    pro_debate_calls = [
        call for call in pro.calls if call.get("img") is not None
    ]

    assert results == [
        "Debate-Judge image result 1",
        "Debate-Judge image result 2",
        "Debate-Judge image result 3",
    ]
    assert [
        (task in call["task"], call["img"])
        for task, call in zip(tasks, pro_debate_calls)
    ] == [(True, img) for img in images]
    assert [
        call["img"] for call in pro.calls if call.get("img")
    ] == images
    assert [
        call["img"] for call in con.calls if call.get("img")
    ] == images
    assert [
        call["img"] for call in judge.calls if call.get("img")
    ] == images
