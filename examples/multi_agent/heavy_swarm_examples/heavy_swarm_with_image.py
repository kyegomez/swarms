import os

from swarms import HeavySwarm

MODEL = "claude-haiku-4-5-20251001"

IMAGE = os.path.join(
    os.path.dirname(__file__), "revenue_by_region.png"
)

swarm = HeavySwarm(
    name="Chart-Analysis-Swarm",
    description="Reads a revenue chart and advises on what to do next.",
    worker_model_name=MODEL,
    question_agent_model_name=MODEL,
    variant="medium",
    max_loops=1,
    show_dashboard=False,
    agent_prints_on=False,
)

task = (
    "Analyse this chart and recommend what the business should do "
    "next quarter. The task text says nothing about the numbers; "
    "everything has to come from the image."
)

if __name__ == "__main__":
    # The decomposer sees the image too, so the questions it writes for
    # the workers refer to what the chart actually shows.
    questions = swarm.get_questions_only(task, img=IMAGE)
    for key, question in questions.items():
        print(f"[{key}] {question}\n")

    print("=" * 70)
    print(swarm.run(task, img=IMAGE))
