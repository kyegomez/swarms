from swarms import HeavySwarm


def main():
    """
    Run a HeavySwarm query to find the best and most promising treatments for diabetes.

    This function initializes a HeavySwarm instance and queries it to provide
    the top current and theoretical treatments for diabetes, requesting clear,
    structured, and evidence-based results suitable for medical research or clinical review.
    """
    swarm = HeavySwarm(
        name="Diabetes Treatment Research Team",
        description="A team of agents that research the best and most promising treatments for diabetes, including theoretical approaches.",
        worker_model_name="claude-sonnet-4-20250514",
        show_dashboard=True,
        question_agent_model_name="gpt-4.1",
        loops_per_agent=1,
    )

    prompt = (
        "Identify the best and most promising treatments for diabetes, including both current standard therapies and theoretical or experimental approaches. "
        "For each treatment, provide: the treatment name, type (e.g., medication, lifestyle intervention, device, gene therapy, etc.), "
        "mechanism of action, current stage of research or approval status, key clinical evidence or rationale, "
        "potential benefits and risks, and a brief summary of why it is considered promising. "
        "Present the information in a clear, structured format suitable for medical professionals or researchers."
    )

    out = swarm.run(prompt)
    print(out)


if __name__ == "__main__":
    main()
