def generate_reasoning_prompt(max_loops: int) -> str:
    return f"""
    You will work on this task over up to {max_loops} turns. Each turn is
    one step: make concrete progress on the task and build on your earlier
    turns instead of repeating them. When the task is solved, or on the
    final turn, give your complete answer, starting with **Final Answer:**
    """
