"""Prompt templates for the Tree of Thoughts agent.

``TreeOfThoughts`` fills the placeholders: ``{tool}`` is the function the
model must call, ``{steps}`` is a numbered list of reasoning steps (or
``TOT_NO_STEPS``), and ``{guidance}`` and ``{criteria}`` are empty unless the
agent was given ``thought_description`` or ``evaluation_criteria``.
"""

TREE_OF_THOUGHTS_SYSTEM_PROMPT = """You are a careful, deliberate problem solver. You work one step at a time, check each step for errors, and prefer a correct answer over a fast one. When asked to call a function, respond only with that function call."""

TOT_NO_STEPS = "(none yet)"

TOT_GENERATE_PROMPT = """
TASK:
{task}

STEPS SO FAR:
{steps}
{guidance}
You are writing step {depth} of at most {max_depth}. Propose {candidates}. {rule}

Call the `{tool}` function with your candidates."""

TOT_ONE_CANDIDATE = "one candidate next step"

TOT_MANY_CANDIDATES = "{count} distinct candidate next steps, each taking a different approach"

TOT_STEP_RULE = "Each candidate is a single step that builds on the steps so far. If a candidate completes the task, state the full final answer in it and set is_final to true."

TOT_LAST_STEP_RULE = "This is the last step allowed: every candidate must complete the task, state the full final answer, and set is_final to true."

TOT_THOUGHT_GUIDANCE = """
WHAT ONE STEP LOOKS LIKE:
{thought_description}
"""

TOT_EVALUATION_CRITERIA = """
HOW TO JUDGE PROGRESS:
{evaluation_criteria}
"""

TOT_VALUE_PROMPT = """TASK:
{task}

PARTIAL SOLUTION:
{steps}
{criteria}
{instruction}

Call the `{tool}` function with your critique and score."""

TOT_VALUE_PARTIAL_INSTRUCTION = "Check the last step for errors, then judge how likely this path is to reach a correct, complete answer."

TOT_VALUE_FINAL_INSTRUCTION = "The last step gives the final answer. Judge whether that answer is correct and complete."

TOT_VOTE_PROMPT = """TASK:
{task}

{candidates}
{criteria}
Decide which candidate is most likely to lead to a correct, complete answer. A candidate marked FINAL claims to answer the task; judge whether that answer is correct.

Call the `{tool}` function with the number of the best candidate."""

TOT_VOTE_CANDIDATE = """CANDIDATE {index}{marker}:
{steps}"""

TOT_FINAL_MARKER = " (FINAL)"

TOT_FINAL_PROMPT = """TASK:
{task}

REASONING:
{steps}

{status}

Call the `{tool}` function with the complete final answer."""

TOT_FINAL_STATUS_SOLVED = (
    "These steps solve the task. Write the final answer they reach."
)

TOT_FINAL_STATUS_PARTIAL = "The search did not finish; these are the most promising steps found. Complete the reasoning, fix any errors, and write the final answer."

TOT_FINAL_STATUS_EMPTY = (
    "No reasoning steps were found. Solve the task directly."
)
