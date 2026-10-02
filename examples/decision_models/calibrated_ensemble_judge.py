import json
import statistics
import time
from collections import Counter

import litellm

from swarms import Agent, DecisionModel, run_agents_concurrently

TASK = (
    "Explain the difference between a mutex and a semaphore to a junior "
    "engineer in under 120 words, with one concrete example."
)
WORD_LIMIT = 120
RERUNS = 3
LLM_JUDGE_MODEL = "gpt-5.4"
# TypeSafe bills input tokens only; output tokens are free.
DECISION_PRICE_PER_MTOK = 0.042

ENSEMBLE = {
    "A": "gpt-5.4-mini",
    "B": "claude-sonnet-5-5",
    "C": "gemini/gemini-3.5-flash",
    "D": "openrouter/deepseek/deepseek-chat",
}

LEVELS = ["Poor", "Fair", "Good", "Excellent"]
CRITERIA = {
    "accuracy": "How technically accurate is `candidates.{label}`?",
    "clarity": "How clear is `candidates.{label}` for a junior engineer?",
    "example": "How well does the example in `candidates.{label}` make the difference obvious?",
}
WEIGHTS = {
    "accuracy": 0.45,
    "clarity": 0.25,
    "example": 0.2,
    "within_limit": 0.1,
}


def weighted_scores(ratings: dict, answers: dict) -> dict:
    """
    Combine per-criterion ratings with the word-limit check in code.

    Args:
        ratings: Ratings from 0 to 3, keyed by candidate label and criterion.
        answers: Candidate answers keyed by label.

    Returns:
        A score from 0 to 1 per candidate label.
    """
    scores = {}
    for label, by_criterion in ratings.items():
        within_limit = len(answers[label].split()) <= WORD_LIMIT
        scores[label] = WEIGHTS["within_limit"] * within_limit + sum(
            WEIGHTS[name] * by_criterion[name] / (len(LEVELS) - 1)
            for name in CRITERIA
        )
    return scores


def decision_judge(model: DecisionModel, answers: dict) -> tuple:
    """
    Rate every candidate on every criterion in one decision-model request.

    Args:
        model: Decision model used as the judge.
        answers: Candidate answers keyed by label.

    Returns:
        Weighted scores per label, and the request cost in US dollars.
    """
    questions = {
        f"{name}_{label}": {
            "type": "score",
            "instructions": template.format(label=label),
            "criteria": LEVELS,
        }
        for label in answers
        for name, template in CRITERIA.items()
    }
    response = model.run(
        state={"task": TASK, "candidates": answers},
        questions=questions,
    )
    ratings = {
        label: {
            name: response["answers"][f"{name}_{label}"]["score"]
            for name in CRITERIA
        }
        for label in answers
    }
    cost = (
        response["usage"]["input_tokens"]
        * DECISION_PRICE_PER_MTOK
        / 1_000_000
    )
    return weighted_scores(ratings, answers), cost


def llm_judge(answers: dict) -> tuple:
    """
    Rate every candidate on every criterion with an LLM judge.

    Args:
        answers: Candidate answers keyed by label.

    Returns:
        Weighted scores per label, and the call cost in US dollars.
    """
    # A fresh judge per rerun, so no run sees an earlier run's grades.
    judge = Agent(
        agent_name="LLM-Judge",
        system_prompt=(
            "You are a strict grader. Rate each candidate answer from 0 (poor) "
            f"to 3 (excellent) on: {', '.join(CRITERIA)}. Reply with JSON only, "
            'shaped like {"A": {"accuracy": 2, "clarity": 3, "example": 1}}.'
        ),
        model_name=LLM_JUDGE_MODEL,
        max_loops=1,
        output_type="final",
        print_on=False,
    )
    reply = judge.run(
        f"Task: {TASK}\n\nCandidates: {json.dumps(answers)}"
    )
    try:
        ratings = json.loads(
            reply[reply.index("{") : reply.rindex("}") + 1]
        )
        scores = weighted_scores(ratings, answers)
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError(
            f"The LLM judge did not return ratings: {reply[:300]}"
        ) from error

    prompt_cost, completion_cost = litellm.cost_per_token(
        model=LLM_JUDGE_MODEL,
        prompt_tokens=judge.usage["input_tokens"],
        completion_tokens=judge.usage["output_tokens"],
    )
    return scores, prompt_cost + completion_cost


def rerun(judge, label: str) -> dict:
    """
    Run a judge several times and summarise how stable it is.

    Args:
        judge: Callable returning scores per label and a cost.
        label: Name printed for the judge.

    Returns:
        Winners, mean latency, total cost and score spread.
    """
    winners, latencies, costs, runs = [], [], [], []
    for attempt in range(RERUNS):
        started = time.perf_counter()
        scores, cost = judge()
        latencies.append(time.perf_counter() - started)
        costs.append(cost)
        runs.append(scores)
        winners.append(max(scores, key=scores.get))
        ranked = ", ".join(
            f"{k} {v:.2f}"
            for k, v in sorted(scores.items(), key=lambda kv: -kv[1])
        )
        print(f"  {label} run {attempt + 1}: {ranked}")
    spread = statistics.mean(
        statistics.pstdev(run[candidate] for run in runs)
        for candidate in runs[0]
    )
    return {
        "winners": winners,
        "latency": statistics.mean(latencies),
        "cost": sum(costs),
        "spread": spread,
    }


agents = [
    Agent(
        agent_name=label,
        model_name=model_name,
        max_loops=1,
        max_tokens=4000,
        output_type="final",
        print_on=False,
    )
    for label, model_name in ENSEMBLE.items()
]

print(f"Asking {len(agents)} models: {TASK}\n")
answers = run_agents_concurrently(
    agents=agents, task=TASK, return_agent_output_dict=True
)
for label, answer in answers.items():
    print(
        f"[{label}] {ENSEMBLE[label]} ({len(answer.split())} words)\n{answer}\n"
    )

decision_model = DecisionModel(model_name="jev-latest")
print("Judging blind (labels only), several times each:")
results = {
    "Decision model": rerun(
        lambda: decision_judge(decision_model, answers),
        "Decision model",
    ),
    f"LLM judge ({LLM_JUDGE_MODEL})": rerun(
        lambda: llm_judge(answers), "LLM judge"
    ),
}

print()
for name, result in results.items():
    winner, count = Counter(result["winners"]).most_common(1)[0]
    print(
        f"{name}: picked {winner} ({ENSEMBLE[winner]}) in {count}/{RERUNS} "
        f"runs, score spread {result['spread']:.3f}, "
        f"{result['latency']:.2f} s per judgment, ${result['cost']:.5f} total"
    )
print(
    "\nScore spread is the average standard deviation of each candidate's "
    "score across reruns; 0 means the judge gave identical scores every time."
)
