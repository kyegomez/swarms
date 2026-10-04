import time

import litellm

from swarms import Agent, DecisionModel

MOTION = (
    "Remote-first companies will outcompete office-first companies "
    "over the next decade."
)
MAX_ROUNDS = 6
CONVERGED_BELOW = 0.5
DEBATER_MODEL = "gpt-5.4-mini"

STRENGTH_LEVELS = ["Weak", "Adequate", "Strong", "Compelling"]

pro = Agent(
    agent_name="Pro",
    system_prompt=(
        f"You argue FOR the motion: {MOTION} Reply to your opponent's "
        "latest point. Raise a new argument only if nobody has made it yet; "
        "otherwise defend your strongest existing point. Under 100 words."
    ),
    model_name=DEBATER_MODEL,
    max_loops=1,
    output_type="final",
    print_on=False,
)
con = Agent(
    agent_name="Con",
    system_prompt=(
        f"You argue AGAINST the motion: {MOTION} Reply to your opponent's "
        "latest point. Raise a new argument only if nobody has made it yet; "
        "otherwise defend your strongest existing point. Under 100 words."
    ),
    model_name=DEBATER_MODEL,
    max_loops=1,
    output_type="final",
    print_on=False,
)
referee = DecisionModel(model_name="jev-latest")

rounds = []
referee_ms = []
latest_con = "Open the debate."
for round_number in range(1, MAX_ROUNDS + 1):
    # Each debater keeps its own history, so it only needs the opponent's latest turn.
    pro_turn = pro.run(f"Opponent: {latest_con}")
    con_turn = con.run(f"Opponent: {pro_turn}")
    latest_con = con_turn

    started = time.perf_counter()
    answers = referee.run(
        state={
            "motion": MOTION,
            "earlier_rounds": rounds,
            "latest_round": {"pro": pro_turn, "con": con_turn},
        },
        questions={
            "new_argument": {
                "type": "noul",
                "instructions": "The `latest_round` makes a substantive argument that none of the `earlier_rounds` made.",
            },
            "pro_strength": {
                "type": "score",
                "instructions": "Across all rounds, how strong is the case FOR the motion?",
                "criteria": STRENGTH_LEVELS,
            },
            "con_strength": {
                "type": "score",
                "instructions": "Across all rounds, how strong is the case AGAINST the motion?",
                "criteria": STRENGTH_LEVELS,
            },
        },
    )["answers"]
    referee_ms.append((time.perf_counter() - started) * 1000)
    rounds.append({"pro": pro_turn, "con": con_turn})

    novelty = answers["new_argument"]["noul"]
    pro_score = answers["pro_strength"]["score"]
    con_score = answers["con_strength"]["score"]
    converged = round_number > 1 and novelty < CONVERGED_BELOW
    print(
        f"Round {round_number} | new argument {novelty:.2f} | "
        f"pro {pro_score:.2f} | con {con_score:.2f} | "
        f"referee {referee_ms[-1]:.0f} ms"
        + (" -> converged, stopping" if converged else "")
    )
    if converged:
        break

if abs(pro_score - con_score) < 0.25:
    verdict = "too close to call"
else:
    verdict = "Pro wins" if pro_score > con_score else "Con wins"
print(
    f"\nVerdict: {verdict} (pro {pro_score:.2f} vs con {con_score:.2f} "
    f"on a 0-{len(STRENGTH_LEVELS) - 1} scale)"
)

calls_made = 2 * len(rounds)
calls_fixed = 2 * MAX_ROUNDS
tokens_per_call = (
    pro.usage["total_tokens"] + con.usage["total_tokens"]
) / calls_made
prompt_cost, completion_cost = litellm.cost_per_token(
    model=DEBATER_MODEL,
    prompt_tokens=pro.usage["input_tokens"]
    + con.usage["input_tokens"],
    completion_tokens=pro.usage["output_tokens"]
    + con.usage["output_tokens"],
)
cost_per_call = (prompt_cost + completion_cost) / calls_made
saved_calls = calls_fixed - calls_made

print(
    f"\nRounds: {len(rounds)} of {MAX_ROUNDS}. "
    f"LLM calls: {calls_made} instead of {calls_fixed}, "
    f"{saved_calls} saved."
)
if saved_calls:
    print(
        f"That is about {saved_calls * tokens_per_call:,.0f} tokens and "
        f"${saved_calls * cost_per_call:.4f} saved; later rounds carry more "
        "history, so the real saving is higher."
    )
else:
    print(
        "The debate kept producing new arguments, so it ran every round."
    )
print(
    f"The referee added {sum(referee_ms) / 1000:.2f} s across "
    f"{len(referee_ms)} checks."
)
