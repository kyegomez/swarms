import asyncio
import math
import random
import time

import litellm

from swarms import (
    Agent,
    DecisionModel,
    run_agents_with_different_tasks,
)

NUM_RESUMES = 500
SHORTLIST_SHARE = 0.05
LLM_SAMPLE = 8
SCREEN_MODEL = "gpt-5.4-mini"
REVIEW_MODEL = "gpt-5.4"

JOB = {
    "title": "Senior Backend Engineer",
    "location": "Remote, United States only",
    "requirements": [
        "5+ years of backend engineering",
        "Python in production",
        "Building or operating distributed systems",
        "Hands-on PostgreSQL",
        "Authorized to work in the US without sponsorship",
    ],
}

SCREEN_QUESTIONS = {
    "fit": {
        "type": "score",
        "instructions": "How well does the `resume` match the `job`?",
        "criteria": [
            "No relevant experience",
            "Some transferable skills",
            "Partial match",
            "Strong match",
            "Exceptional match",
        ],
    },
    "python": {
        "type": "noul",
        "instructions": "The `resume` shows Python used in production backend work.",
    },
    "distributed": {
        "type": "noul",
        "instructions": "The `resume` shows building or operating distributed systems.",
    },
    "postgres": {
        "type": "noul",
        "instructions": "The `resume` shows hands-on PostgreSQL experience.",
    },
    "senior": {
        "type": "noul",
        "instructions": "The `resume` shows at least 5 years of backend engineering.",
    },
    "us_eligible": {
        "type": "noul",
        "instructions": "The candidate can work in the United States without visa sponsorship.",
    },
}

WEIGHTS = {
    "fit": 0.4,
    "python": 0.15,
    "distributed": 0.15,
    "postgres": 0.1,
    "senior": 0.1,
    "us_eligible": 0.1,
}

FIRST_NAMES = (
    "Ava Ben Chen Dara Eli Fatima Gus Hana Ivan Jade Kofi Lena Mateo "
    "Nia Omar Priya"
).split()
LAST_NAMES = (
    "Alvarez Brooks Cohen Dubois Eze Fischer Garcia Hughes Iyer Jensen "
    "Kim Lopez Moreau Novak"
).split()
SKILLS = (
    "Python Go Java TypeScript React PostgreSQL MySQL Kafka Kubernetes "
    "AWS Redis gRPC Terraform Swift Figma Selenium Spark Django"
).split()
ROLES = [
    "Backend Engineer",
    "Senior Backend Engineer",
    "Data Engineer",
    "Frontend Engineer",
    "DevOps Engineer",
    "Mobile Developer",
    "QA Analyst",
    "Product Designer",
    "Site Reliability Engineer",
    "Full-Stack Developer",
]
COMPANIES = [
    "a fintech startup",
    "a logistics company",
    "a hospital network",
    "an e-commerce marketplace",
    "a game studio",
    "a consulting firm",
    "a ride-sharing company",
    "a public SaaS company",
]
HIGHLIGHTS = [
    "built a Python payments API serving 5k requests per second",
    "migrated a monolith to services on Kubernetes",
    "ran on-call for a sharded PostgreSQL fleet",
    "designed React dashboards for operations teams",
    "automated regression tests with Selenium",
    "built Spark ETL pipelines feeding the data warehouse",
    "shipped an iOS app with 200k monthly users",
    "led a redesign of the checkout flow",
    "built an event pipeline on Kafka across three regions",
    "wrote internal tooling in Python and Django",
]
LOCATIONS = {
    "Austin, TX": "US citizen",
    "Remote (Denver, CO)": "Green card holder",
    "New York, NY": "US citizen",
    "Seattle, WA": "Requires H-1B transfer",
    "Toronto, Canada": "Requires visa sponsorship",
    "Berlin, Germany": "Requires visa sponsorship",
    "Bangalore, India": "Requires visa sponsorship",
    "Remote (Lisbon, Portugal)": "US citizen living abroad",
}


def make_resumes(count: int, seed: int = 7) -> list:
    """
    Generate synthetic resumes with a realistic spread of fits.

    Args:
        count: Number of resumes.
        seed: Random seed, so every run screens the same pool.

    Returns:
        Resume dictionaries.
    """
    rng = random.Random(seed)
    resumes = []
    for index in range(count):
        location = rng.choice(list(LOCATIONS))
        jobs = [
            {
                "role": rng.choice(ROLES),
                "company": rng.choice(COMPANIES),
                "years": rng.randint(1, 5),
                "highlight": rng.choice(HIGHLIGHTS),
            }
            for _ in range(rng.randint(1, 3))
        ]
        resumes.append(
            {
                "id": f"R{index:03d}",
                "name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
                "location": location,
                "work_authorization": LOCATIONS[location],
                "skills": rng.sample(SKILLS, rng.randint(3, 7)),
                "experience": jobs,
            }
        )
    return resumes


def composite_score(answers: dict) -> float:
    """
    Combine the screening answers with the weights above.

    Args:
        answers: Decision model answers keyed by question id.

    Returns:
        A score from 0 to 1.
    """
    fit_levels = len(SCREEN_QUESTIONS["fit"]["criteria"]) - 1
    values = {"fit": answers["fit"]["score"] / fit_levels}
    for question_id in WEIGHTS:
        if question_id != "fit":
            values[question_id] = answers[question_id]["noul"]
    return sum(WEIGHTS[key] * values[key] for key in WEIGHTS)


async def screen_all(model: DecisionModel, resumes: list) -> list:
    """
    Screen every resume with the decision model, 20 at a time.

    Args:
        model: Decision model that answers the screening questions.
        resumes: Resumes to screen.

    Returns:
        One result per resume with its score and latency.
    """
    limit = asyncio.Semaphore(20)

    async def screen(resume: dict) -> dict:
        async with limit:
            started = time.perf_counter()
            response = await model.arun(
                state={"job": JOB, "resume": resume},
                questions=SCREEN_QUESTIONS,
            )
            return {
                "resume": resume,
                "score": composite_score(response["answers"]),
                "seconds": time.perf_counter() - started,
            }

    return await asyncio.gather(*[screen(r) for r in resumes])


def llm_cost(model_name: str, usage: dict) -> float:
    """
    Price an agent's token usage with litellm's price table.

    Args:
        model_name: Model the agent ran on.
        usage: The agent's usage dictionary.

    Returns:
        Cost in US dollars.
    """
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model_name,
        prompt_tokens=usage["input_tokens"],
        completion_tokens=usage["output_tokens"],
    )
    return prompt_cost + completion_cost


def make_agent(
    name: str, model_name: str, system_prompt: str
) -> Agent:
    """
    Build a quiet single-turn agent.

    Args:
        name: Agent name.
        model_name: Model to run on.
        system_prompt: The agent's instructions.

    Returns:
        The agent.
    """
    return Agent(
        agent_name=name,
        system_prompt=system_prompt,
        model_name=model_name,
        max_loops=1,
        output_type="final",
        print_on=False,
    )


def bar_chart(title: str, rows: list, value_format: str) -> None:
    """
    Print a horizontal bar chart in the terminal.

    Args:
        title: Chart title, naming the measure and unit.
        rows: Pairs of label and value.
        value_format: Format string for each value.
    """
    print(f"\n{title}")
    largest = max(value for _, value in rows) or 1
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        bar = "█" * max(1, round(40 * value / largest))
        print(
            f"  {label:<{width}}  {bar} {value_format.format(value)}"
        )


resumes = make_resumes(NUM_RESUMES)
shortlist_size = math.ceil(NUM_RESUMES * SHORTLIST_SHARE)
decision_model = DecisionModel(model_name="jev-latest")

print(
    f"Screening {NUM_RESUMES} resumes with {decision_model.model_name}..."
)
started = time.perf_counter()
screened = asyncio.run(screen_all(decision_model, resumes))
decision_wall_s = time.perf_counter() - started
decision_cost = decision_model.calculate_cost()["total_cost"]
decision_item_s = sum(r["seconds"] for r in screened) / len(screened)

shortlist = sorted(screened, key=lambda r: r["score"], reverse=True)[
    :shortlist_size
]
print(
    f"Done in {decision_wall_s:.1f} s for ${decision_cost:.4f}. "
    f"Top {shortlist_size}:"
)
for result in shortlist[:5]:
    resume = result["resume"]
    print(
        f"  {resume['id']} {resume['name']:<16} score {result['score']:.2f} "
        f"| {resume['location']}"
    )

print(
    f"\nEstimating LLM screening from {LLM_SAMPLE} sample resumes "
    f"with {SCREEN_MODEL}..."
)
sample = random.Random(1).sample(resumes, LLM_SAMPLE)
llm_screen_costs, llm_screen_seconds = [], []
for resume in sample:
    # A fresh agent per resume keeps earlier resumes out of its context.
    screener = make_agent(
        "Screener",
        SCREEN_MODEL,
        "You screen resumes. Rate the fit from 0 to 10 and list which job requirements the resume meets.",
    )
    started = time.perf_counter()
    screener.run(f"Job: {JOB}\n\nResume: {resume}")
    llm_screen_seconds.append(time.perf_counter() - started)
    llm_screen_costs.append(llm_cost(SCREEN_MODEL, screener.usage))
llm_screen_cost = sum(llm_screen_costs) / LLM_SAMPLE * NUM_RESUMES
llm_item_s = sum(llm_screen_seconds) / LLM_SAMPLE

print(
    f"Deep review of the top {shortlist_size} with {REVIEW_MODEL}..."
)
reviewers = [
    make_agent(
        f"Reviewer-{r['resume']['id']}",
        REVIEW_MODEL,
        "You are a senior engineering hiring manager. Assess the candidate's strengths, gaps and risks for the job, then give three interview questions.",
    )
    for r in shortlist
]
reviews = run_agents_with_different_tasks(
    [
        (agent, f"Job: {JOB}\n\nResume: {r['resume']}")
        for agent, r in zip(reviewers, shortlist)
    ]
)
recruiters = [
    make_agent(
        f"Recruiter-{r['resume']['id']}",
        REVIEW_MODEL,
        "You are a recruiter. Write a short, personal outreach email that references the candidate's real experience.",
    )
    for r in shortlist
]
drafts = run_agents_with_different_tasks(
    [
        (
            agent,
            f"Resume: {r['resume']}\n\nHiring manager review: {review}",
        )
        for agent, r, review in zip(recruiters, shortlist, reviews)
    ]
)
review_cost = sum(
    llm_cost(REVIEW_MODEL, agent.usage)
    for agent in reviewers + recruiters
)

top = shortlist[0]["resume"]
print(f"\nOutreach draft for {top['name']} ({top['id']}):\n")
print(drafts[0])

bar_chart(
    f"Cost to process {NUM_RESUMES} resumes (USD)",
    [
        (
            "LLM agents screen all, then review",
            llm_screen_cost + review_cost,
        ),
        (
            "Decision model screens, agents review",
            decision_cost + review_cost,
        ),
    ],
    "${:.4f}",
)
bar_chart(
    "Time to screen one resume (seconds)",
    [
        (f"LLM agent ({SCREEN_MODEL})", llm_item_s),
        (
            f"Decision model ({decision_model.model_name})",
            decision_item_s,
        ),
    ],
    "{:.2f} s",
)
print(
    f"\nScreening cost: ${decision_cost:.4f} with the decision model vs "
    f"about ${llm_screen_cost:.4f} with LLM agents "
    f"(estimated from {LLM_SAMPLE} samples). "
    f"Both pipelines spend ${review_cost:.4f} on the deep reviews."
)
