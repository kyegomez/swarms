import json

from swarms import Agent, CheckedSwarm

INVOICE = [
    ("A-100", 3, 19.99),
    ("B-220", 1, 249.00),
    ("C-310", 12, 2.35),
    ("A-100", 2, 19.99),
]


def total_matches(answer: str):
    total = json.loads(answer)["total"]
    expected = sum(qty * price for _, qty, price in INVOICE)
    return (
        round(total, 2) == round(expected, 2),
        "The total is wrong.",
    )


def skus_match(answer: str):
    skus = {sku for sku, _, _ in INVOICE}
    return set(json.loads(answer)) == skus, "The SKUs are wrong."


swarm = CheckedSwarm(
    agents=[
        Agent(agent_name=name, model_name="gpt-5.4-mini", max_loops=1)
        for name in ("Totaller", "SKU-Lister")
    ],
    checks=[total_matches, skus_match],
    parts=[
        'The invoice total in USD, as JSON only, no code fence: {"total": <number>}',
        "The distinct SKUs on the invoice, as a JSON list of strings only, no code fence",
    ],
)

task = (
    "Read this invoice. Each line is SKU, quantity, unit price in USD.\n"
    + "\n".join(
        f"{sku}, {qty}, {price:.2f}" for sku, qty, price in INVOICE
    )
)

print(swarm.run(task))
print(swarm.last_run)
print(swarm.usage)
