from swarms import Agent
from swarms.structs.phased_fan_out_workflow import (
    PhasedFanOutWorkflow,
)

CONTRACTS = {
    "acme_supply.txt": "If Supplier undergoes a change of control, Buyer may terminate this Agreement on 30 days' notice.",
    "globex_license.txt": "This license is non-transferable. Licensee may not assign this license without prior written consent.",
    "initech_services.txt": "Either party may terminate for convenience with 90 days' notice. Fees are payable monthly in arrears.",
    "umbrella_lease.txt": "A merger, acquisition or sale of a majority of Tenant's shares is deemed an assignment requiring Landlord's consent.",
}

reader = Agent(
    agent_name="Clause-Reader",
    agent_description="Reads one contract and quotes any clause triggered by a change of ownership or control.",
    system_prompt="You read a single contract and quote, word for word, every clause that is triggered by a change of control, merger, acquisition or sale of shares. If there is none, say so.",
    model_name="gpt-5.4-mini",
    max_loops=1,
)

checker = Agent(
    agent_name="Clause-Checker",
    agent_description="Independently checks whether a quoted clause really is a change-of-control clause.",
    system_prompt="You are a skeptical contracts lawyer. Given a contract and a quoted clause, decide whether the clause is triggered by a change of control: a merger, an acquisition, or a sale of a controlling stake. A clause that treats such an event as an assignment counts. A plain ban on assignment that never mentions a change of ownership does not. Give a verdict and one sentence of reasoning.",
    model_name="gpt-5.4-mini",
    max_loops=1,
)

task = (
    "Which of these contracts contain a change-of-control clause? "
    "Read each contract separately, then have every candidate clause checked independently.\n\n"
    + "\n\n".join(
        f"# {name}\n{text}" for name, text in CONTRACTS.items()
    )
)

workflow = PhasedFanOutWorkflow(
    model_name="gpt-5.4",
    agents=[reader, checker],
    allow_inline_agents=False,
    size_guideline="small",
)
print(workflow.run(task))
print(workflow.usage)
