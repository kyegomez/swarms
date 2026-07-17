from swarms import Agent, GroupChat


MODEL_NAME = "openrouter/google/gemma-4-26b-a4b-it:free"


manufacturing_strategist = Agent(
    agent_name="Manufacturing Strategist",
    agent_description=(
        "Maps plant location, supplier, capital expenditure, and operating "
        "model tradeoffs for reshoring manufacturing."
    ),
    system_prompt=(
        "You are a manufacturing strategist. Focus on where reshoring makes "
        "economic sense, what production should move first, and how companies "
        "can phase factory investments without disrupting customers."
    ),
    model_name=MODEL_NAME,
    max_loops=1,
    persistent_memory=False,
    print_on=False,
    output_type="final",
)

supply_chain_economist = Agent(
    agent_name="Supply Chain Economist",
    agent_description=(
        "Evaluates landed cost, resilience, inventory, logistics, tariffs, "
        "and supplier concentration risks."
    ),
    system_prompt=(
        "You are a supply chain economist. Ground the discussion in total "
        "landed cost, resilience, tariff exposure, lead times, working capital, "
        "and the hidden costs of geographically concentrated supply chains."
    ),
    model_name=MODEL_NAME,
    max_loops=1,
    persistent_memory=False,
    print_on=False,
    output_type="final",
)

workforce_leader = Agent(
    agent_name="Workforce Leader",
    agent_description=(
        "Represents labor availability, training pipelines, wages, safety, "
        "and regional workforce development."
    ),
    system_prompt=(
        "You are a workforce development leader. Push the group to address "
        "skills gaps, apprenticeships, wage quality, safety, community impact, "
        "and how automation changes the kinds of jobs created."
    ),
    model_name=MODEL_NAME,
    max_loops=1,
    persistent_memory=False,
    print_on=False,
    output_type="final",
)

automation_engineer = Agent(
    agent_name="Automation Engineer",
    agent_description=(
        "Assesses robotics, factory software, quality control, and process "
        "automation needed for competitive domestic production."
    ),
    system_prompt=(
        "You are an automation engineer. Explain how robotics, digital twins, "
        "quality systems, predictive maintenance, and flexible manufacturing "
        "can make reshored production competitive despite higher labor costs."
    ),
    model_name=MODEL_NAME,
    max_loops=1,
    persistent_memory=False,
    print_on=False,
    output_type="final",
)

policy_advisor = Agent(
    agent_name="Trade Policy Advisor",
    agent_description=(
        "Connects reshoring decisions to industrial policy, permitting, "
        "tax credits, national security, and trade relationships."
    ),
    system_prompt=(
        "You are a trade policy advisor. Discuss incentives, permitting, "
        "national security priorities, friendly-shoring, environmental rules, "
        "and where policy can help or distort reshoring decisions."
    ),
    model_name=MODEL_NAME,
    max_loops=1,
    persistent_memory=False,
    print_on=False,
    output_type="final",
)

chat = GroupChat(
    name="Reshoring Manufacturing GroupChat",
    description=(
        "A cross-functional panel debating practical reshoring strategies for "
        "manufacturing."
    ),
    agents=[
        manufacturing_strategist,
        supply_chain_economist,
        workforce_leader,
        automation_engineer,
        policy_advisor,
    ],
    max_loops=10,
    threshold=0.45,
    recency_penalty=0.2,
    output_type="str-all-except-first",
)

result = chat.run(
    "Discuss whether US companies should reshore more manufacturing over the "
    "next decade. Identify the strongest business case, the biggest risks, "
    "which industries should move first, and a practical 3-step roadmap."
)

print(result)
