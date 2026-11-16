"""Marketing's brief is checked against the pricing spec before the engineer reads it."""

from swarms import (
    Agent,
    DecisionModel,
    FileChannel,
    GatedChannel,
    spec_gate,
)

SPEC = "Discount codes never stack. When a customer enters several codes at checkout, apply only the single code that gives the largest discount."

# The checker reads the spec beside each message, so it can catch a false fact that sounds plain.
channel = GatedChannel(
    FileChannel("runs/spec_gated_channel.md"),
    spec_gate(DecisionModel("jev-latest"), SPEC),
)

marketing = Agent(
    agent_name="Marketing",
    system_prompt="You run promotions. You are sure customers can combine every discount code they hold, and you tell engineers so.",
    model_name="gpt-5.4-mini",
    max_loops=1,
    channel=channel,
)

engineer = Agent(
    agent_name="Pricing-Engineer",
    system_prompt="You write checkout pricing rules. Follow the spec and use what others in the conversation tell you.",
    model_name="gpt-5.4-mini",
    max_loops=1,
    output_type="final",
    channel=channel,
)

marketing.run(
    "Brief the pricing engineer on how discount codes combine in the holiday sale."
)
rule = engineer.run(
    f"Write the checkout rule for discount codes as pseudocode.\n\nSpec: {SPEC}"
)

for verdict in channel.blocked:
    print(f"Blocked {verdict['agent_name']}: {verdict['reason']}")
    print(f"    {verdict['content'][:200]}\n")
print(rule)
