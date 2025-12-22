"""
Custom Fairy Example

Create a swarm with custom fairy agents instead of the defaults.
Useful when you need specialized agent personalities.
"""

from swarms import Agent
from fairy_swarm import FairySwarm

copywriter = Agent(
    agent_name="Copywriter-Fairy",
    agent_description="Expert at writing compelling marketing copy and headlines",
    system_prompt="""You are the Copywriter Fairy - a wordsmith who crafts compelling copy.

Your Specialties:
- Headlines and taglines
- Call-to-action text
- Marketing copy
- Brand messaging

Always USE YOUR TOOLS to add text elements to the canvas.
Output format: COPY CREATED: [your copy], ACTIONS TAKEN: [tools used]""",
    model_name="gpt-4o-mini",
    max_loops=1,
)

ux_designer = Agent(
    agent_name="UX-Designer-Fairy",
    agent_description="Focused on user experience and interaction design",
    system_prompt="""You are the UX Designer Fairy - focused on user experience.

Your Specialties:
- User flows and journeys
- Interaction patterns
- Accessibility considerations
- Usability optimization

Always USE YOUR TOOLS to create UX elements on the canvas.
Output format: UX DESIGN: [your design], ACTIONS TAKEN: [tools used]""",
    model_name="gpt-4o-mini",
    max_loops=1,
)

swarm = FairySwarm(
    name="Marketing Team",
    model_name="gpt-4o-mini",
    max_loops=2,
    verbose=True,
    fairies=[copywriter, ux_designer],
    auto_create_default_fairies=False,
)

result = swarm.run(
    "Create a product launch landing page with compelling headlines, "
    "persuasive copy, and an optimized user flow for conversions."
)

print(result)
