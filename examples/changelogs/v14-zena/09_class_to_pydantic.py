"""Zena: generate a Pydantic schema from a class signature.

A constructor is already a schema: names, types, which args are required,
and a docstring describing each. This turns that into a BaseModel subclass
so it can drive validation or structured LLM output, without maintaining a
parallel definition that drifts.

Returns a model CLASS, not an instance.
"""

import json

from swarms import Agent
from swarms.utils.class_to_pydantic import (
    class_init_to_pydantic_model,
)

AgentSchema = class_init_to_pydantic_model(
    Agent,
    include=(
        "agent_name",
        "agent_description",
        "system_prompt",
        "model_name",
    ),
)

# The shape. Class-level, so no values are needed. Descriptions come from
# the Google-style Args: block, making this usable for function calling.
print(json.dumps(AgentSchema.model_json_schema(), indent=2))

# The values. Needs an instance.
spec = AgentSchema(agent_name="Analyst", system_prompt="You analyze.")
print(spec.model_dump())

# Round-trips straight back into a real Agent.
agent = Agent(**spec.model_dump())
print(type(agent).__name__, agent.agent_name, agent.model_name)
