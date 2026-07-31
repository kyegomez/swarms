"""Build a Pydantic model from a class's __init__ signature.

Types and defaults are both lifted straight off the constructor: parameters
without a default become required fields, parameters with one keep it.
"""

import json
from typing import List, Optional

from swarms.utils.class_to_pydantic import (
    class_init_to_pydantic_model,
)


class Agent:
    def __init__(
        self,
        name: str,
        age: int,
        model_name: str = "gpt-5.4",
        temperature: float = 0.5,
        max_loops: int = 1,
        tools: List[str] = [],
        owner: Optional[str] = None,
    ):
        """An agent.

        Args:
            name (str): The agent's name.
            age (int): The agent's age in years.
            model_name (str): LiteLLM model string backing the agent.
            temperature (float): Sampling temperature.
            max_loops (int): How many loops the agent runs before returning.
            tools (List[str]): Names of tools the agent may call.
            owner (Optional[str]): Who owns this agent, if anyone.
        """
        self.name = name
        self.age = age


# Returns a MODEL CLASS named AgentSchema, not an instance.
AgentSchema = class_init_to_pydantic_model(Agent)

# The shape. Class-level, so no values are needed.
print("--- schema ---")
print(json.dumps(AgentSchema.model_json_schema(), indent=2))

# Only the two parameters without defaults are required; the rest keep the
# defaults declared on __init__.
print("\n--- fields ---")
for field_name, field in AgentSchema.model_fields.items():
    default = (
        "<factory>" if field.default_factory else repr(field.default)
    )
    state = (
        "required" if field.is_required() else f"default={default}"
    )
    print(f"  {field_name:12} {state}")

# The values. Needs an instance — model_dump_json() is an instance method.
print("\n--- instance ---")
agent = AgentSchema(name="Alice", age=30)
print(agent.model_dump_json(indent=2))
