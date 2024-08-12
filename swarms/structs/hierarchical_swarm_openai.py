import os
from typing import List

from pydantic import BaseModel

from swarms.models.openai_function_caller import OpenAIFunctionCaller
from swarms import OpenAIChat
from swarms.structs.agent import Agent
from swarms.structs.concat import concat_strings

api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class
model = OpenAIChat(
    api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)


# Initialize the agents
growth_agent1 = Agent(
    agent_name="marketing_specialist",
    system_prompt="You're the marketing specialist, your purpose is to help companies grow by improving their marketing strategies!",
    agent_description="Improve a company's marketing strategies!",
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    saved_state_path="marketing_specialist.json",
    stopping_token="Stop!",
    context_length=1000,
)

growth_agent2 = Agent(
    agent_name="sales_specialist",
    system_prompt="You're the sales specialist, your purpose is to help companies grow by improving their sales strategies!",
    agent_description="Improve a company's sales strategies!",
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    saved_state_path="sales_specialist.json",
    stopping_token="Stop!",
    context_length=1000,
)

growth_agent3 = Agent(
    agent_name="product_development_specialist",
    system_prompt="You're the product development specialist, your purpose is to help companies grow by improving their product development strategies!",
    agent_description="Improve a company's product development strategies!",
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    saved_state_path="product_development_specialist.json",
    stopping_token="Stop!",
    context_length=1000,
)

team = [growth_agent1, growth_agent2, growth_agent3]


# class HiearchicalSwarm(BaseModel):
#     agents: List[Agent]
#     director: Agent
#     planner: Agent
#     max_loops: int = 3
#     verbose: bool = True

#     def run(self, task: str):
#         # Plan
#         # Plan -> JSON Function call -> workers -> response fetch back to boss -> planner
#         responses = []
#         responses.append(task)

#         for _ in range(self.max_loops):
#             # Plan
#             plan = self.planner.run(concat_strings(responses))
#             logger.info(f"Agent {self.planner.agent_name} planned: {plan}")
#             responses.append(plan)

#             # Execute json function calls
#             calls = self.director.run(plan)
#             logger.info(
#                 f"Agent {self.director.agent_name} called: {calls}"
#             )
#             responses.append(calls)
#             # Parse and send tasks to agents
#             output = parse_then_send_tasks_to_agents(self.agents, calls)

#             # Fetch back to boss
#             responses.append(output)

#         return concat_strings(responses)

#     def __call__(self, task: str):
#         responses = []
#         responses.append(task)

#         for _ in range(self.max_loops):
#             output = self.step(task, responses)
#             responses.append(output)

#         return concat_strings(responses)

#     def step(self, responses: List[str] = None) -> str:
#         # Plan
#         # Plan -> JSON Function call -> workers -> response fetch back to boss -> planner

#         # Plan
#         plan = self.planner.run(concat_strings(responses))
#         logger.info(f"Agent {self.planner.agent_name} planned: {plan}")
#         responses.append(plan)

#         # Execute json function calls
#         calls = self.director.run(plan)
#         logger.info(f"Agent {self.director.agent_name} called: {calls}")
#         responses.append(calls)
#         # Parse and send tasks to agents
#         output = parse_then_send_tasks_to_agents(self.agents, calls)

#         # Fetch back to boss
#         responses.append(output)

#         return concat_strings(responses)

#     def plan(self, task: str, responses: List[str] = None):
#         # Plan
#         # Plan -> JSON Function call -> workers -> response fetch back to boss -> planner
#         # responses = []
#         # responses.append(task)

#         # Plan
#         plan = self.planner.run(concat_strings(responses))
#         logger.info(f"Agent {self.planner.agent_name} planned: {plan}")
#         responses.append(plan)

#         return concat_strings(responses)


def agents_list(
    agents: List[Agent] = team,
) -> str:
    responses = []

    for agent in agents:
        name = agent.agent_name
        description = agent.description
        response = f"Agent Name {name}: Description {description}"
        responses.append(response)

    return concat_strings(responses)


def parse_then_send_tasks_to_agents(agents: List[Agent], response: dict):
    # Initialize an empty dictionary to store the output of each agent
    output = []

    # Loop over the tasks in the response
    for call in response["calls"]:
        name = call["agent_name"]
        task = call["task"]

        # Loop over the agents
        for agent in agents:
            # If the agent's name matches the name in the task, run the task
            if agent.agent_name == name:
                out = agent.run(task)
                print(out)

                output.append(f"{name}: {out}")

                # Store the output in the dictionary
                # output[name] = out
                break

    return output


class HierarchicalOrderCall(BaseModel):
    agent_name: str
    task: str


class CallTeam(BaseModel):
    calls: List[HierarchicalOrderCall]


# Example usage:
system_prompt = f"""
You're a director agent, your responsibility is to serve the user efficiently, effectively and skillfully.You have a swarm of agents available to distribute tasks to, interact with the user and then submit tasks to the worker agents. Provide orders to the worker agents that are direct, explicit, and simple. Ensure that they are given tasks that are understandable, actionable, and simple to execute.


######
Workers available:

{agents_list(team)}


"""


# Initialize the function caller
function_caller = OpenAIFunctionCaller(
    system_prompt=system_prompt,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=500,
    temperature=0.5,
    base_model=CallTeam,
)

# Run the function caller
response = function_caller.run(
    "Now let's grow the company! Send an order to the marketing specialist, sales specialist, and product development specialist to improve the company's growth strategies."
)
# print(response)
print(response)
print(type(response))


out = parse_then_send_tasks_to_agents(team, response)
print(out)
