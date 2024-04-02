import logging
from collections import defaultdict
from typing import Callable, Sequence
from swarms import Agent, Anthropic
from swarms.structs.base_swarm import BaseSwarm

# Assuming the existence of an appropriate Agent class and logger setup
class AgentRearrange(BaseSwarm):
    def __init__(
        self,
        agents: Sequence[Agent] = None,
        verbose: bool = False,
        custom_prompt: str = None,
        callbacks: Sequence[Callable] = None,
    ):
        super().__init__()
        if not all(isinstance(agent, Agent) for agent in agents):
            raise ValueError(
                "All elements must be instances of the Agent class."
            )
        self.agents = agents
        self.verbose = verbose
        self.custom_prompt = custom_prompt
        self.callbacks = callbacks if callbacks is not None else []
        self.flows = defaultdict(list)

    def parse_pattern(self, pattern: str):
        """
        Parse the interaction pattern to set up task flows, supporting both sequential
        and concurrent executions within the same pattern.
        """
        try:
            self.flows.clear()  # Ensure flows are reset each time pattern is parsed
            # Split pattern into potentially concurrent flows
            concurrent_flows = pattern.split(",")
            for flow in concurrent_flows:
                # Trim whitespace and identify sequential parts within each concurrent flow
                parts = [part.strip() for part in flow.split("->")]
                if len(parts) > 1:
                    # Link each part sequentially to the next as source -> destination
                    for i in range(len(parts) - 1):
                        source = parts[i]
                        destination = parts[i + 1]
                        # Validate and add each sequential link
                        if source not in [
                            agent.agent_name for agent in self.agents
                        ]:
                            logging.error(
                                f"Source agent {source} not found."
                            )
                            return False
                        if destination not in [
                            agent.agent_name for agent in self.agents
                        ]:
                            logging.error(
                                f"Destination agent {destination} not"
                                " found."
                            )
                            return False
                        self.flows[source].append(destination)
                else:
                    # Handle single agent case if needed
                    self.flows[parts[0]] = []

            return True
        except Exception as e:
            logging.error(f"Error parsing pattern: {e}")
            return False

    def self_find_agent_by_name(self, name: str):
        for agent in self.agents:
            if agent.agent_name == name:
                return agent
        return None

    def agent_exists(self, name: str):
        for agent in self.agents:
            if agent.agent_name == name:
                return True

        return False

    def parse_concurrent_flow(
        self,
        flow: str,
    ):
        sequential_agents = flow.split("->")
        for i, source_name in enumerate(sequential_agents[:-1]):
            destination_name = sequential_agents[i + 1].strip()
            self.parse_sequential_flow(
                source_name.strip(), destination_name
            )

    def parse_sequential_flow(
        self,
        source: str,
        destination: str,
    ):
        if not self.self_find_agent_by_name(
            source
        ) or not self.self_find_agent_by_name(destination):
            return False
        self.flows[source].append(destination)

    def execute_task(
        self,
        dest_agent_name: str,
        source: str,
        task: str,
        specific_tasks: dict,
    ):
        dest_agent = self.self_find_agent_by_name(dest_agent_name)
        if not dest_agent:
            return None
        task_to_run = specific_tasks.get(dest_agent_name, task)
        if self.custom_prompt:
            out = dest_agent.run(
                f"{task_to_run} {self.custom_prompt}"
            )
        else:
            out = dest_agent.run(f"{task_to_run} (from {source})")
        return out

    def process_flows(self, pattern, default_task, specific_tasks):
        if not self.parse_pattern(pattern):
            return None

        results = []
        for source, destinations in self.flows.items():
            if not destinations:
                task = specific_tasks.get(source, default_task)
                source_agent = self.self_find_agent_by_name(source)
                if source_agent:
                    result = source_agent.run(task)
                    results.append(result)
            else:
                for destination in destinations:
                    task = specific_tasks.get(
                        destination, default_task
                    )
                    destination_agent = self.self_find_agent_by_name(
                        destination
                    )
                    if destination_agent:
                        result = destination_agent.run(task)
                        results.append(result)
        return results

    def __call__(
        self,
        pattern: str = None,
        default_task: str = None,
        **specific_tasks,
    ):
        self.flows.clear()  # Reset previous flows
        results = self.process_flows(
            pattern, default_task, specific_tasks
        )
        return results


## Initialize the workflow
agent = Agent(
    agent_name="t",
    agent_description=(
        "Generate a transcript for a youtube video on what swarms"
        " are!"
    ),
    system_prompt=(
        "Generate a transcript for a youtube video on what swarms"
        " are!"
    ),
    llm=Anthropic(),
    max_loops=1,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
)

agent2 = Agent(
    agent_name="t1",
    agent_description=(
        "Generate a transcript for a youtube video on what swarms"
        " are!"
    ),
    llm=Anthropic(),
    max_loops=1,
    system_prompt="Summarize the transcript",
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
)

agent3 = Agent(
    agent_name="t2",
    agent_description=(
        "Generate a transcript for a youtube video on what swarms"
        " are!"
    ),
    llm=Anthropic(),
    max_loops=1,
    system_prompt="Finalize the transcript",
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
)


# Rearrange the agents
rearrange = AgentRearrange(
    agents=[agent, agent2, agent3],
    verbose=True,
    # custom_prompt="Summarize the transcript",
)

# Run the workflow on a task
results = rearrange(
    # pattern="t -> t1, t2 -> t2",
    pattern="t -> t1 -> t2",
    default_task=(
        "Generate a transcript for a YouTube video on what swarms"
        " are!"
    ),
    t="Generate a transcript for a YouTube video on what swarms are!",
    # t2="Summarize the transcript",
    # t3="Finalize the transcript",
)
# print(results)
