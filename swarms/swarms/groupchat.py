from swarms.agents import SimpleAgent
from termcolor import colored


class GroupChat:
    """
    Groupchat

    Args:
        agents (list): List of agents
        dashboard (bool): Whether to print a dashboard or not

    Example:
    >>> from swarms.structs import Flow
    >>> from swarms.models import OpenAIChat
    >>> from swarms.swarms.groupchat import GroupChat
    >>> from swarms.agents import SimpleAgent
    >>> api_key = ""
    >>> llm = OpenAIChat()
    >>> agent1 = SimpleAgent("Captain Price", Flow(llm=llm, max_loops=4))
    >>> agent2 = SimpleAgent("John Mactavis", Flow(llm=llm, max_loops=4))
    >>> chat = GroupChat([agent1, agent2])
    >>> chat.assign_duty(agent1.name, "Buy the groceries")
    >>> chat.assign_duty(agent2.name, "Clean the house")
    >>> response = chat.run("Captain Price", "Hello, how are you John?")
    >>> print(response)



    """

    def __init__(self, agents, dashboard: bool = False):
        # Ensure that all provided agents are instances of simpleagents
        if not all(isinstance(agent, SimpleAgent) for agent in agents):
            raise ValueError("All agents must be instances of SimpleAgent")
        self.agents = {agent.name: agent for agent in agents}

        # Dictionary to store duties for each agent
        self.duties = {}

        # Dictionary to store roles for each agent
        self.roles = {}

        self.dashboard = dashboard

    def assign_duty(self, agent_name, duty):
        """Assigns duty to the agent"""
        if agent_name not in self.agents:
            raise ValueError(f"No agent named {agent_name} found.")

    def assign_role(self, agent_name, role):
        """Assigns a role to the specified agent"""
        if agent_name not in self.agents:
            raise ValueError(f"No agent named {agent_name} found")

        self.roles[agent_name] = role

    def run(self, sender_name: str, message: str):
        """Runs the groupchat"""
        if self.dashboard:
            metrics = print(
                colored(
                    f"""
            
            Groupchat Configuration:
            ------------------------
                                    
            Agents: {self.agents}
            Message: {message}
            Sender: {sender_name}
            """,
                    "red",
                )
            )

            print(metrics)

        responses = {}
        for agent_name, agent in self.agents.items():
            if agent_name != sender_name:
                if agent_name in self.duties:
                    message += f"Your duty is {self.duties[agent_name]}"
                if agent_name in self.roles:
                    message += (
                        f"You are the {self.roles[agent_name]} in this conversation"
                    )

                responses[agent_name] = agent.run(message)
        return responses
