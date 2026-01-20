"""
GroupChat Speaker Function Examples

This example demonstrates how to use different speaker functions in the GroupChat:
- Round Robin: Agents speak in a fixed order, cycling through the list
- Random: Agents speak in random order
- Priority: Agents speak based on priority weights
- Custom: User-defined speaker functions

The example also shows how agents can mention each other using @agent_name syntax.
"""

from swarms import Agent
from swarms.structs.groupchat import GroupChat, random_speaker


def create_example_agents():
    """Create example agents for demonstration."""

    # Create agents with different expertise
    analyst = Agent(
        agent_name="analyst",
        system_prompt="You are a data analyst. You excel at analyzing data, creating charts, and providing insights.",
        model_name="claude-3-5-sonnet-20240620",
        streaming_on=True,
        print_on=True,
    )

    researcher = Agent(
        agent_name="researcher",
        system_prompt="You are a research specialist. You are great at gathering information, fact-checking, and providing detailed research.",
        model_name="claude-3-5-sonnet-20240620",
        streaming_on=True,
        print_on=True,
    )

    writer = Agent(
        agent_name="writer",
        system_prompt="You are a content writer. You excel at writing clear, engaging content and summarizing information.",
        model_name="claude-3-5-sonnet-20240620",
        streaming_on=True,
        print_on=True,
    )

    return [analyst, researcher, writer]


def example_random():
    agents = create_example_agents()

    # Create group chat with random speaker function
    group_chat = GroupChat(
        name="Random Team",
        description="A team that speaks in random order",
        agents=agents,
        speaker_function=random_speaker,
        interactive=False,
    )

    # Test the random behavior
    task = "Let's create a marketing strategy for a personal healthcare ai consumer assistant app. @analyst @researcher @writer please contribute."

    response = group_chat.run(task)
    print(f"Response:\n{response}\n")


if __name__ == "__main__":
    # example_round_robin()
    example_random()
