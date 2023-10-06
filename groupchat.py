from swarms.swarms.groupchat import GroupChat
from langchain.llms import OpenAIChat
from swarms import Worker

llm = OpenAIChat(
    model_name='gpt-4',
    openai_api_key="api-key",
    temperature=0.5
)

Worker1 = Worker(
    llm=llm,
    ai_name="Optimus Prime",
    ai_role="Worker in a swarm",
    external_tools=None,
    human_in_the_loop=False,
    temperature=0.5,
)

Worker2 = Worker(
    llm=llm,
    ai_name="Optimus Prime",
    ai_role="Worker in a swarm",
    external_tools=None,
    human_in_the_loop=False,
    temperature=0.5,
)

workers = [Worker1, Worker2]
messages = [
    {
        "role": "system",
        "context": f"Read the above conversation. Then select the next role from {self.agent_names} to play. Only return the role.",
    }
]

agent = GroupChat(
    workers,
    messages,
    name="groupchat",
    max_consecutive_auto_reply=1,
    human_input_mode="NEVER",
    system_message="Group chat manager",
)

agent.run(messages, workers)
