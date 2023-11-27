<<<<<<< HEAD
from swarms import OpenAI, Agent
=======
from swarms import OpenAI, Flow
>>>>>>> 3d3dddaf0c7894ec2df14c51f7dd843c41c878c4
from swarms.swarms.groupchat import GroupChatManager, GroupChat


api_key = ""

llm = OpenAI(
    openai_api_key=api_key,
    temperature=0.5,
    max_tokens=3000,
)

# Initialize the flow
<<<<<<< HEAD
flow1 = Agent(
=======
flow1 = Flow(
>>>>>>> 3d3dddaf0c7894ec2df14c51f7dd843c41c878c4
    llm=llm,
    max_loops=1,
    system_message="YOU ARE SILLY, YOU OFFER NOTHING OF VALUE",
    name="silly",
    dashboard=True,
)
<<<<<<< HEAD
flow2 = Agent(
=======
flow2 = Flow(
>>>>>>> 3d3dddaf0c7894ec2df14c51f7dd843c41c878c4
    llm=llm,
    max_loops=1,
    system_message="YOU ARE VERY SMART AND ANSWER RIDDLES",
    name="detective",
    dashboard=True,
)
<<<<<<< HEAD
flow3 = Agent(
=======
flow3 = Flow(
>>>>>>> 3d3dddaf0c7894ec2df14c51f7dd843c41c878c4
    llm=llm,
    max_loops=1,
    system_message="YOU MAKE RIDDLES",
    name="riddler",
    dashboard=True,
)
<<<<<<< HEAD
manager = Agent(
=======
manager = Flow(
>>>>>>> 3d3dddaf0c7894ec2df14c51f7dd843c41c878c4
    llm=llm,
    max_loops=1,
    system_message="YOU ARE A GROUP CHAT MANAGER",
    name="manager",
    dashboard=True,
)


# Example usage:
agents = [flow1, flow2, flow3]

group_chat = GroupChat(agents=agents, messages=[], max_round=10)
chat_manager = GroupChatManager(groupchat=group_chat, selector=manager)
chat_history = chat_manager("Write me a riddle")
