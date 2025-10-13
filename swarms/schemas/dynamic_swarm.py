from pydantic import BaseModel
from swarms.tools.base_tool import BaseTool, Field

agents = []


class ConversationEntry(BaseModel):
    agent_name: str = Field(
        description="The name of the agent who made the entry."
    )
    message: str = Field(description="The message sent by the agent.")


class LeaveConversation(BaseModel):
    agent_name: str = Field(
        description="The name of the agent who left the conversation."
    )


class JoinGroupChat(BaseModel):
    agent_name: str = Field(
        description="The name of the agent who joined the conversation."
    )
    group_chat_name: str = Field(
        description="The name of the group chat."
    )
    initial_message: str = Field(
        description="The initial message sent by the agent."
    )


conversation_entry = BaseTool().base_model_to_dict(ConversationEntry)
leave_conversation = BaseTool().base_model_to_dict(LeaveConversation)
join_group_chat = BaseTool().base_model_to_dict(JoinGroupChat)

print(conversation_entry)
print(leave_conversation)
print(join_group_chat)
