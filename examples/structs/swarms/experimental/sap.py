import asyncio
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
import psutil
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from swarm_models import OpenAIChat

from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

load_dotenv()

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Collection for swarm activity (tasks, responses, messages)
swarm_activity = chroma_client.create_collection(name="swarm_activity")

# Collection for agent capabilities
agent_capabilities = chroma_client.create_collection(name="agent_capabilities")


class Message(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str
    message_type: str  # e.g., "task", "request", "response"
    content: Any
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    conversation_id: Optional[str] = None


class AgentHealth(BaseModel):
    agent_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = "available"  # available, busy, failed
    active_tasks: int = 0
    system_load: float = 0.0  # Placeholder for actual system load


class Swarm:
    def __init__(self, agents: List[Agent], chroma_client: chromadb.Client, api_key: str):
        self.agents = agents
        self.chroma_client = chroma_client
        self.api_key = api_key
        self.health: Dict[str, AgentHealth] = {}
        self.register_agents()
        logger.info(f"Swarm initialized with {len(agents)} agents.")

    def register_agents(self):
        for agent in self.agents:
            self.health[agent.agent_name] = AgentHealth(agent_name=agent.agent_name)
            agent_capabilities.add(
                documents=[agent.system_prompt],
                ids=[agent.agent_name],
                metadatas=[{"agent_name": agent.agent_name}],
            )

    async def find_best_agent(self, task: str) -> Optional[Agent]:
        results = agent_capabilities.query(query_texts=[task], n_results=1)
        if results["ids"] and results["ids"][0]:
            agent_name = results["ids"][0][0]
            agent = next((a for a in self.agents if a.agent_name == agent_name), None)
            if agent:
                logger.info(f"Best agent for task '{task}' is {agent.agent_name}.")
                return agent
        logger.warning(f"No suitable agent found for task: {task}")
        return None

    def post_message(self, message: Message):
        swarm_activity.add(
            documents=[message.model_dump_json()],
            ids=[message.message_id],
            metadatas=[message.model_dump()],  # Store metadata for querying
        )

    def query_messages(self, query: str, message_type: Optional[str] = None, n_results: int = 5) -> List[Message]:
         filter_query = {}
         if message_type:
             filter_query = {"message_type": message_type}
         results = swarm_activity.query(query_texts=[query], n_results=n_results, where=filter_query)

         messages = []
         if results["documents"]:
             for doc in results['documents'][0]: # Because the query returns a list of lists of documents
                 try:
                     messages.append(Message.model_parse_raw(doc))
                 except Exception as e:
                     logger.error(f"Error parsing message document: {e}")

         return messages


    async def run_task(self, task: str, conversation_id: Optional[str] = None) -> Optional[Any]:
        agent = await self.find_best_agent(task)
        if not agent:
            return None

        self.update_agent_health(agent, "busy")

        self.post_message(Message(agent_name="Swarm", message_type="task", content=task, conversation_id=conversation_id))

        try:
            result = await agent.run(task)
            self.post_message(
                Message(
                    agent_name=agent.agent_name,
                    message_type="response",
                    content=result,
                    conversation_id=conversation_id,
                )
            )
            self.update_agent_health(agent, "available")
            return result
        except Exception as e:
            self.update_agent_health(agent, "failed")
            logger.error(f"Agent {agent.agent_name} failed to execute task: {e}")
            return None

    def update_agent_health(self, agent: Agent, status: str):
        health = self.health.get(agent.agent_name)
        if health:
            health.status = status
            health.active_tasks = agent.active_tasks  # Assuming agent tracks active tasks
            health.system_load = psutil.cpu_percent()
            health.timestamp = datetime.utcnow()
            self.post_message(Message(agent_name="Swarm", message_type="health_update", content=health.model_dump()))

    def run(self, task: str, conversation_id: Optional[str] = None) -> Any:
        return asyncio.run(self.run_task(task, conversation_id))



# Initialize the OpenAI model and agents
api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

# Example agent creation
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
)

# Example agents list
agents_list = [agent]

# Create the swarm
swarm = Swarm(
    agents=agents_list, chroma_client=chroma_client, api_key=api_key
)

# Execute tasks asynchronously
task = "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?"
conversation_id = str(uuid.uuid4())  # Create a conversation ID
print(swarm.run(task, conversation_id))


# Example of querying messages related to the conversation:
past_messages = swarm.query_messages(query="", message_type="response", n_results=10)
print(f"Past messages in conversation {conversation_id}: {past_messages}")
