import os
import sys
import datetime
from typing import List, Dict, Any, Optional

from swarms import Agent
from swarm_models import OpenAIChat
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

from pulsar import Client, Producer
from pydantic import BaseModel, Field
from loguru import logger
import json

# Configure Loguru logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("swarm_logs.log", rotation="10 MB", level="DEBUG")

# Apache Pulsar configuration
PULSAR_SERVICE_URL = os.getenv(
    "PULSAR_SERVICE_URL", "pulsar://localhost:6650"
)


# Define Pydantic schemas for structured output
class AgentOutputMetadata(BaseModel):
    agent_name: str
    task: str
    timestamp: datetime.datetime
    status: str


class AgentOutputData(BaseModel):
    output: str
    additional_info: Optional[Dict[str, Any]] = None


class AgentOutputSchema(BaseModel):
    metadata: AgentOutputMetadata
    data: AgentOutputData


class SwarmOutputSchema(BaseModel):
    results: List[AgentOutputSchema] = Field(default_factory=list)


# SwarmManager class to manage agents and tasks
class SwarmManager:
    def __init__(
        self,
        agents: List[Agent],
        pulsar_service_url: str = PULSAR_SERVICE_URL,
        topic_prefix: str = "swarm_topic_",  # Prefix for Pulsar topics
    ):
        self.agents = agents
        self.pulsar_service_url = pulsar_service_url
        self.topic_prefix = topic_prefix
        self.client: Optional[Client] = None
        self.producers: Dict[str, Producer] = {}
        self.swarm_results = SwarmOutputSchema()

    def connect_pulsar(self) -> None:
        try:
            self.client = Client(self.pulsar_service_url)
            logger.info(f"Connected to Pulsar service at {self.pulsar_service_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Pulsar service: {e}")
            raise

    def initialize_producers(self) -> None:
        if not self.client:
            raise ConnectionError("Pulsar client is not connected.")

        for agent in self.agents:
            topic = f"{self.topic_prefix}{agent.agent_name}"
            try:
                producer = self.client.create_producer(topic)
                self.producers[agent.agent_name] = producer
                logger.debug(f"Initialized producer for agent {agent.agent_name} on topic {topic}")
            except Exception as e:
                logger.error(f"Failed to create producer for agent {agent.agent_name}: {e}")
                raise

    def run_task(self, agent: Agent, task: str) -> AgentOutputSchema:
        logger.info(f"Agent {agent.agent_name} is starting task: {task}")
        timestamp = datetime.datetime.utcnow()

        try:
            output = agent.run(task)
            status = "Success"
            logger.info(f"Agent {agent.agent_name} completed task successfully.")
        except Exception as e:
            output = str(e)
            status = "Failed"
            logger.error(f"Agent {agent.agent_name} failed to complete task: {e}")

        metadata = AgentOutputMetadata(
            agent_name=agent.agent_name, task=task, timestamp=timestamp, status=status
        )
        data = AgentOutputData(output=output)
        agent_output = AgentOutputSchema(metadata=metadata, data=data)

        try:
            producer = self.producers.get(agent.agent_name)
            if producer:
                producer.send(agent_output.model_dump_json().encode("utf-8")) # Send as JSON string
                logger.debug(f"Published output for agent {agent.agent_name} to Pulsar topic.")
            else:
                logger.warning(f"No producer found for agent {agent.agent_name}. Skipping publish step.")
        except Exception as e:
            logger.error(f"Failed to publish output for agent {agent.agent_name}: {e}")

        return agent_output

    def run(self, task: str) -> SwarmOutputSchema:
        try:
            self.connect_pulsar()
            self.initialize_producers()

            with concurrent.futures.ThreadPoolExecutor() as executor: # Parallel execution
                futures = [executor.submit(self.run_task, agent, task) for agent in self.agents]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        self.swarm_results.results.append(result)
                    except Exception as e:
                        logger.error(f"A task encountered an error: {e}")
                        # Add a result with error information to the SwarmOutputSchema
                        failed_metadata = AgentOutputMetadata(
                            agent_name="Unknown",  # Or some other identifier
                            task=task,
                            timestamp=datetime.datetime.utcnow(),
                            status="Failed"
                        )
                        failed_data = AgentOutputData(output=str(e))
                        failed_result = AgentOutputSchema(metadata=failed_metadata, data=failed_data)
                        self.swarm_results.results.append(failed_result)



            logger.info("Swarm run completed.")
            return self.swarm_results

        except Exception as e:
            logger.error(f"Swarm run encountered an error: {e}")
            raise
        finally:
            if self.client:
                self.client.close()
                logger.info("Pulsar client connection closed.")


# Example usage (similar to before)
if __name__ == "__main__":
    # ... (agent and model initialization)
    swarm = SwarmManager(agents=[agent1, agent2])
    task_description = "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?"
    results = swarm.run(task_description)
    print(results.model_dump_json(indent=4)) # Output JSON
