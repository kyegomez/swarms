import asyncio
import pulsar

from pulsar import ConsumerType
from loguru import logger
from swarms import Agent
from typing import List, Dict, Any
import json


class ScalableAsyncAgentSwarm:
    """
    A scalable, asynchronous swarm of agents leveraging Apache Pulsar for inter-agent communication.
    Provides load balancing, health monitoring, dead letter queues, and centralized logging.
    """

    def __init__(
        self,
        pulsar_url: str,
        topic: str,
        dlq_topic: str,
        agents_config: List[Dict[str, Any]],
    ):
        """
        Initializes the async swarm with agents.

        Args:
            pulsar_url (str): The URL of the Apache Pulsar broker.
            topic (str): The main topic for task distribution.
            dlq_topic (str): The Dead Letter Queue topic for failed messages.
            agents_config (List[Dict[str, Any]]): List of agent configurations with `name`, `description`, and `model_name`.
        """
        self.pulsar_url = pulsar_url
        self.topic = topic
        self.dlq_topic = dlq_topic
        self.agents_config = agents_config
        self.client = pulsar.Client(pulsar_url)
        self.consumer = self.client.subscribe(
            topic,
            subscription_name="swarm-task-sub",
            consumer_type=ConsumerType.Shared,
        )
        self.dlq_producer = self.client.create_producer(dlq_topic)
        self.response_logger = []
        self.agents = [
            self.create_agent(config) for config in agents_config
        ]
        self.agent_index = 0

        logger.info(
            "Swarm initialized with agents: {}",
            [agent["name"] for agent in agents_config],
        )

    def create_agent(
        self, agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Creates a new agent configuration with asynchronous capabilities.

        Args:
            agent_config (Dict[str, Any]): Configuration dictionary with agent details.

        Returns:
            Dict[str, Any]: A dictionary containing agent metadata and functionality.
        """
        agent_name = agent_config["name"]
        description = agent_config["description"]
        model_name = agent_config.get("model_name", "gpt-4o-mini")

        class AsyncAgent:
            """
            An asynchronous agent that processes tasks and communicates via Apache Pulsar.
            """

            def __init__(
                self, name: str, description: str, model_name: str
            ):
                self.name = name
                self.description = description
                self.agent = Agent(
                    agent_name=name,
                    model_name=model_name,
                    max_loops="auto",
                    interactive=True,
                    streaming_on=True,
                )
                logger.info(
                    f"Initialized agent '{name}' - {description}"
                )

            async def process_task(
                self, message: str
            ) -> Dict[str, Any]:
                """
                Processes a single task using the agent.

                Args:
                    message (str): The task message.

                Returns:
                    Dict[str, Any]: JSON-formatted response.
                """
                try:
                    logger.info(
                        f"Agent {self.name} processing task: {message}"
                    )
                    response = await asyncio.to_thread(
                        self.agent.run, message
                    )
                    logger.info(f"Agent {self.name} completed task.")
                    return {
                        "agent_name": self.name,
                        "response": response,
                    }
                except Exception as e:
                    logger.error(
                        f"Agent {self.name} encountered an error: {e}"
                    )
                    return {"agent_name": self.name, "error": str(e)}

        return {
            "name": agent_name,
            "instance": AsyncAgent(
                agent_name, description, model_name
            ),
        }

    async def distribute_task(self, message: str):
        """
        Distributes a task to the next available agent using round-robin.

        Args:
            message (str): The task message.
        """
        agent = self.agents[self.agent_index]
        self.agent_index = (self.agent_index + 1) % len(self.agents)

        try:
            response = await agent["instance"].process_task(message)
            self.log_response(response)
        except Exception as e:
            logger.error(
                f"Error processing task by agent {agent['name']}: {e}"
            )
            self.send_to_dlq(message)

    async def monitor_health(self):
        """
        Periodically monitors the health of agents.
        """
        while True:
            logger.info("Performing health check for all agents.")
            for agent in self.agents:
                logger.info(f"Agent {agent['name']} is online.")
            await asyncio.sleep(10)

    def send_to_dlq(self, message: str):
        """
        Sends a failed message to the Dead Letter Queue (DLQ).

        Args:
            message (str): The message to send to the DLQ.
        """
        try:
            self.dlq_producer.send(message.encode("utf-8"))
            logger.info("Message sent to Dead Letter Queue.")
        except Exception as e:
            logger.error(f"Failed to send message to DLQ: {e}")

    def log_response(self, response: Dict[str, Any]):
        """
        Logs the response to a centralized list for later analysis.

        Args:
            response (Dict[str, Any]): The agent's response.
        """
        self.response_logger.append(response)
        logger.info(f"Response logged: {response}")

    async def listen_and_distribute(self):
        """
        Listens to the main Pulsar topic and distributes tasks to agents.
        """
        while True:
            msg = self.consumer.receive()
            try:
                message = msg.data().decode("utf-8")
                logger.info(f"Received task: {message}")
                await self.distribute_task(message)
                self.consumer.acknowledge(msg)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                self.send_to_dlq(msg.data().decode("utf-8"))
                self.consumer.negative_acknowledge(msg)

    async def run(self):
        """
        Runs the swarm asynchronously with health monitoring and task distribution.
        """
        logger.info("Starting the async swarm...")
        task_listener = asyncio.create_task(
            self.listen_and_distribute()
        )
        health_monitor = asyncio.create_task(self.monitor_health())
        await asyncio.gather(task_listener, health_monitor)

    def shutdown(self):
        """
        Safely shuts down the swarm and logs all responses.
        """
        logger.info("Shutting down the swarm...")
        self.client.close()
        with open("responses.json", "w") as f:
            json.dump(self.response_logger, f, indent=4)
        logger.info("Responses saved to 'responses.json'.")


# from scalable_agent_swarm import ScalableAsyncAgentSwarm  # Assuming your swarm class is saved here

if __name__ == "__main__":
    # Example Configuration
    PULSAR_URL = "pulsar://localhost:6650"
    TOPIC = "stock-analysis"
    DLQ_TOPIC = "stock-analysis-dlq"

    # Agents configuration
    AGENTS_CONFIG = [
        {
            "name": "Stock-Analysis-Agent-1",
            "description": "Analyzes stock trends.",
            "model_name": "gpt-4o-mini",
        },
        {
            "name": "Stock-News-Agent",
            "description": "Summarizes stock news.",
            "model_name": "gpt-4o-mini",
        },
        {
            "name": "Tech-Trends-Agent",
            "description": "Tracks tech sector trends.",
            "model_name": "gpt-4o-mini",
        },
    ]

    # Tasks to send
    TASKS = [
        "Analyze the trend for tech stocks in Q4 2024",
        "Summarize the latest news on the S&P 500",
        "Identify the top-performing sectors in the stock market",
        "Provide a forecast for AI-related stocks for 2025",
    ]

    # Initialize and run the swarm
    swarm = ScalableAsyncAgentSwarm(
        PULSAR_URL, TOPIC, DLQ_TOPIC, AGENTS_CONFIG
    )
    try:
        # Run the swarm in the background
        swarm_task = asyncio.create_task(swarm.run())

        # Send tasks to the topic
        client = pulsar.Client(PULSAR_URL)
        producer = client.create_producer(TOPIC)

        for task in TASKS:
            producer.send(task.encode("utf-8"))
            print(f"Sent task: {task}")

        producer.close()
        client.close()

        # Keep the swarm running
        asyncio.run(swarm_task)
    except KeyboardInterrupt:
        swarm.shutdown()
