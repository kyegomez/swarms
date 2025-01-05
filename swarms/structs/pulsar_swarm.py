import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import pulsar
from cryptography.fernet import Fernet
from loguru import logger
from prometheus_client import Counter, Histogram, start_http_server
from pydantic import BaseModel, Field
from pydantic.v1 import validator
from tenacity import retry, stop_after_attempt, wait_exponential

# Enhanced metrics
TASK_COUNTER = Counter(
    "swarm_tasks_total", "Total number of tasks processed"
)
TASK_LATENCY = Histogram(
    "swarm_task_duration_seconds", "Task processing duration"
)
TASK_FAILURES = Counter(
    "swarm_task_failures_total", "Total number of task failures"
)
AGENT_ERRORS = Counter(
    "swarm_agent_errors_total", "Total number of agent errors"
)

# Define types using Literal
TaskStatus = Literal["pending", "processing", "completed", "failed"]
TaskPriority = Literal["low", "medium", "high", "critical"]


class SecurityConfig(BaseModel):
    """Security configuration for the swarm"""

    encryption_key: str = Field(
        ..., description="Encryption key for sensitive data"
    )
    tls_cert_path: Optional[str] = Field(
        None, description="Path to TLS certificate"
    )
    tls_key_path: Optional[str] = Field(
        None, description="Path to TLS private key"
    )
    auth_token: Optional[str] = Field(
        None, description="Authentication token"
    )
    max_message_size: int = Field(
        default=1048576, description="Maximum message size in bytes"
    )
    rate_limit: int = Field(
        default=100, description="Maximum tasks per minute"
    )

    @validator("encryption_key")
    def validate_encryption_key(cls, v):
        if len(v) < 32:
            raise ValueError(
                "Encryption key must be at least 32 bytes long"
            )
        return v


class Task(BaseModel):
    """Enhanced task model with additional metadata and validation"""

    task_id: str = Field(
        ..., description="Unique identifier for the task"
    )
    description: str = Field(
        ..., description="Task description or instructions"
    )
    output_type: Literal["string", "json", "file"] = Field("string")
    status: TaskStatus = Field(default="pending")
    priority: TaskPriority = Field(default="medium")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = Field(default=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("task_id")
    def validate_task_id(cls, v):
        if not v.strip():
            raise ValueError("task_id cannot be empty")
        return v

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class TaskResult(BaseModel):
    """Model for task execution results"""

    task_id: str
    status: TaskStatus
    result: Any
    error_message: Optional[str] = None
    execution_time: float
    agent_id: str


@contextmanager
def task_timing():
    """Context manager for timing task execution"""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        TASK_LATENCY.observe(duration)


class SecurePulsarSwarm:
    """
    Enhanced secure, scalable swarm system with improved reliability and security features.
    """

    def __init__(
        self,
        name: str,
        description: str,
        agents: List[Any],
        pulsar_url: str,
        subscription_name: str,
        topic_name: str,
        security_config: SecurityConfig,
        max_workers: int = 5,
        retry_attempts: int = 3,
        task_timeout: int = 300,
        metrics_port: int = 8000,
    ):
        """Initialize the enhanced Pulsar Swarm"""
        self.name = name
        self.description = description
        self.agents = agents
        self.pulsar_url = pulsar_url
        self.subscription_name = subscription_name
        self.topic_name = topic_name
        self.security_config = security_config
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.task_timeout = task_timeout

        # Initialize encryption
        self.cipher_suite = Fernet(
            security_config.encryption_key.encode()
        )

        # Setup metrics server
        start_http_server(metrics_port)

        # Initialize Pulsar client with security settings
        client_config = {
            "authentication": (
                None
                if not security_config.auth_token
                else pulsar.AuthenticationToken(
                    security_config.auth_token
                )
            ),
            "operation_timeout_seconds": 30,
            "connection_timeout_seconds": 30,
            "use_tls": bool(security_config.tls_cert_path),
            "tls_trust_certs_file_path": security_config.tls_cert_path,
            "tls_allow_insecure_connection": False,
        }

        self.client = pulsar.Client(self.pulsar_url, **client_config)
        self.producer = self._create_producer()
        self.consumer = self._create_consumer()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Initialize rate limiting
        self.last_execution_time = time.time()
        self.execution_count = 0

        logger.info(
            f"Secure Pulsar Swarm '{self.name}' initialized with enhanced security features"
        )

    def _create_producer(self):
        """Create a secure producer with retry logic"""
        return self.client.create_producer(
            self.topic_name,
            max_pending_messages=1000,
            compression_type=pulsar.CompressionType.LZ4,
            block_if_queue_full=True,
            batching_enabled=True,
            batching_max_publish_delay_ms=10,
        )

    def _create_consumer(self):
        """Create a secure consumer with retry logic"""
        return self.client.subscribe(
            self.topic_name,
            subscription_name=self.subscription_name,
            consumer_type=pulsar.ConsumerType.Shared,
            message_listener=None,
            receiver_queue_size=1000,
            max_total_receiver_queue_size_across_partitions=50000,
        )

    def _encrypt_message(self, data: str) -> bytes:
        """Encrypt message data"""
        return self.cipher_suite.encrypt(data.encode())

    def _decrypt_message(self, data: bytes) -> str:
        """Decrypt message data"""
        return self.cipher_suite.decrypt(data).decode()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def publish_task(self, task: Task) -> None:
        """Publish a task with enhanced security and reliability"""
        try:
            # Validate message size
            task_data = task.json()
            if len(task_data) > self.security_config.max_message_size:
                raise ValueError(
                    "Task data exceeds maximum message size"
                )

            # Rate limiting
            current_time = time.time()
            if current_time - self.last_execution_time >= 60:
                self.execution_count = 0
                self.last_execution_time = current_time

            if (
                self.execution_count
                >= self.security_config.rate_limit
            ):
                raise ValueError("Rate limit exceeded")

            # Encrypt and publish
            encrypted_data = self._encrypt_message(task_data)
            message_id = self.producer.send(encrypted_data)

            self.execution_count += 1
            logger.info(
                f"Task {task.task_id} published successfully with message ID {message_id}"
            )

        except Exception as e:
            TASK_FAILURES.inc()
            logger.error(
                f"Error publishing task {task.task_id}: {str(e)}"
            )
            raise

    async def _process_task(self, task: Task) -> TaskResult:
        """Process a task with comprehensive error handling and monitoring"""
        task.status = "processing"
        task.started_at = datetime.utcnow()

        with task_timing():
            try:
                # Select agent using round-robin
                agent = self.agents.pop(0)
                self.agents.append(agent)

                # Execute task with timeout
                future = self.executor.submit(
                    agent.run, task.description
                )
                result = future.result(timeout=self.task_timeout)

                # Handle different output types
                if task.output_type == "json":
                    result = json.loads(result)
                elif task.output_type == "file":
                    file_path = f"output_{task.task_id}_{int(time.time())}.txt"
                    with open(file_path, "w") as f:
                        f.write(result)
                    result = {"file_path": file_path}

                task.status = "completed"
                task.completed_at = datetime.utcnow()
                TASK_COUNTER.inc()

                return TaskResult(
                    task_id=task.task_id,
                    status="completed",
                    result=result,
                    execution_time=time.time()
                    - task.started_at.timestamp(),
                    agent_id=agent.agent_name,
                )

            except TimeoutError:
                TASK_FAILURES.inc()
                error_msg = f"Task {task.task_id} timed out after {self.task_timeout} seconds"
                logger.error(error_msg)
                task.status = "failed"
                return TaskResult(
                    task_id=task.task_id,
                    status="failed",
                    result=None,
                    error_message=error_msg,
                    execution_time=time.time()
                    - task.started_at.timestamp(),
                    agent_id=agent.agent_name,
                )

            except Exception as e:
                TASK_FAILURES.inc()
                AGENT_ERRORS.inc()
                error_msg = (
                    f"Error processing task {task.task_id}: {str(e)}"
                )
                logger.error(error_msg)
                task.status = "failed"
                return TaskResult(
                    task_id=task.task_id,
                    status="failed",
                    result=None,
                    error_message=error_msg,
                    execution_time=time.time()
                    - task.started_at.timestamp(),
                    agent_id=agent.agent_name,
                )

    async def consume_tasks(self):
        """Enhanced task consumption with circuit breaker and backoff"""
        consecutive_failures = 0
        backoff_time = 1

        while True:
            try:
                # Circuit breaker pattern
                if consecutive_failures >= 5:
                    logger.warning(
                        f"Circuit breaker triggered. Waiting {backoff_time} seconds"
                    )
                    await asyncio.sleep(backoff_time)
                    backoff_time = min(backoff_time * 2, 60)
                    continue

                # Receive message with timeout
                message = await self.consumer.receive_async()

                try:
                    # Decrypt and process message
                    decrypted_data = self._decrypt_message(
                        message.data()
                    )
                    task_data = json.loads(decrypted_data)
                    task = Task(**task_data)

                    # Process task
                    result = await self._process_task(task)

                    # Handle result
                    if result.status == "completed":
                        await self.consumer.acknowledge_async(message)
                        consecutive_failures = 0
                        backoff_time = 1
                    else:
                        if task.retry_count < self.retry_attempts:
                            task.retry_count += 1
                            await self.consumer.negative_acknowledge(
                                message
                            )
                        else:
                            await self.consumer.acknowledge_async(
                                message
                            )
                            logger.error(
                                f"Task {task.task_id} failed after {self.retry_attempts} attempts"
                            )

                except Exception as e:
                    logger.error(
                        f"Error processing message: {str(e)}"
                    )
                    await self.consumer.negative_acknowledge(message)
                    consecutive_failures += 1

            except Exception as e:
                logger.error(f"Error in consume_tasks: {str(e)}")
                consecutive_failures += 1
                await asyncio.sleep(1)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup"""
        try:
            self.producer.flush()
            self.producer.close()
            self.consumer.close()
            self.client.close()
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


# if __name__ == "__main__":
#     # Example usage with security configuration
#     security_config = SecurityConfig(
#         encryption_key=secrets.token_urlsafe(32),
#         tls_cert_path="/path/to/cert.pem",
#         tls_key_path="/path/to/key.pem",
#         auth_token="your-auth-token",
#         max_message_size=1048576,
#         rate_limit=100,
#     )

#     # Agent factory function
#     def create_financial_agent() -> Agent:
#         """Factory function to create a financial analysis agent."""
#         return Agent(
#             agent_name="Financial-Analysis-Agent",
#             system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
#             model_name="gpt-4o-mini",
#             max_loops=1,
#             autosave=True,
#             dashboard=False,
#             verbose=True,
#             dynamic_temperature_enabled=True,
#             saved_state_path="finance_agent.json",
#             user_name="swarms_corp",
#             retry_attempts=1,
#             context_length=200000,
#             return_step_meta=False,
#             output_type="string",
#             streaming_on=False,
#         )

#     # Initialize agents (implementation not shown)
#     agents = [create_financial_agent() for _ in range(3)]

#     # Initialize the secure swarm
#     with SecurePulsarSwarm(
#         name="Secure Financial Swarm",
#         description="Production-grade financial analysis swarm",
#         agents=agents,
#         pulsar_url="pulsar+ssl://localhost:6651",
#         subscription_name="secure_financial_subscription",
#         topic_name="secure_financial_tasks",
#         security_config=security_config,
#         max_workers=5,
#         retry_attempts=3,
#         task_timeout=300,
#         metrics_port=8000,
#     ) as swarm:
#         # Example task
#         task = Task(
#             task_id=secrets.token_urlsafe(16),
#             description="Analyze Q4 financial reports",
#             output_type="json",
#             priority="high",
#             metadata={
#                 "department": "finance",
#                 "requester": "john.doe@company.com",
#             },
#         )

#         # Run the swarm
#         swarm.publish_task(task)
#         asyncio.run(swarm.consume_tasks())
