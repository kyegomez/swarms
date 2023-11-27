from typing import Dict, List, Optional
from dataclass import dataclass

from swarms.models import OpenAI


@dataclass
class OpenAIAssistant:
    name: str = "OpenAI Assistant"
    instructions: str = None
    tools: List[Dict] = None
    model: str = None
    openai_api_key: str = None
    temperature: float = 0.5
    max_tokens: int = 100
    stop: List[str] = None
    echo: bool = False
    stream: bool = False
    log: bool = False
    presence: bool = False
    dashboard: bool = False
    debug: bool = False
    max_loops: int = 5
    stopping_condition: Optional[str] = None
    loop_interval: int = 1
    retry_attempts: int = 3
    retry_interval: int = 1
    interactive: bool = False
    dynamic_temperature: bool = False
    state: Dict = None
    response_filters: List = None
    response_filter: Dict = None
    response_filter_name: str = None
    response_filter_value: str = None
    response_filter_type: str = None
    response_filter_action: str = None
    response_filter_action_value: str = None
    response_filter_action_type: str = None
    response_filter_action_name: str = None
    client = OpenAI()
    role: str = "user"
    instructions: str = None

    def create_assistant(self, task: str):
        assistant = self.client.create_assistant(
            name=self.name,
            instructions=self.instructions,
            tools=self.tools,
            model=self.model,
        )
        return assistant

    def create_thread(self):
        thread = self.client.beta.threads.create()
        return thread

    def add_message_to_thread(self, thread_id: str, message: str):
        message = self.client.beta.threads.add_message(
            thread_id=thread_id, role=self.user, content=message
        )
        return message

    def run(self, task: str):
        run = self.client.beta.threads.runs.create(
            thread_id=self.create_thread().id,
            assistant_id=self.create_assistant().id,
            instructions=self.instructions,
        )

        out = self.client.beta.threads.runs.retrieve(
            thread_id=run.thread_id, run_id=run.id
        )

        return out
