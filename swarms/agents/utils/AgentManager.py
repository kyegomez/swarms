from typing import Dict, Optional
import logging

from celery import Task

from langchain.agents.agent import AgentExecutor
from langchain.callbacks.manager import CallbackManager
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.memory.chat_memory import BaseChatMemory

from swarms.tools.main import BaseToolSet, ToolsFactory
from .AgentBuilder import AgentBuilder
from .Calback import EVALCallbackHandler, ExecutionTracingCallbackHandler


callback_manager_instance = CallbackManager(EVALCallbackHandler())


class AgentManager:
    def __init__(self, toolsets: list[BaseToolSet] = []):
        if not isinstance(toolsets, list):
            raise TypeError("Toolsets must be a list")
        self.toolsets: list[BaseToolSet] = toolsets
        self.memories: Dict[str, BaseChatMemory] = {}
        self.executors: Dict[str, AgentExecutor] = {}

    def create_memory(self) -> BaseChatMemory:
        return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def get_or_create_memory(self, session: str) -> BaseChatMemory:
        if not isinstance(session, str):
            raise TypeError("Session must be a string")
        if not session:
            raise ValueError("Session is empty")
        if not (session in self.memories):
            self.memories[session] = self.create_memory()
        return self.memories[session]

    def create_executor(self, session_id: str, task: Optional[Task] = None, openai_api_key: str = None) -> AgentExecutor:
        try:
            # Create an agent setup with the provided toolsets
            agent_setup = AgentSetup(self.toolsets)

            # Setup the parser for the agent
            agent_setup.setup_parser()

            # Initialize an empty list for callbacks
            callbacks = []

            # Create and setup an evaluation callback, then add it to the callbacks list
            eval_callback = EvaluationCallbackHandler()
            eval_callback.set_parser(agent_setup.get_parser())
            callbacks.append(eval_callback)

            # If a task is provided, create and setup an execution tracing callback, then add it to the callbacks list
            if task:
                execution_trace_callback = ExecutionTracingCallbackHandler(task)
                execution_trace_callback.set_parser(agent_setup.get_parser())
                callbacks.append(execution_trace_callback)

            # Create a callback manager with the callbacks
            callback_manager = CallbackManager(callbacks)

            # Setup the language model with the callback manager and OpenAI API key
            agent_setup.setup_language_model(callback_manager, openai_api_key)

            # Setup the global tools for the agent
            agent_setup.setup_tools()

            # Get or create a memory for the session
            chat_memory = self.get_or_create_memory(session_id)

            # Create a list of tools by combining global tools and per session tools
            tools = [
                *agent_setup.get_global_tools(),
                *ToolsFactory.create_per_session_tools(
                    self.toolsets,
                    get_session=lambda: (session_id, self.executors[session_id]),
                ),
            ]

            # Set the callback manager for each tool
            for tool in tools:
                tool.callback_manager = callback_manager

            # Create an executor from the agent and tools
            executor = AgentExecutor.from_agent_and_tools(
                agent=agent_setup.get_agent(),
                tools=tools,
                memory=chat_memory,
                callback_manager=callback_manager,
                verbose=True,
            )

            # Store the executor in the executors dictionary
            self.executors[session_id] = executor

            # Return the executor
            return executor
        except Exception as e:
            logging.error(f"Error while creating executor: {str(e)}")
            raise e

    @staticmethod
    def create(toolsets: list[BaseToolSet]) -> "AgentManager":
        if not isinstance(toolsets, list):
            raise TypeError("Toolsets must be a list")
        return AgentManager(toolsets=toolsets)