import os
import logging
from typing import Dict, Optional
from celery import Task
from langchain.agents.agent import AgentExecutor
from langchain.callbacks.manager import CallbackManager
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from swarms.agents.tools.main import BaseToolSet, ToolsFactory
from swarms.agents.prompts.prompts import EVAL_PREFIX, EVAL_SUFFIX

from swarms.agents.utils.agent_setup import AgentSetup
# from .callback import EVALCallbackHandler, ExecutionTracingCallbackHandler
from swarms.agents.utils.Calback import EVALCallbackHandler, ExecutionTracingCallbackHandler

callback_manager_instance = CallbackManager(EVALCallbackHandler())

class AgentCreator:
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

    def create_executor(self, session: str, execution: Optional[Task] = None, openai_api_key: str = None) -> AgentExecutor:
        try:
            builder = AgentSetup(self.toolsets)
            builder.setup_parser()

            callbacks = []
            eval_callback = EVALCallbackHandler()
            eval_callback.set_parser(builder.get_parser())
            callbacks.append(eval_callback)

            if execution:
                execution_callback = ExecutionTracingCallbackHandler(execution)
                execution_callback.set_parser(builder.get_parser())
                callbacks.append(execution_callback)

            callback_manager = CallbackManager(callbacks)
            builder.setup_llm(callback_manager, openai_api_key)
            if builder.llm is None:
                raise ValueError('LLM not created')

            builder.setup_global_tools()

            agent = builder.get_agent()
            if not agent:
                raise ValueError("Agent not created")

            memory: BaseChatMemory = self.get_or_create_memory(session)
            tools = [
                *builder.get_global_tools(),
                *ToolsFactory.create_per_session_tools(
                    self.toolsets,
                    get_session=lambda: (session, self.executors[session]),
                ),
            ]

            for tool in tools:
                tool.callback_manager = callback_manager

            executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=tools,
                memory=memory,
                callback_manager=callback_manager,
                verbose=True,
            )

            if 'agent' not in executor.__dict__:
                executor.__dict__['agent'] = agent
            self.executors[session] = executor

            return executor
        except Exception as e:
            logging.error(f"Error while creating executor: {str(e)}")
            raise e

    @staticmethod
    def create(toolsets: list[BaseToolSet]) -> "AgentCreator":
        if not isinstance(toolsets, list):
            raise TypeError("Toolsets must be a list")
        return AgentCreator(toolsets=toolsets)