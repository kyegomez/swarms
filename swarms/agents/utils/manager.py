from typing import Dict, Optional
from celery import Task

from langchain.agents.agent import AgentExecutor
from langchain.callbacks.base import CallbackManager
from langchain.callbacks import set_handler
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.memory.chat_memory import BaseChatMemory

from swarms.tools.main import BaseToolSet, ToolsFactory

from .builder import AgentBuilder
from .callback import EVALCallbackHandler, ExecutionTracingCallbackHandler


set_handler(EVALCallbackHandler())


class AgentManager:
    def __init__(
        self,
        toolsets: list[BaseToolSet] = [],
    ):
        self.toolsets: list[BaseToolSet] = toolsets
        self.memories: Dict[str, BaseChatMemory] = {}
        self.executors: Dict[str, AgentExecutor] = {}

    def create_memory(self) -> BaseChatMemory:
        return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def get_or_create_memory(self, session: str) -> BaseChatMemory:
        if not (session in self.memories):
            self.memories[session] = self.create_memory()
        return self.memories[session]

    def create_executor(
        self, session: str, execution: Optional[Task] = None
    ) -> AgentExecutor:
        builder = AgentBuilder(self.toolsets)
        builder.build_parser()

        callbacks = []
        eval_callback = EVALCallbackHandler()
        eval_callback.set_parser(builder.get_parser())
        callbacks.append(eval_callback)
        if execution:
            execution_callback = ExecutionTracingCallbackHandler(execution)
            execution_callback.set_parser(builder.get_parser())
            callbacks.append(execution_callback)

        callback_manager = CallbackManager(callbacks)

        builder.build_llm(callback_manager)
        builder.build_global_tools()

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
            agent=builder.get_agent(),
            tools=tools,
            memory=memory,
            callback_manager=callback_manager,
            verbose=True,
        )
        self.executors[session] = executor
        return executor

    @staticmethod
    def create(toolsets: list[BaseToolSet]) -> "AgentManager":
        return AgentManager(
            toolsets=toolsets,
        )