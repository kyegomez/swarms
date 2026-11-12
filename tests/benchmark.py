import argparse
import asyncio
import importlib
import operator
import os
import statistics
import subprocess
import sys
import threading
import time
import tracemalloc
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from functools import partial
from typing import Annotated, Callable


class StubModel:
    def __init__(self, delay: float, threads: list):
        self.delay = delay
        self.threads = threads

    def run(self, task=None, *args, **kwargs) -> str:
        self.threads.append(threading.active_count())
        time.sleep(self.delay)
        return "done"

    async def arun(self, task=None, *args, **kwargs) -> str:
        self.threads.append(threading.active_count())
        await asyncio.sleep(self.delay)
        return "done"


def timed(call) -> float:
    start = time.perf_counter()
    call()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Framework overhead with a stubbed model."
    )
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    os.environ.setdefault("SWARMS_TELEMETRY_ON", "False")
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    os.environ.setdefault("AGNO_TELEMETRY", "false")
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

    columns = []
    for name in FRAMEWORKS:
        with ProcessPoolExecutor(1) as pool:
            try:
                column = pool.submit(measure, name, args).result()
            except ModuleNotFoundError:
                column = [(r, "not installed") for r, _ in columns[0]]
            columns.append(column)

    print(f"| benchmark | {' | '.join(FRAMEWORKS)} |")
    print("|---" * (len(FRAMEWORKS) + 1) + "|")
    for cells in zip(*columns):
        values = " | ".join(str(value) for _, value in cells)
        print(f"| {cells[0][0]} | {values} |")


def measure(name: str, args: argparse.Namespace) -> list:
    module, fanout = FRAMEWORKS[name]
    # Frameworks print panels and warnings, which would land in the table.
    silence = os.open(os.devnull, os.O_WRONLY)
    os.dup2(silence, 1)
    os.dup2(silence, 2)
    threads: list = []
    model = StubModel(args.delay, threads)
    build = partial(fanout, model, args.agents)
    importlib.import_module(module)

    def median(call) -> float:
        return statistics.median(
            timed(call) for _ in range(args.repeat)
        )

    command = [sys.executable, "-c", f"import {module}"]
    import_s = median(lambda: subprocess.run(command, check=True))
    # A first build loads what the adapter imports, so memory counts agents only.
    fanout(model, 1)
    tracemalloc.start()
    run = build()
    agent_kb = tracemalloc.get_traced_memory()[0] / 1024 / args.agents
    run()
    peak_mb = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    build_ms = timed(build) * 1000 / args.agents
    wall = median(run)

    rows = [
        ("cold import (s)", import_s),
        ("construction per agent, workflow included (ms)", build_ms),
        ("memory per agent, workflow included (KB)", agent_kb),
        (f"{args.agents}-agent fan-out wall (s)", wall),
        ("fan-out wall minus model delay (s)", wall - args.delay),
        ("fan-out peak memory, agents included (MB)", peak_mb),
        ("fan-out peak threads", max(threads)),
        (
            "model calls per fan-out",
            len(threads) // (args.repeat + 1),
        ),
    ]
    return [(row, round(value, 3)) for row, value in rows]


def swarms_fanout(model: StubModel, agents: int) -> Callable:
    from swarms import Agent, ConcurrentWorkflow

    workflow = ConcurrentWorkflow(
        agents=[
            Agent(agent_name=f"agent-{i}", llm=model, print_on=False)
            for i in range(agents)
        ],
        on_error="raise",
    )
    return lambda: workflow.run("task")


def langgraph_fanout(model: StubModel, agents: int) -> Callable:
    from langgraph.graph import START, StateGraph

    graph = StateGraph(Annotated[list, operator.add])
    for i in range(agents):
        graph.add_node(f"agent-{i}", lambda _: [model.run()])
        graph.add_edge(START, f"agent-{i}")
    app = graph.compile()
    return lambda: app.invoke([], {"max_concurrency": agents})


def gathered(calls: list) -> Callable:
    """Return a call that awaits every coroutine factory at once."""

    async def main() -> list:
        return await asyncio.gather(*(call() for call in calls))

    return lambda: asyncio.run(main())


def langchain_fanout(model: StubModel, agents: int) -> Callable:
    """Build one prompt and chat model chain per agent, run in parallel."""
    from langchain_core.language_models.fake_chat_models import (
        FakeListChatModel,
    )
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel

    class StubChat(FakeListChatModel):
        def _call(self, *args, **kwargs) -> str:
            return model.run()

    chains = {
        f"agent-{i}": ChatPromptTemplate.from_messages(
            [("system", f"You are agent-{i}."), ("human", "{task}")]
        )
        | StubChat(responses=["done"])
        for i in range(agents)
    }
    fan_out = RunnableParallel(chains)
    config = {"max_concurrency": agents}
    return lambda: fan_out.invoke({"task": "task"}, config)


def crewai_fanout(model: StubModel, agents: int) -> Callable:
    """Build CrewAI agents and kick each off on its own thread."""
    from crewai import Agent, BaseLLM

    class StubLLM(BaseLLM):
        def call(self, messages, *args, **kwargs) -> str:
            return model.run()

    llm = StubLLM(model="stub")
    crew = [
        Agent(
            role=f"agent-{i}",
            goal="Answer the task.",
            backstory="A benchmark agent.",
            llm=llm,
        )
        for i in range(agents)
    ]
    # A Crew allows one async task at its end, so threads do the fan-out.
    pool = ThreadPoolExecutor(agents)
    return lambda: list(pool.map(lambda a: a.kickoff("task"), crew))


def autogen_fanout(model: StubModel, agents: int) -> Callable:
    """Build AutoGen assistant agents and await every run at once."""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core.models import CreateResult, RequestUsage
    from autogen_ext.models.replay import ReplayChatCompletionClient

    class StubClient(ReplayChatCompletionClient):
        async def create(self, messages, **kwargs) -> CreateResult:
            return CreateResult(
                finish_reason="stop",
                content=await model.arun(),
                usage=RequestUsage(
                    prompt_tokens=0, completion_tokens=0
                ),
                cached=False,
            )

    client = StubClient(["done"])
    team = [
        AssistantAgent(f"agent_{i}", model_client=client)
        for i in range(agents)
    ]
    return gathered([lambda a=a: a.run(task="task") for a in team])


def openai_agents_fanout(model: StubModel, agents: int) -> Callable:
    """Build OpenAI Agents SDK agents and await every run at once."""
    from agents import (
        Agent,
        Model,
        ModelResponse,
        Runner,
        Usage,
        set_tracing_disabled,
    )
    from openai.types.responses import (
        ResponseOutputMessage,
        ResponseOutputText,
    )

    set_tracing_disabled(True)

    class StubClient(Model):
        async def get_response(
            self, *args, **kwargs
        ) -> ModelResponse:
            text = ResponseOutputText(
                text=await model.arun(),
                type="output_text",
                annotations=[],
            )
            message = ResponseOutputMessage(
                id="stub",
                content=[text],
                role="assistant",
                status="completed",
                type="message",
            )
            return ModelResponse(
                output=[message], usage=Usage(), response_id=None
            )

        def stream_response(self, *args, **kwargs) -> None:
            raise NotImplementedError

    client = StubClient()
    team = [
        Agent(name=f"agent-{i}", model=client) for i in range(agents)
    ]
    return gathered([lambda a=a: Runner.run(a, "task") for a in team])


def pydantic_ai_fanout(model: StubModel, agents: int) -> Callable:
    """Build Pydantic AI agents and await every run at once."""
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    async def reply(messages, info) -> ModelResponse:
        return ModelResponse(parts=[TextPart(await model.arun())])

    team = [
        Agent(FunctionModel(reply), name=f"agent-{i}")
        for i in range(agents)
    ]
    return gathered([lambda a=a: a.run("task") for a in team])


def llama_index_fanout(model: StubModel, agents: int) -> Callable:
    """Build LlamaIndex ReAct agents and await every run at once."""
    from llama_index.core.agent.workflow import ReActAgent
    from llama_index.core.llms import (
        ChatMessage,
        ChatResponse,
        MockLLM,
    )

    class StubLLM(MockLLM):
        async def astream_chat(self, messages, **kwargs):
            # ReActAgent only accepts a reply in its Thought/Answer format.
            text = (
                "Thought: I can answer.\nAnswer: "
                + await model.arun()
            )
            message = ChatMessage(role="assistant", content=text)

            async def stream():
                yield ChatResponse(message=message, delta=text)

            return stream()

    llm = StubLLM()
    team = [
        ReActAgent(name=f"agent-{i}", llm=llm, tools=[])
        for i in range(agents)
    ]

    async def run(agent: ReActAgent):
        return await agent.run("task")

    return gathered([lambda a=a: run(a) for a in team])


def agno_fanout(model: StubModel, agents: int) -> Callable:
    """Build Agno agents and await every run at once."""
    from agno.agent import Agent
    from agno.models.base import Model
    from agno.models.response import ModelResponse

    class StubLLM(Model):
        def invoke(self, *args, **kwargs) -> ModelResponse:
            return ModelResponse(content=model.run())

        async def ainvoke(self, *args, **kwargs) -> ModelResponse:
            return ModelResponse(content=await model.arun())

        def invoke_stream(self, *args, **kwargs) -> None:
            raise NotImplementedError

        async def ainvoke_stream(self, *args, **kwargs) -> None:
            raise NotImplementedError

        def _parse_provider_response(self, response, **kwargs):
            return response

        def _parse_provider_response_delta(self, response):
            return response

    llm = StubLLM(id="stub")
    team = [
        Agent(name=f"agent-{i}", model=llm, telemetry=False)
        for i in range(agents)
    ]
    return gathered([lambda a=a: a.arun("task") for a in team])


FRAMEWORKS = {
    "swarms": ("swarms", swarms_fanout),
    "langgraph": ("langgraph.graph", langgraph_fanout),
    "langchain": ("langchain_core.runnables", langchain_fanout),
    "crewai": ("crewai", crewai_fanout),
    "autogen": ("autogen_agentchat.agents", autogen_fanout),
    # agents.run, since tests/agents shadows a bare agents when it is absent.
    "openai-agents": ("agents.run", openai_agents_fanout),
    "pydantic-ai": ("pydantic_ai", pydantic_ai_fanout),
    "llamaindex": (
        "llama_index.core.agent.workflow",
        llama_index_fanout,
    ),
    "agno": ("agno.agent", agno_fanout),
}


if __name__ == "__main__":
    main()
