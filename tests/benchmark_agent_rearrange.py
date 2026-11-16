import argparse
import asyncio
import importlib
import json
import operator
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
from typing import Annotated, Any, Callable, Dict, List, Tuple

Layers = List[List[str]]


class StubModel:
    """A model that waits for a fixed delay and records thread counts."""

    def __init__(self, delay: float):
        self.delay = delay
        self.threads: List[int] = []
        self.calls: List[Tuple[float, float]] = []

    def run(self, task: Any = None, *args: Any, **kwargs: Any) -> str:
        self.threads.append(threading.active_count())
        start = time.perf_counter()
        time.sleep(self.delay)
        self.calls.append((start, time.perf_counter() - start))
        return "done"

    async def arun(
        self, task: Any = None, *args: Any, **kwargs: Any
    ) -> str:
        self.threads.append(threading.active_count())
        start = time.perf_counter()
        await asyncio.sleep(self.delay)
        self.calls.append((start, time.perf_counter() - start))
        return "done"


def chain(n: int) -> Layers:
    """Build n agents that run one after another.

    Args:
        n: Number of agents.

    Returns:
        One single-agent layer per agent.
    """
    return [[f"a{i}"] for i in range(n)]


def fanout(n: int) -> Layers:
    """Build n agents that all run at once.

    Args:
        n: Number of agents.

    Returns:
        A single layer holding every agent.
    """
    return [[f"a{i}" for i in range(n)]]


def diamond(n: int) -> Layers:
    """Build one source, n - 2 parallel workers and one sink.

    Args:
        n: Number of agents.

    Returns:
        Three layers: source, workers, sink.
    """
    workers = [f"a{i}" for i in range(1, n - 1)]
    return [["a0"], workers, [f"a{n - 1}"]]


def layered(n: int, width: int = 8) -> Layers:
    """Build n agents as consecutive parallel layers of a fixed width.

    Args:
        n: Number of agents.
        width: Agents per layer.

    Returns:
        n / width layers of width agents each.
    """
    names = [f"a{i}" for i in range(n)]
    return [names[i : i + width] for i in range(0, n, width)]


LOOP = asyncio.new_event_loop()


def run_async(call: Callable) -> Any:
    """Await a coroutine factory on the benchmark's one event loop.

    Args:
        call: A function returning the coroutine to await.

    Returns:
        The coroutine's result.
    """
    return LOOP.run_until_complete(call())


def model_time(calls: list, layers: Layers) -> float:
    """Sum, over the steps of one run, the fastest model call in each.

    Args:
        calls: The start time and duration of every model call in the run.
        layers: The topology, which gives the width of each step.

    Returns:
        Seconds the run would take with no orchestration cost at all.
    """
    durations = [duration for _, duration in sorted(calls)]
    total, position = 0.0, 0

    for layer in layers:
        step = durations[position : position + len(layer)]
        total += min(step)
        position += len(layer)

    return total


TOPOLOGIES: Dict[str, Callable[[int], Layers]] = {
    "chain": chain,
    "fanout": fanout,
    "diamond": diamond,
    "layered": layered,
}


class SwarmsAdapter:
    """AgentRearrange: the topology is one flow string."""

    module = "swarms"

    def agents(self, model: StubModel, names: List[str]) -> dict:
        from swarms import Agent

        return {
            name: Agent(agent_name=name, llm=model, print_on=False)
            for name in names
        }

    def compose(self, agents: dict, layers: Layers) -> Callable:
        from swarms import AgentRearrange

        flow = " -> ".join(", ".join(layer) for layer in layers)
        self.workflow = AgentRearrange(
            agents=list(agents.values()), flow=flow
        )
        return lambda: self.workflow.run("task")

    def rearrange(self, agents: dict, layers: Layers) -> None:
        flow = " -> ".join(", ".join(layer) for layer in layers)
        self.workflow.set_custom_flow(flow)


class LangGraphAdapter:
    """LangGraph StateGraph whose nodes are plain functions."""

    module = "langgraph.graph"
    state: Any = Annotated[list, operator.add]

    def agents(self, model: StubModel, names: List[str]) -> dict:
        return {name: (lambda _: [model.run()]) for name in names}

    def start(self) -> Any:
        return []

    def compose(self, agents: dict, layers: Layers) -> Callable:
        from langgraph.graph import START, StateGraph

        graph = StateGraph(self.state)
        previous = None
        for layer in layers:
            for name in layer:
                graph.add_node(name, agents[name])
                graph.add_edge(previous or START, name)
            previous = layer if len(layer) > 1 else layer[0]
        app = graph.compile()
        config = {
            "max_concurrency": len(agents),
            "recursion_limit": 100000,
        }
        return lambda: app.invoke(self.start(), config)

    def rearrange(self, agents: dict, layers: Layers) -> None:
        self.compose(agents, layers)


class LangGraphAgentsAdapter(LangGraphAdapter):
    """LangGraph StateGraph whose nodes are prebuilt chat agents."""

    def agents(self, model: StubModel, names: List[str]) -> dict:
        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )
        from langgraph.graph import MessagesState
        from langgraph.prebuilt import create_react_agent

        class StubChat(FakeListChatModel):
            def _call(self, *args: Any, **kwargs: Any) -> str:
                return model.run()

        self.state = MessagesState
        chat = StubChat(responses=["done"])
        return {
            name: create_react_agent(chat, tools=[], name=name)
            for name in names
        }

    def start(self) -> Any:
        return {"messages": [("user", "task")]}


class LangChainAdapter:
    """LangChain expression language: sequences of parallel maps."""

    module = "langchain_core.runnables"

    def agents(self, model: StubModel, names: List[str]) -> dict:
        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )
        from langchain_core.prompts import ChatPromptTemplate

        class StubChat(FakeListChatModel):
            def _call(self, *args: Any, **kwargs: Any) -> str:
                return model.run()

        chat = StubChat(responses=["done"])
        return {
            name: ChatPromptTemplate.from_messages(
                [("system", f"You are {name}."), ("human", "{task}")]
            )
            | chat
            for name in names
        }

    def compose(self, agents: dict, layers: Layers) -> Callable:
        from langchain_core.runnables import (
            RunnableLambda,
            RunnableParallel,
            RunnableSequence,
        )

        def merge(outputs: dict) -> dict:
            answers = (m.content for m in outputs.values())
            return {"task": "\n".join(answers)}

        steps = []
        for layer in layers:
            steps.append(
                RunnableParallel(
                    {name: agents[name] for name in layer}
                )
            )
            steps.append(RunnableLambda(merge))
        pipeline = RunnableSequence(*steps)
        config = {"max_concurrency": len(agents)}
        return lambda: pipeline.invoke({"task": "task"}, config)

    def rearrange(self, agents: dict, layers: Layers) -> None:
        self.compose(agents, layers)


class AutoGenAdapter:
    """AutoGen GraphFlow: a directed graph of assistant agents."""

    module = "autogen_agentchat.teams"

    def agents(self, model: StubModel, names: List[str]) -> dict:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_core.models import CreateResult, RequestUsage
        from autogen_ext.models.replay import (
            ReplayChatCompletionClient,
        )

        class StubClient(ReplayChatCompletionClient):
            async def create(
                self, messages: Any, **kwargs: Any
            ) -> CreateResult:
                return CreateResult(
                    finish_reason="stop",
                    content=await model.arun(),
                    usage=RequestUsage(
                        prompt_tokens=0, completion_tokens=0
                    ),
                    cached=False,
                )

        client = StubClient(["done"])
        return {
            name: AssistantAgent(name, model_client=client)
            for name in names
        }

    def compose(self, agents: dict, layers: Layers) -> Callable:
        from autogen_agentchat.teams import DiGraphBuilder, GraphFlow

        builder = DiGraphBuilder()
        for agent in agents.values():
            builder.add_node(agent)
        for sources, targets in zip(layers, layers[1:]):
            for source in sources:
                for target in targets:
                    builder.add_edge(agents[source], agents[target])
        team = GraphFlow(
            participants=list(agents.values()), graph=builder.build()
        )

        async def go() -> Any:
            await team.reset()
            return await team.run(task="task")

        return lambda: run_async(go)

    def rearrange(self, agents: dict, layers: Layers) -> None:
        self.compose(agents, layers)


class AgnoAdapter:
    """Agno Workflow: a list of steps and parallel blocks."""

    module = "agno.workflow"

    def agents(self, model: StubModel, names: List[str]) -> dict:
        from agno.agent import Agent
        from agno.models.base import Model
        from agno.models.response import ModelResponse

        class StubLLM(Model):
            def invoke(
                self, *args: Any, **kwargs: Any
            ) -> ModelResponse:
                return ModelResponse(content=model.run())

            async def ainvoke(
                self, *args: Any, **kwargs: Any
            ) -> ModelResponse:
                return ModelResponse(content=await model.arun())

            def invoke_stream(
                self, *args: Any, **kwargs: Any
            ) -> None:
                raise NotImplementedError

            async def ainvoke_stream(
                self, *args: Any, **kwargs: Any
            ) -> None:
                raise NotImplementedError

            def _parse_provider_response(
                self, response: Any, **kwargs: Any
            ) -> Any:
                return response

            def _parse_provider_response_delta(
                self, response: Any
            ) -> Any:
                return response

        llm = StubLLM(id="stub")
        return {
            name: Agent(name=name, model=llm, telemetry=False)
            for name in names
        }

    def compose(self, agents: dict, layers: Layers) -> Callable:
        from agno.workflow import Parallel, Step, Workflow

        steps = []
        for index, layer in enumerate(layers):
            block = [
                Step(name=name, agent=agents[name]) for name in layer
            ]
            if len(block) == 1:
                steps.append(block[0])
            else:
                steps.append(Parallel(*block, name=f"layer{index}"))
        workflow = Workflow(
            name="benchmark", steps=steps, telemetry=False
        )
        return lambda: run_async(lambda: workflow.arun("task"))

    def rearrange(self, agents: dict, layers: Layers) -> None:
        self.compose(agents, layers)


class CrewAIAdapter:
    """CrewAI sequential Crew with asynchronous tasks for parallel layers."""

    module = "crewai"

    def agents(self, model: StubModel, names: List[str]) -> dict:
        from crewai import Agent, BaseLLM

        class StubLLM(BaseLLM):
            def call(
                self, messages: Any, *args: Any, **kwargs: Any
            ) -> str:
                return model.run()

        llm = StubLLM(model="stub")
        return {
            name: Agent(
                role=name,
                goal="Answer the task.",
                backstory="A benchmark agent.",
                llm=llm,
            )
            for name in names
        }

    def compose(self, agents: dict, layers: Layers) -> Callable:
        from crewai import Crew, Process, Task

        tasks: list = []
        previous: list = []
        for layer in layers:
            current = [
                Task(
                    description="Answer the task.",
                    expected_output="An answer.",
                    agent=agents[name],
                    async_execution=len(layer) > 1,
                    context=previous or None,
                )
                for name in layer
            ]
            tasks.extend(current)
            previous = current
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.sequential,
        )
        return lambda: crew.kickoff()

    def rearrange(self, agents: dict, layers: Layers) -> None:
        self.compose(agents, layers)


FRAMEWORKS: Dict[str, type] = {
    "swarms": SwarmsAdapter,
    "langgraph": LangGraphAdapter,
    "langgraph-agents": LangGraphAgentsAdapter,
    "langchain": LangChainAdapter,
    "autogen": AutoGenAdapter,
    "agno": AgnoAdapter,
    "crewai": CrewAIAdapter,
}


def timed(call: Callable) -> float:
    """Time one call.

    Args:
        call: The function to run.

    Returns:
        Seconds the call took.
    """
    start = time.perf_counter()
    call()
    return time.perf_counter() - start


def measure(
    framework: str, topology: str, n: int, delay: float, repeat: int
) -> dict:
    """Measure one framework on one topology at one size.

    Args:
        framework: Key into FRAMEWORKS.
        topology: Key into TOPOLOGIES.
        n: Number of agents.
        delay: Seconds each stubbed model call waits.
        repeat: Timed runs to take the median of.

    Returns:
        The measurements as a flat dictionary.
    """
    # Frameworks print panels and warnings, which are not results.
    silence = os.open(os.devnull, os.O_WRONLY)
    os.dup2(silence, 1)
    os.dup2(silence, 2)
    adapter = FRAMEWORKS[framework]()
    importlib.import_module(adapter.module)
    model = StubModel(delay)
    layers = TOPOLOGIES[topology](n)
    names = [name for layer in layers for name in layer]
    other = chain(n) if topology != "chain" else fanout(n)

    # A first build loads what the adapter imports, so memory counts agents only.
    warm = [["a0"], ["a1", "a2"], ["a3"]]
    adapter.compose(
        adapter.agents(model, ["a0", "a1", "a2", "a3"]), warm
    )()

    tracemalloc.start()
    agents = adapter.agents(model, names)
    run = adapter.compose(agents, layers)
    built_kb = tracemalloc.get_traced_memory()[0] / 1024
    run()
    peak_mb = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()

    def build() -> Callable:
        return adapter.compose(adapter.agents(model, names), layers)

    build_ms = timed(build) * 1000
    model.threads.clear()
    walls, ideals = [], []

    for _ in range(repeat):
        model.calls.clear()
        walls.append(timed(run))
        ideals.append(model_time(model.calls, layers))

    overheads = [wall - ideal for wall, ideal in zip(walls, ideals)]
    wall = statistics.median(walls)
    ideal = statistics.median(ideals)
    calls = len(model.threads) / repeat
    threads = max(model.threads)
    try:
        rearrange_ms = 1000 * statistics.median(
            timed(lambda: adapter.rearrange(agents, target))
            for target in [other, layers] * 2
        )
    except Exception:
        # The framework cannot express the other topology.
        rearrange_ms = None
    return {
        "framework": framework,
        "topology": topology,
        "agents": n,
        "depth": len(layers),
        "wall_s": wall,
        "ideal_s": ideal,
        "overhead_ms": statistics.median(overheads) * 1000,
        "build_ms": build_ms,
        "rearrange_ms": rearrange_ms,
        "built_kb_per_agent": built_kb / n,
        "peak_mb": peak_mb,
        "peak_threads": threads,
        "model_calls": calls,
    }


def worker(argv: List[str]) -> None:
    """Run one measurement and write it to a file as JSON.

    Args:
        argv: Framework, topology, size, delay, repeat and output path.
    """
    framework, topology, n, delay, repeat, path = argv
    try:
        result = measure(
            framework, topology, int(n), float(delay), int(repeat)
        )
    except Exception as error:
        result = {
            "framework": framework,
            "topology": topology,
            "agents": int(n),
            "error": f"{type(error).__name__}: {error}"[:300],
        }
    with open(path, "w") as handle:
        json.dump(result, handle)


def run_isolated(
    cell: Tuple[str, str, int], args: argparse.Namespace
) -> dict:
    """Measure one cell in a fresh interpreter.

    Args:
        cell: Framework, topology and size.
        args: Parsed command line options.

    Returns:
        The measurements, or an error entry.
    """
    framework, topology, n = cell
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "result.json")
        command = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            framework,
            topology,
            str(n),
            str(args.delay),
            str(args.repeat),
            path,
        ]
        try:
            subprocess.run(command, timeout=args.timeout)
            with open(path) as handle:
                return json.load(handle)
        except (subprocess.TimeoutExpired, OSError) as error:
            return {
                "framework": framework,
                "topology": topology,
                "agents": n,
                "error": type(error).__name__,
            }


def main() -> None:
    """Run the suite and print one table per topology."""
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(
        description="AgentRearrange against other workflow frameworks."
    )
    parser.add_argument("--sizes", default="8,32,128,512")
    parser.add_argument("--topologies", default=",".join(TOPOLOGIES))
    parser.add_argument("--frameworks", default=",".join(FRAMEWORKS))
    parser.add_argument("--delay", type=float, default=0.01)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    os.environ.setdefault("SWARMS_TELEMETRY_ON", "False")
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    os.environ.setdefault("AGNO_TELEMETRY", "false")

    sizes = [int(size) for size in args.sizes.split(",")]
    frameworks = args.frameworks.split(",")
    results = []

    for topology in args.topologies.split(","):
        for n in sizes:
            for framework in frameworks:
                result = run_isolated((framework, topology, n), args)
                results.append(result)
                print(json.dumps(result), file=sys.stderr, flush=True)

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=1)

    for topology in args.topologies.split(","):
        print(f"\n### {topology}: overhead beyond model time (ms)\n")
        print(f"| agents | {' | '.join(frameworks)} |")
        print("|---" * (len(frameworks) + 1) + "|")
        for n in sizes:
            cells = []
            for framework in frameworks:
                match = next(
                    r
                    for r in results
                    if (r["framework"], r["topology"], r["agents"])
                    == (framework, topology, n)
                )
                cells.append(
                    "n/a"
                    if "error" in match
                    else f"{match['overhead_ms']:.1f}"
                )
            print(f"| {n} | {' | '.join(cells)} |")


if __name__ == "__main__":
    main()
