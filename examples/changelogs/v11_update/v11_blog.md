# Swarms v11: Release Notes

**Release period:** March 13 – April 19, 2026

Multi-agent systems are only as good as the architectures that organize them. For the past six weeks, that's been the central question driving v11: what new coordination patterns make agents more capable, more reliable, and easier to build with?

The answer came in waves. First, new swarm architectures — AdvisorSwarm, PlannerGeneratorEvaluator, PlannerWorkerSwarm, DriftDetectionAgent, and two new HeavySwarm modes — each solving a distinct class of problem: cost-efficient reasoning, iterative self-correction, parallel decomposition, output drift monitoring, and deep multi-domain analysis. Then infrastructure: GraphWorkflow gained checkpoint resumption and topology serialization so long-running workflows survive interruptions. The CLI grew a `swarms init` command and a full autoswarm file writer so you can go from a prompt to a running Python script without touching a config file. Then security: a full audit pass closed SSRF vectors, shell injection paths, unsafe YAML deserialization, and weak file permissions. And finally, the codebase audit: eleven structs and utilities that had been sitting unmaintained — `BoardOfDirectors`, `OpenAIAssistant`, `GRPO`, `MAKER`, and others — were deleted entirely, reducing the surface area the team has to maintain going forward.

v11 is the largest release since v10. What follows is the full account of what changed, why, and how to use it.

---

## New Features

### AdvisorSwarm

AdvisorSwarm implements Anthropic's advisor strategy: a cheaper executor model drives the task end-to-end while a powerful advisor model is consulted on-demand between turns. Both agents share the same conversation context — the advisor never produces user-facing output, it only shapes what the executor does next. You control how many advisory calls are allowed via `max_advisor_uses`.

This pattern gets you near-Opus quality reasoning at Sonnet prices. The executor handles the repetitive turns; the advisor steps in at inflection points to provide strategic course-corrections.

```python
from swarms import AdvisorSwarm

swarm = AdvisorSwarm(
    name="Code Advisor",
    description="Advisor-guided code generation",
    executor_model_name="claude-sonnet-4-6",
    advisor_model_name="claude-opus-4-6",
    max_advisor_uses=3,  # 1 plan + up to 2 review-refine cycles
    max_loops=1,
    verbose=True,
)

result = swarm.run(
    "Write a Python function that implements binary search on a sorted list. "
    "Include proper error handling, type hints, and edge cases."
)

print(result)
```

---

### PlannerGeneratorEvaluator

A three-agent orchestration harness inspired by GAN-style adversarial feedback loops. The **Planner** expands your prompt into a concrete step-by-step plan. The **Generator** executes each step. The **Evaluator** scores each output against the plan's success criteria — if a step falls below threshold, the Generator retries with the evaluator's critique as context. You get a score trajectory, retry counts, and a shared state file per run.

This pattern works well for tasks where correctness is verifiable: writing that must hit specific points, code that must pass criteria, research that must cover defined areas.

```python
from swarms import PlannerGeneratorEvaluator

harness = PlannerGeneratorEvaluator(
    model_name="gpt-4.1",
    max_steps=3,
    max_retries_per_step=2,
    output_type="final",
    verbose=True,
)

result = harness.run(
    "Write a comprehensive guide on the benefits and risks of intermittent fasting"
)

print(result)
print(f"Steps completed: {harness.last_result.total_steps_completed}")
print(f"Total retries:   {harness.last_result.total_retries}")
print(f"Duration:        {harness.last_result.total_duration:.1f}s")
print(f"Shared state:    {harness.last_result.output_path}")
```

---

### PlannerWorkerSwarm

A parallel execution harness where a Planner model decomposes a task into subtasks, dispatches them concurrently to a pool of specialized worker agents, and a Judge model synthesizes the results. Workers run in a thread pool with configurable concurrency and per-worker timeouts. The Planner assigns tasks by matching each subtask to the most appropriate worker based on agent descriptions.

Use this when you have a large task that can be decomposed and you want multiple specialists running in parallel instead of sequentially.

```python
from swarms import Agent
from swarms.structs.planner_worker_swarm import PlannerWorkerSwarm

workers = [
    Agent(
        agent_name="Research-Agent",
        agent_description="Gathers factual information and data points",
        system_prompt="You are a research specialist. Provide thorough, factual information.",
        model_name="gpt-5.4",
        max_loops=1,
    ),
    Agent(
        agent_name="Analysis-Agent",
        agent_description="Analyzes data and identifies patterns",
        system_prompt="You are an analysis specialist. Identify patterns and draw structured conclusions.",
        model_name="gpt-5.4",
        max_loops=1,
    ),
    Agent(
        agent_name="Strategy-Agent",
        agent_description="Evaluates strategic implications and recommendations",
        system_prompt="You are a strategy specialist. Provide actionable recommendations.",
        model_name="gpt-5.4",
        max_loops=1,
    ),
]

swarm = PlannerWorkerSwarm(
    name="Market-Research-Swarm",
    description="Conducts market research through parallel agent collaboration",
    agents=workers,
    planner_model_name="gpt-5.4",
    judge_model_name="gpt-5.4",
    max_workers=5,
    worker_timeout=120,
)

result = swarm.run(
    "Research the current state of the electric vehicle market: "
    "top manufacturers, key technology trends, adoption challenges, and a 5-year outlook."
)
print(result)
```

---

### DriftDetectionAgent

An agent that monitors a SequentialWorkflow for drift — when outputs start diverging from the original task intent. The agent runs in a dedicated loop between workflow steps, scores each output against the original goal, and triggers a correction pass when the score falls below a configurable threshold. Useful for long-running pipelines where model behavior can degrade over many turns.

```python
from swarms import Agent, SequentialWorkflow
from swarms.structs.sequential_workflow import SequentialWorkflow

researcher = Agent(
    agent_name="Researcher",
    system_prompt="You are a research agent. Stay tightly focused on the assigned topic.",
    model_name="gpt-5.4",
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    system_prompt="You are a technical writer. Summarize research into clear prose.",
    model_name="gpt-5.4",
    max_loops=1,
)

workflow = SequentialWorkflow(
    name="Research-Pipeline",
    agents=[researcher, writer],
    max_loops=3,
    drift_detection_enabled=True,
    drift_threshold=0.7,
    drift_model_name="gpt-5.4",
)

result = workflow.run("Explain the architecture of transformer models")
print(result)
```

---

### HeavySwarm: Grok 4.20 Architecture Modes

HeavySwarm now ships two Grok-native agent architectures that decompose tasks by **thinking style** rather than task phase.

**`use_grok_agents=True`** — A 4-agent architecture: Captain Swarm (orchestrator + synthesizer with conflict mediation), Harper (research & evidence), Benjamin (logic, math & code), and Lucas (creative & contrarian analysis).

**`use_grok_heavy=True`** — A 16-agent architecture: a Grok captain plus 15 named domain specialists for the deepest multi-domain analysis available. The two modes are mutually exclusive.

```python
from swarms import HeavySwarm

# 4-agent Grok mode
swarm = HeavySwarm(
    worker_model_name="grok-4",
    question_agent_model_name="grok-4",
    use_grok_agents=True,
    loops_per_agent=1,
)

result = swarm.run("Analyze the long-term macroeconomic effects of universal basic income")
print(result)
```

```python
from swarms import HeavySwarm

# 16-agent Grok Heavy mode
swarm = HeavySwarm(
    worker_model_name="grok-4",
    question_agent_model_name="grok-4",
    use_grok_heavy=True,
    show_dashboard=True,
    loops_per_agent=1,
)

result = swarm.run(
    "What are the most transformative technologies that will reshape "
    "civilization over the next 50 years? Analyze from all relevant dimensions."
)
print(result)
```

---

### GraphWorkflow Persistence: Checkpointing and Serialization

Two new persistence features for long-running GraphWorkflows.

**Checkpointing** saves workflow state to disk after each loop iteration. If a run is interrupted, you can resume from the last checkpoint rather than starting over. Set `checkpoint_dir` on the workflow and call `resume_from_checkpoint()` to restore.

**Serialization** lets you save and reload the full topology — nodes, edges, and agent configuration — as JSON. Use `save_spec()` to export and `GraphWorkflow.from_spec()` to restore.

```python
from swarms.structs.graph_workflow import GraphWorkflow
from swarms import Agent

analyst = Agent(agent_name="Analyst", model_name="gpt-5.4", max_loops=1)
writer = Agent(agent_name="Writer", model_name="gpt-5.4", max_loops=1)
reviewer = Agent(agent_name="Reviewer", model_name="gpt-5.4", max_loops=1)

workflow = GraphWorkflow(
    checkpoint_dir="./checkpoints/my_workflow",
    max_loops=5,
)
workflow.add_node(analyst)
workflow.add_node(writer)
workflow.add_node(reviewer)
workflow.add_edge("Analyst", "Writer")
workflow.add_edge("Writer", "Reviewer")
workflow.set_entry_point("Analyst")
workflow.set_end_point("Reviewer")

# Save topology to disk
workflow.save_spec("workflow_spec.json")

# Resume an interrupted run
workflow.resume_from_checkpoint()

result = workflow.run("Write a technical analysis of quantum computing adoption timelines")
print(result)
```

---

### `swarms init` CLI Command

A new `swarms init` command scaffolds a new Swarms project in the current directory — creates the directory structure, a starter `main.py`, and a `.env` template. Pair it with `swarms autoswarm` to go from zero to a running multi-agent system in under a minute.

```bash
# Initialize a new project
swarms init

# Generate a swarm config from a natural language description
swarms autoswarm --output-dir ./my_swarm --no-run

# Run it
python my_swarm/swarm_config.py
```

---

### `swarms autoswarm` Generates Ready-to-Run Python Files

`swarms autoswarm` previously only printed a config to stdout. It now writes a complete, runnable `.py` file with `Agent()` constructors, `SwarmRouter` setup, and a `__main__` block. Use `--output` to specify a file path, `--output-dir` to specify a directory, and `--no-run` to generate without immediately executing.

```bash
swarms autoswarm \
  --task "Build a 3-agent pipeline: researcher, writer, editor" \
  --output-dir ./generated \
  --no-run
```

---

### Concurrent `batch_run`

`Agent.batch_run()` now executes tasks concurrently using `concurrent.futures` with proper task isolation. Each task runs in its own isolated agent copy so conversation state doesn't bleed between tasks. This replaces the previous sequential loop and can produce significant speedups for independent batch workloads.

```python
from swarms import Agent

agent = Agent(
    agent_name="Summarizer",
    system_prompt="Summarize the following text in 2 sentences.",
    model_name="gpt-5.4",
    max_loops=1,
)

tasks = [
    "Explain quantum entanglement...",
    "Describe the history of the Roman Empire...",
    "What is gradient descent in machine learning...",
]

# All three run concurrently
results = agent.batch_run(tasks)
for task, result in zip(tasks, results):
    print(f"Task: {task[:40]}...")
    print(f"Result: {result}\n")
```

---

## Improvements

### HierarchicalSwarm: Task Distribution Visualization

`HierarchicalSwarm.run()` now prints a rich tree panel after the director parses orders, showing exactly which agent was assigned which subtask. No more guessing what the director decided — the distribution is visible at runtime.

### AgentRearrange: O(1) Agent Lookup

Agent lookup inside `HierarchicalSwarm` was a linear scan over the agent list on every lookup. It's now a dict lookup built once at initialization — O(1) regardless of swarm size.

### Sequential Awareness for Repeated Agents

`AgentRearrange` flows that use the same agent more than once (e.g., `"Writer -> Reviewer -> Writer"`) now give each occurrence the correct positional context. Previously, both Writer instances received identical neighbor info. Each occurrence now gets its actual predecessor and successor.

### `load_swarms_env()` for Reliable `.env` Loading

A new `swarms/env.py` module provides `load_swarms_env()` using `find_dotenv` to search upward through parent directories for a `.env` file. All entry points (`__init__.py`, `cli/main.py`, `auto_generate_swarm_config.py`) now use this instead of bare `load_dotenv()`, so `.env` files are found consistently regardless of where you invoke the CLI.

### Conversation Caching Disabled by Default

Conversation string caching (`cache_enabled`) now defaults to `False`. Caching is still available and opt-in — set `cache_enabled=True` on any `Agent`. This prevents unexpected cache hits when running agents in batch or test contexts.

### Dependency Cleanup

Removed unused packages from requirements, pinned litellm to match pyproject, and added the missing `requests` dependency to pyproject. The dependency surface is now smaller and the lockfile more stable.

---

## Security

v11 includes a full security audit pass — the most comprehensive one since the project launched. Swarms agents can fetch URLs, execute tools, parse untrusted config files, and cache credentials. Each of those surfaces was reviewed and hardened. Below is every change, what it closes, and why it matters.

### SSRF: Private and Loopback URLs Now Blocked

**What it was:** When agents were given URL-fetching tools, nothing prevented them from requesting internal network addresses — `http://192.168.x.x`, `http://10.x.x.x`, `http://localhost`, `http://169.254.169.254` (AWS metadata endpoint), and similar. A malicious task or a prompt injection in fetched content could have redirected an agent to exfiltrate internal infrastructure data.

**What changed:** The request path now validates every URL before making an outbound call. Requests targeting RFC 1918 private ranges, loopback addresses, and link-local ranges are rejected with an error before any network connection is made.

### Shell Injection: `os.system` Replaced with `subprocess.run`

**What it was:** Several tool execution and CLI paths used `os.system()` with string-interpolated arguments. `os.system` passes its argument directly to the shell, so any unsanitized input — a file path, a task string, a config value — could execute arbitrary shell commands.

**What changed:** All `os.system()` calls are replaced with `subprocess.run()` using list arguments. When arguments are passed as a list rather than a shell string, the OS exec syscall is called directly with no shell interpolation, eliminating the injection surface entirely.

### YAML Deserialization: `yaml.load` → `yaml.safe_load`

**What it was:** Config and spec files were loaded with `yaml.load()` and no explicit `Loader`. PyYAML's default loader supports Python-specific tags like `!!python/object` that allow arbitrary object instantiation from a YAML file. A crafted config file — or a YAML file fetched from an external source — could execute arbitrary Python at parse time.

**What changed:** Every `yaml.load()` call is replaced with `yaml.safe_load()`. The safe loader only supports standard YAML types (strings, numbers, lists, dicts) and raises an error on any Python-specific tags.

### Auth Cache File Permissions

**What it was:** The authentication cache file was written to the system temp directory with default permissions — typically world-readable on Linux and macOS. Any process running as a different user on the same machine could read cached credentials.

**What changed:** The cache file is now written to `~/.swarms/` in the user's home directory with `0600` permissions (owner read/write only). No other user or process can read it.

### Bash Blocklist Expanded with Regex Patterns

**What it was:** The bash tool blocklist used simple string matching, which could be bypassed with minor variations — extra spaces, different quoting, aliased commands.

**What changed:** The blocklist now uses compiled regex patterns, covering command variations, whitespace differences, and common bypass forms. Dangerous commands are caught regardless of how they are invoked.

### Bare `except` Clauses Now Log Instead of Silent Pass

**What it was:** A number of `except: pass` and `except Exception: pass` blocks silently discarded errors. This masked real failures and made debugging production issues extremely difficult — a component would fail and the system would continue as if nothing happened.

**What changed:** All bare silent catches now log the exception with `logger.error()` before continuing. Errors surface in logs instead of disappearing.

### Temp Directory Cleanup on Zip Failure

**What it was:** If a zip extraction failed mid-way, the temporary directory created for the extraction was left on disk. On long-running processes or repeated failures, this accumulated disk usage and potentially left partial — and possibly sensitive — file contents in a world-readable temp location.

**What changed:** Temp directory cleanup is now wrapped in a `try/finally` block, ensuring the directory is removed whether extraction succeeds or fails.

### litellm Dependency Pinned Against Known Vulnerability

A known security issue in an unpinned litellm version range was identified and the dependency has been pinned to a safe version. The litellm version in `pyproject.toml` now matches the pin in `requirements.txt`, eliminating the discrepancy that allowed a vulnerable version to be installed in some environments.

---

## Bug Fixes

### AgentRearrange No Longer Silently Swallows Exceptions

Three nested `try/except` layers in `AgentRearrange` (`_run`, `run`, `__call__`) were catching all exceptions and returning `None`. Callers had no way to distinguish a failed workflow from a successful one with an empty result. All silent catches have been removed — exceptions now propagate correctly.

Closes #1464, #1465

### Anthropic Temperature Compatibility

Removed `dynamic_temperature_enabled` from the agent parameter set. This field was being forwarded to Anthropic's API, which doesn't accept it, causing hard failures when using Anthropic models. The field is gone; temperature is now handled correctly for all providers.

### GraphWorkflow `max_loops` Was Always 1

A misplaced `return` statement inside the `while` loop in `GraphWorkflow.run()` caused the workflow to exit after the first iteration regardless of the `max_loops` value. The `return` is now outside the loop; per-loop results accumulate and end-point outputs are fed as context into subsequent iterations.

### Broken Documentation Links

Fixed 6 broken links in the docs and README: HierarchicalSwarm and ConcurrentWorkflow README links, 3 broken footer links in `mkdocs.yml`, and phantom entries in `structs/overview.md` for structs that don't exist (TaskQueueSwarm, MatrixSwarm, Deep Research Swarm).

---

## Refactors

### HeavySwarm Prompts Extracted to Dedicated Module

All prompt constants and tool schemas (~490 lines) have been moved from `heavy_swarm.py` into `swarms/prompts/heavy_swarm_prompts.py`. Re-exports in `heavy_swarm.py` keep backwards compatibility. The struct file is now focused on orchestration logic.

### Dead uvloop Execution Functions Removed

`run_agents_concurrently_uvloop` and `run_agents_with_tasks_uvloop` had zero production callers — they were only referenced in `__init__.py` re-exports, example files, and docs. Both functions (~280 lines), their exports, and 8 example files have been deleted. The actively-used `run_agents_concurrently` and `run_agents_with_different_tasks` are unaffected.

### AgentRearrange Error Handling Tests Consolidated

Tests from `test_fix_1464_1465.py` have been merged into `test_agent_rearrange.py`. All AgentRearrange tests now live in one file.

---

## Deprecations and Removals

The following structs and utilities have been deleted. They had no active users in the codebase, their dependencies were either removed or unmaintained, and keeping them imposed ongoing maintenance cost.

- `SwarmTemplates`: unused struct with no callers
- `BoardOfDirectors`: superseded by `HierarchicalSwarm`
- `EuroSwarm` (parliament simulation): example-only, not a production struct
- `MAKER`: unused struct
- `Agent GRPO`: experimental, no adoption
- `OpenAIAssistant` / wrapper: OpenAI Assistants API deprecated upstream
- `check-tokens` utility: dead code
- `data-to-text` utility: dead code; callers migrated to stdlib `csv`
- `run_agents_concurrently_uvloop`: no production callers, superseded by `run_agents_concurrently`
- `run_agents_with_tasks_uvloop`: no production callers, superseded by `run_agents_with_different_tasks`
- Trivy CI/CD vulnerability scanning workflow: removed from CI pipeline

If you were importing any of these directly, update your imports. The stdlib `csv` module is a drop-in replacement for `data-to-text` in all known usages.

---

## What v11 Means in Practice

Step back and the shape of this release is clear: Swarms is consolidating around a smaller set of well-maintained, well-tested primitives — and expanding the range of what those primitives can do.

On the architecture side, v11 adds five new coordination patterns that didn't exist in v10. AdvisorSwarm gives you a principled way to get high-quality reasoning without paying for Opus on every turn. PlannerGeneratorEvaluator gives you a feedback loop for tasks where quality is measurable. PlannerWorkerSwarm gives you genuine parallelism across specialists. DriftDetectionAgent gives long-running pipelines a way to self-correct before outputs drift too far. And the two new HeavySwarm Grok modes give you 4-agent and 16-agent depth for problems that require broad multi-domain analysis. Every one of these is provider-agnostic via LiteLLM.

On the infrastructure side, GraphWorkflow is now a first-class choice for production workflows — not just experiments. Checkpoint resumption means a multi-hour graph run can survive a crash. Topology serialization means you can version, share, and restore workflow configurations. Concurrent `batch_run` means single-agent batch workloads are no longer artificially serialized.

On the reliability side, the security audit closed real vulnerabilities — SSRF, shell injection, unsafe deserialization, weak file permissions — not theoretical ones. The silent exception swallowing in `AgentRearrange` was a silent correctness bug that affected anyone running workflows with error handling. The Anthropic temperature compatibility fix unblocked a class of users who hit hard failures with Anthropic models.

And on the maintenance side, eleven dead structs are gone. The import surface is smaller. The test suite is cleaner. The dependency set is tighter.

Upgrade with `pip install -U swarms`. If you're importing anything from the removals list above, those imports will break — check the deprecations section for replacements. Everything else is backwards compatible.
