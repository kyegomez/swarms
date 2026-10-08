# Swarms v11 Changelog

**Period covered:** March 13 – April 19, 2026  
**Total commits:** 120  
**Pull requests merged:** 35 (PRs #1443 – #1546)  
**Issues closed:** #1049, #1469, #1481, #1464, #1465, #1502, #1526  
**Version bump:** 10.0.0 → 10.0.2 (released March 14, 2026)

---

## Introduction

Over four weeks, the Swarms team and community contributors shipped 120 commits across 35 pull requests — 3 new swarm architectures, 7 bug fixes, 5 security patches, and the removal of over 13,000 lines of deprecated code. New additions include AdvisorSwarm (executor + on-demand advisor pair), PlannerWorkerSwarm (parallel task queue with dependency ordering and a judge cycle), and PlannerGeneratorEvaluator (GAN-style generate-score-retry harness). HeavySwarm gained Grok 4.20 Heavy support and a 16-agent mode.

GraphWorkflow gained topology serialization and mid-run checkpointing. AgentRearrange batch execution is now concurrent with proper task isolation. Security fixes closed vulnerabilities in YAML deserialization, shell injection, SSRF, and auth cache permissions. Silent exception swallowing, a max_loops iteration bug, and an Anthropic API compatibility break were all resolved.

---

## Table of Contents

1. [New Swarm Structs](#1-new-swarm-structs)
   - [AdvisorSwarm](#11-advisorswarm)
   - [PlannerWorkerSwarm](#12-plannerworkerswarm)
   - [PlannerGeneratorEvaluator](#13-plannergeneratorevaluator)
2. [New Agent Capabilities](#2-new-agent-capabilities)
   - [DriftDetectionAgent (SequentialWorkflow)](#21-driftdetectionagent-sequentialworkflow)
3. [HeavySwarm Enhancements](#3-heavyswarm-enhancements)
   - [Grok 4.20 Heavy Architecture](#31-grok-420-heavy-architecture)
   - [16-Agent Heavy Mode](#32-16-agent-heavy-mode)
   - [Prompts Extracted to Module](#33-prompts-extracted-to-module)
4. [GraphWorkflow Improvements](#4-graphworkflow-improvements)
   - [Topology Serialization & Deserialization](#41-topology-serialization--deserialization)
   - [Checkpoint Support](#42-checkpoint-support)
   - [max_loops Bug Fix](#43-max_loops-bug-fix)
5. [AgentRearrange Improvements](#5-agentrearrange-improvements)
   - [Concurrent batch_run](#51-concurrent-batch_run)
   - [O(1) Agent Lookup](#52-o1-agent-lookup)
   - [Exception Propagation Fix](#53-exception-propagation-fix)
   - [Repeated-Agent Flow Awareness Fix](#54-repeated-agent-flow-awareness-fix)
6. [CLI Enhancements](#6-cli-enhancements)
   - [swarms init Command](#61-swarms-init-command)
   - [autoswarm File Writer](#62-autoswarm-file-writer)
7. [Security Hardening](#7-security-hardening)
8. [Conversation Caching Refactor](#8-conversation-caching-refactor)
9. [Dependency & Compatibility Fixes](#9-dependency--compatibility-fixes)
   - [Anthropic Temperature Compatibility](#91-anthropic-temperature-compatibility)
   - [litellm Version Pin & Security Fix](#92-litellm-version-pin--security-fix)
10. [Environment Loading Improvements](#10-environment-loading-improvements)
11. [Codebase Audit & Cleanup](#11-codebase-audit--cleanup)
    - [Removed Deprecated Structs](#111-removed-deprecated-structs)
    - [Dead Code Removal (uvloop/winloop)](#112-dead-code-removal-uvloopwinloop)
12. [HierarchicalSwarm Fixes](#12-hierarchicalswarm-fixes)
13. [Documentation Fixes](#13-documentation-fixes)
14. [Dependency Updates](#14-dependency-updates)
15. [Full PR List](#15-full-pr-list)

---

## 1. New Swarm Structs

### 1.1 AdvisorSwarm

**PR #1532** | Merged April 14, 2026 | Closes #1526  
**Author:** Steve-Dusty

Implements Anthropic's advisor strategy pattern: a cheaper **executor** model drives the task end-to-end while a more powerful **advisor** model is consulted on-demand between executor turns. The advisor reads the same shared conversation context but never calls tools or produces user-facing output — it only provides strategic guidance.

**Files added:**
- `swarms/structs/advisor_swarm.py` (366 lines)
- `swarms/prompts/advisor_swarm_prompts.py`
- `swarms/structs/swarm_router.py` — registered as a SwarmRouter option
- `tests/structs/test_advisor_swarm.py` (22 unit tests + 4 integration tests)
- `docs/swarms/structs/advisor_swarm.md`
- `examples/multi_agent/advisor_swarm_examples/advisor_swarm_example.py`

**Architecture:**
```
Main loop ──> [ Executor (Sonnet) ] ──on-demand──> [ Advisor (Opus) ]
                    |       ^                            |       ^
               read/write   |                      sends advice  |
                    v       |                            v       |
              [ Shared conversation context · tools · history ]
```

**Example:**
```python
from swarms import AdvisorSwarm

# Pair a cheap executor with a powerful advisor
swarm = AdvisorSwarm(
    executor_model_name="claude-sonnet-4-6",
    advisor_model_name="claude-opus-4-6",
    max_advisor_uses=3,   # budget: how many times advisor is consulted
    max_loops=5,          # executor turns
)

result = swarm.run("Design and implement a rate-limiting middleware in Python")
print(result)
```

**With custom pre-configured agents (e.g., tools or MCP):**
```python
from swarms import Agent, AdvisorSwarm

executor = Agent(
    agent_name="CodeExecutor",
    model_name="claude-sonnet-4-6",
    tools=[my_code_tool, my_search_tool],
)

advisor = Agent(
    agent_name="ArchitectAdvisor",
    model_name="claude-opus-4-6",
)

swarm = AdvisorSwarm(
    executor_agent=executor,
    advisor_agent=advisor,
    max_advisor_uses=5,
)
result = swarm.run("Refactor the auth module to use OAuth2")
```

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `executor_model_name` | `"claude-sonnet-4-6"` | Model for the executor |
| `advisor_model_name` | `"claude-opus-4-6"` | Model for the advisor |
| `max_advisor_uses` | `3` | Max advisor consultations per `run()` |
| `max_loops` | `1` | Executor turns |
| `executor_agent` | `None` | Optional pre-configured Agent |
| `advisor_agent` | `None` | Optional pre-configured Agent |
| `tools` | `None` | Tools available to executor only |

---

### 1.2 PlannerWorkerSwarm

**PR #1443** | Merged March 17, 2026  
**Author:** Steve-Dusty

A multi-agent harness with three roles: **Planner**, **Workers** (parallel), and an optional **Judge**. The Planner decomposes a task into a structured plan with dependencies; workers execute tasks concurrently respecting dependency order; the Judge cycles back if quality criteria aren't met.

**Files added:**
- `swarms/structs/planner_worker_swarm.py` (937 lines)
- `swarms/schemas/planner_worker_schemas.py` (178 lines — Pydantic schemas)
- `swarms/prompts/planner_worker_prompts.py`
- `swarms/structs/swarm_router.py` — registered
- `tests/structs/test_planner_worker_swarm.py` (495 lines)
- `examples/planner_worker_swarm_example.py`
- `docs/swarms/structs/planner_worker_swarm.md`

**Architecture:**

The `TaskQueue` uses a thread-safe dict with optimistic concurrency. Workers `claim()` tasks atomically; dependency resolution ensures a task only becomes available when all `depends_on` tasks are `COMPLETED`.

```python
from swarms import PlannerWorkerSwarm

swarm = PlannerWorkerSwarm(
    name="ResearchSwarm",
    description="Multi-agent research and synthesis pipeline",
    director_model_name="gpt-5.4",
    worker_model_name="claude-sonnet-4-6",
    num_workers=4,
    max_loops=3,
    agent_as_judge=True,    # enable quality-gate judge cycle
)

result = swarm.run(
    "Produce a comprehensive market analysis report for EV battery technology"
)
```

**Task priority and dependency example:**
```python
from swarms.schemas.planner_worker_schemas import PlannerTask, TaskPriority, PlannerTaskStatus

# Tasks are created by the Planner automatically, but you can inspect them:
task = PlannerTask(
    title="Gather raw data",
    description="Collect EV market data from public sources",
    priority=TaskPriority.HIGH,
    depends_on=[],           # no dependencies — runs first
    status=PlannerTaskStatus.PENDING,
)
```

**Key components:**
- `TaskQueue` — thread-safe, priority-ordered, dependency-aware
- `PlannerTask` / `PlannerTaskSpec` — Pydantic schemas for structured task plans
- `CycleVerdict` — judge output schema (`APPROVE` / `REVISE` with feedback)

---

### 1.3 PlannerGeneratorEvaluator

**PR #1501** | Merged March 29, 2026  
**Commit:** `78d91a06` | March 25, 2026

A domain-agnostic three-agent orchestration harness inspired by GAN-style architecture. A **Planner** expands a short prompt into a structured plan; a **Generator** proposes and executes each step; an **Evaluator** scores outputs against criteria and triggers retries when below threshold. All three agents communicate through a single shared file on disk.

**Files added:**
- `swarms/structs/planner_generator_evaluator.py` (988 lines)
- `swarms/prompts/planner_generator_evaluator_prompts.py` (164 lines)
- `tests/structs/test_planner_generator_evaluator.py` (289 lines)
- `docs/swarms/structs/planner_generator_evaluator.md`
- `examples/multi_agent/planner_generator_evaluator/pge_example.py`
- `examples/multi_agent/planner_generator_evaluator/pge_with_tools_example.py`

**Flow:**
```
User prompt
    │
    ▼
[ Planner ]  ──── structured plan with eval criteria ────►
    │
    ▼  (for each step)
[ Generator ] ◄──── proposes step contract ────► [ Evaluator ]
    │                                                   │
    │◄──────── score + feedback if below threshold ─────┘
    │
    ▼  (if all criteria pass)
[ Next step ... ]
    │
    ▼
Final output
```

**Example:**
```python
from swarms import PlannerGeneratorEvaluator

harness = PlannerGeneratorEvaluator(
    planner_model_name="gpt-5.4",
    generator_model_name="claude-sonnet-4-6",
    evaluator_model_name="claude-sonnet-4-6",
    score_threshold=0.8,     # minimum score to proceed
    max_retries=3,           # max Generator retries per step
    shared_file_path="./workspace/pge_shared.md",
)

result = harness.run("Build a Python CLI tool that converts CSV to Parquet")
```

**With tools (for code execution):**
```python
from swarms import PlannerGeneratorEvaluator

def run_python(code: str) -> str:
    """Execute Python code and return stdout."""
    ...

harness = PlannerGeneratorEvaluator(
    generator_tools=[run_python],
    score_threshold=0.75,
)
result = harness.run("Implement and test a red-black tree in Python")
```

**Key components:**
- `StepContract` — negotiated contract between Generator and Evaluator for each step
- Score trajectory tracking — `refine` vs `pivot` decisions based on score history
- Hard threshold enforcement — steps cannot advance until all criteria pass

---

## 2. New Agent Capabilities

### 2.1 DriftDetectionAgent (SequentialWorkflow)

**PR #1522** | Merged April 14, 2026  
**Commits:** `2c6a61ce`, `7ce4f781`  
**Author:** adichaudhary

Adds an optional **drift detection** judge to `SequentialWorkflow`. After the pipeline completes, a separate judge agent scores the final output's semantic alignment with the original task (0–1 scale). A warning is logged when the score falls below the configured `drift_threshold`.

**Example:**
```python
from swarms import Agent, SequentialWorkflow

agents = [
    Agent(agent_name="Researcher", model_name="gpt-5.4"),
    Agent(agent_name="Writer", model_name="claude-sonnet-4-6"),
    Agent(agent_name="Editor", model_name="claude-sonnet-4-6"),
]

workflow = SequentialWorkflow(
    agents=agents,
    drift_detection=True,       # enable post-pipeline alignment check
    drift_threshold=0.75,       # warn if alignment score < 0.75
    drift_model="claude-sonnet-4-5",
)

result = workflow.run("Write a technical blog post about transformer attention mechanisms")
# If the Editor drifts too far from the original task, a warning is emitted
```

**Parameters added to `SequentialWorkflow`:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `drift_detection` | `False` | Enable post-pipeline drift scoring |
| `drift_threshold` | `0.75` | Minimum acceptable alignment score |
| `drift_model` | `"claude-sonnet-4-5"` | Model for the judge agent |

The drift loop was extracted into a dedicated `_drift_detection_loop()` method for testability (commit `7ce4f781`).

---

## 3. HeavySwarm Enhancements

### 3.1 Grok 4.20 Heavy Architecture

**PR #1499** | Merged March 30, 2026  
**Commit:** `67acb98f`  
**Author:** Steve-Dusty

Added full Grok 4.20 Heavy architecture support to `HeavySwarm`. This is a distinct swarm configuration optimized for xAI's Grok models, including dedicated examples and comprehensive tests.

**Files added:**
- `swarms/structs/heavy_swarm.py` — major additions (+788 lines)
- `tests/structs/test_heavy_swarm_grok.py` (424 lines)
- `examples/multi_agent/heavy_swarm_examples/heavy_swarm_grok_basic.py`
- `examples/multi_agent/heavy_swarm_examples/heavy_swarm_grok_medical.py`
- `examples/multi_agent/heavy_swarm_examples/heavy_swarm_grok_no_dashboard.py`
- `examples/multi_agent/heavy_swarm_examples/heavy_swarm_grok_question_preview.py`
- `examples/multi_agent/heavy_swarm_examples/heavy_swarm_grok_router.py`
- `docs/swarms/structs/heavy_swarm.md` — extended documentation (+215 lines)

**Example:**
```python
from swarms import HeavySwarm

swarm = HeavySwarm(
    model_name="xai/grok-4-0709",
    num_agents=4,
    question="What are the key mechanisms by which GLP-1 agonists reduce cardiovascular risk?",
    verbose=True,
)
result = swarm.run()
```

**Router pattern (multiple models):**
```python
from swarms import HeavySwarm

# Route to Grok-based HeavySwarm via SwarmRouter
from swarms import SwarmRouter

router = SwarmRouter(
    swarm_type="HeavySwarm",
    model_name="xai/grok-4-0709",
)
result = router.run("Analyze the thermodynamic feasibility of fusion at room temperature")
```

### 3.2 16-Agent Heavy Mode

**PR #1509** | Merged April 2, 2026  
**Commit:** `c9c6ea03`  
**Author:** Steve-Dusty

Extended HeavySwarm with a 16-agent Grok 4.20 Heavy mode, providing maximum parallelism for deeply complex research tasks.

```python
from swarms import HeavySwarm

# 16-agent mode for maximum research depth
swarm = HeavySwarm(
    model_name="xai/grok-4-0709",
    num_agents=16,
    question="Synthesize the current state of quantum error correction and near-term scalability prospects",
)
result = swarm.run()
```

**Files added:**
- `examples/multi_agent/heavy_swarm_examples/heavy_swarm_16_agents.py`
- `examples/multi_agent/heavy_swarm_examples/heavy_swarm_4_agents.py`
- `tests/structs/test_heavy_swarm_grok_heavy.py` (179 lines)
- Documentation updates for 16-agent mode

### 3.3 Prompts Extracted to Module

**Commit:** `812293e3` | March 29, 2026

All HeavySwarm prompts and JSON schemas were moved from inline strings inside `heavy_swarm.py` to a dedicated module `swarms/prompts/heavy_swarm_prompts.py` (779 lines). This improves maintainability and allows prompt reuse across configurations.

---

## 4. GraphWorkflow Improvements

### 4.1 Topology Serialization & Deserialization

**PR #1497** | Merged April 5, 2026  
**Commits:** `849c47d8`, `f5fec709`  
**Author:** adichaudhary

`GraphWorkflow` can now save its full topology to JSON and reconstruct it later. Hashing uses SHA-256 (replacing Python's non-deterministic `hash()`), and conversation history is replayed on resume.

**New methods on `GraphWorkflow`:**

```python
from swarms import GraphWorkflow, Agent

# Build workflow
wf = GraphWorkflow()
wf.add_node(Agent(agent_name="Analyst", model_name="gpt-5.4"))
wf.add_node(Agent(agent_name="Writer", model_name="claude-sonnet-4-6"))
wf.add_edge("Analyst", "Writer")

# Save topology to JSON
wf.save_spec("my_workflow.json")

# Later: reconstruct from JSON (no agents need to be rebuilt manually)
wf2 = GraphWorkflow.from_topology_spec("my_workflow.json", agents=[...])
wf2.run("Analyze Q3 earnings and write a summary")
```

**Changes:**
- `to_spec()` — serializes nodes, edges, and metadata to a dict
- `save_spec(path)` — writes the spec JSON to disk
- `from_topology_spec(path, agents)` — reconstructs topology; conversation history replayed
- SHA-256 replaces `hash()` for deterministic, stable node IDs across processes
- `Applied Copilot changes` commit (`2ebced2c`): deterministic ordering, consistent `ValueError`, test coverage

### 4.2 Checkpoint Support

**PR #1498** | Merged April 8, 2026  
**Commit:** `a7fcf189`  
**Author:** adichaudhary

Long-running `GraphWorkflow` executions can now be checkpointed and resumed, protecting against partial failures.

```python
from swarms import GraphWorkflow

wf = GraphWorkflow(
    checkpoint_dir="./checkpoints",   # new parameter
    checkpoint_interval=5,            # save every 5 steps
)
wf.add_node(...)
wf.add_edge(...)

# If interrupted, resume from the last checkpoint:
wf2 = GraphWorkflow.load_checkpoint("./checkpoints/my_workflow_step_10.json")
wf2.run()   # continues from step 11
```

### 4.3 max_loops Bug Fix

**PR #1490** | Merged April 2, 2026  
**Commit:** `7f90de56` | Closes #1481  
**Author:** Steve-Dusty

Fixed a bug where `GraphWorkflow` with `max_loops > 1` would only execute the first iteration, silently skipping remaining loops.

**Before (broken):**
```python
wf = GraphWorkflow(max_loops=3)
wf.run("task")  # only executed 1 loop, not 3
```

**After (fixed):**
```python
wf = GraphWorkflow(max_loops=3)
wf.run("task")  # correctly executes all 3 loops
```

---

## 5. AgentRearrange Improvements

### 5.1 Concurrent batch_run

**PR #1512** | Merged April 7, 2026  
**Commit:** `197a8f6f`  
**Author:** adichaudhary

`AgentRearrange.batch_run()` now executes tasks concurrently using `ThreadPoolExecutor` with proper task isolation. Each task gets its own deep-copied agent instances to prevent state bleed between concurrent runs.

**Before:** Tasks were processed sequentially.  
**After:** Tasks run in parallel; each task has isolated agent state.

```python
from swarms import Agent, AgentRearrange

agents = [
    Agent(agent_name="Researcher", model_name="gpt-5.4"),
    Agent(agent_name="Writer", model_name="claude-sonnet-4-6"),
]

pipeline = AgentRearrange(
    agents=agents,
    flow="Researcher -> Writer",
)

# Now runs all 100 tasks concurrently with isolated agent state
results = pipeline.batch_run(
    tasks=["task_1", "task_2", ..., "task_100"],
    batch_size=10,
)
```

**Note:** Non-deepcopyable agent fallback was explicitly removed as out-of-scope — agents must support deep copy for concurrent execution.

### 5.2 O(1) Agent Lookup

**Commit:** `04d24755` | March 18, 2026  
**Author:** Steve-Dusty | Closes #1469 (machine-ID stability fix)

Agent lookups inside `AgentRearrange` were changed from linear scan O(n) to O(1) dict lookup. This is a significant performance improvement for large swarms.

```python
# Before: O(n) scan on every lookup
agent = next(a for a in self.agents if a.agent_name == name)

# After: O(1) dict lookup
agent = self._agent_map[name]
```

The internal `_agent_map: Dict[str, Agent]` is built once at construction and kept in sync on mutations.

### 5.3 Exception Propagation Fix

**PR #1476** | Merged March 19, 2026  
**Commit:** `f623f112`

`AgentRearrange` was silently swallowing exceptions from agents, making debugging extremely difficult. Exceptions now propagate correctly.

**Before:**
```python
# Exceptions were caught and silently dropped — no error surface
result = pipeline.run("task")  # succeeds even if agents threw
```

**After:**
```python
# Exceptions propagate to the caller
try:
    result = pipeline.run("task")
except Exception as e:
    print(f"Agent failed: {e}")  # now visible
```

Tests were updated to use real LLMs instead of mocks (commit `7e087d54`) to catch this class of bug going forward.

### 5.4 Repeated-Agent Flow Awareness Fix

**PR #1479** | Merged April 4, 2026  
**Commit:** `af38ee0b`

Fixed incorrect sequential awareness when the same agent appears multiple times in a flow string (e.g., `"A -> B -> A -> C"`). The previous implementation lost track of position for repeated agents, causing wrong context to be passed.

```python
# Flow with repeated agent — now works correctly
pipeline = AgentRearrange(
    agents=[agent_a, agent_b, agent_c],
    flow="AgentA -> AgentB -> AgentA -> AgentC",
)
result = pipeline.run("task")
# AgentA's second appearance correctly receives AgentB's output, not the original task
```

---

## 6. CLI Enhancements

### 6.1 swarms init Command

**Commits:** `a8332559`, `070284d9` | April 5, 2026

Added a new `swarms init` CLI command that scaffolds a project directory with a starter agent configuration file.

```bash
# Create a new swarms project
swarms init my_research_agent

# Generates:
# my_research_agent/
# ├── agent.yaml
# └── main.py
```

The `init` command was integrated alongside the existing `autoswarm` argument without conflict (merge commit `61904d19`).

### 6.2 autoswarm File Writer

**PR #1489** | Merged April 8, 2026  
**Commit:** `96c1a470` | Closes PR #1487  
**Author:** Steve-Dusty

`swarms autoswarm` now always writes a ready-to-run Python file to disk in addition to executing the swarm. Use `--no-run` to generate the file without executing.

```bash
# Generate and run (default)
swarms autoswarm "Build a multi-agent customer support pipeline"

# Generate file only, do not execute
swarms autoswarm "Build a multi-agent customer support pipeline" --no-run

# Specify output file path
swarms autoswarm "Build a pipeline" -o ./my_swarm.py --no-run
```

**`--output-dir` flag** was also added (commit `0358a6df`):
```bash
# Write generated file to a specific directory
swarms autoswarm "Build a pipeline" --output-dir ./generated_swarms/
```

**Files changed:**
- `swarms/agents/auto_generate_swarm_config.py` — `write_autoswarm_file()` function (+221 lines)
- `swarms/cli/main.py` — `run_autoswarm()` refactored (+91 lines)
- `tests/agents/test_autoswarm_writer.py` (731 lines)

---

## 7. Security Hardening

**Commit:** `759d8a6a` | April 5, 2026

A comprehensive security audit and hardening pass across the codebase. Multiple CVE-class vulnerabilities were fixed.

### YAML Deserialization (RCE)
```python
# Before — arbitrary code execution via yaml.load
data = yaml.load(user_input)

# After — safe deserialization
data = yaml.safe_load(user_input)
```

### Shell Injection
```python
# Before — command injection via os.system + string interpolation
os.system(f"unzip {user_path}")

# After — no shell expansion; args as list
subprocess.run(["unzip", user_path], check=True)
```

### Bash Blocklist Expansion
Expanded the autonomous agent bash blocklist with regex-based pattern matching:
```python
# New regex patterns block:
# - base64 decode piped to shell  (base64 -d ... | bash)
# - writing to /etc, /bin, /usr, /root
# - command substitution feeding sensitive commands
# - sudo / su / pkexec (privilege escalation)
# - /etc/passwd, /etc/shadow, /etc/sudoers reads
# - environment variable dumps (printenv, env |, set |)
# - network exfiltration (nc -e, ncat -e, wget --post, curl -d)
# - history manipulation (history -c, unset histfile)
```

### SSRF (Server-Side Request Forgery)
```python
# Private and loopback URLs are now blocked in the URL-fetch utility:
# 127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, ::1, etc.
```

### Auth Cache Permissions
```python
# Cache file now stored in user home with restricted permissions
cache_dir = Path.home() / ".cache" / "swarms"
cache_file = cache_dir / "auth"
cache_file.chmod(0o600)   # owner read/write only
```

### Temp Directory Cleanup
```python
# Temp dir is now cleaned up on zip failure to avoid leftover sensitive data
```

### Error Handling
```python
# Bare `except: pass` replaced with logging:
except Exception as e:
    logger.error(f"Operation failed: {e}")
```

**Additional security:** Trivy CI/CD vulnerability scanning workflow was removed (commit `0b683f54`) as it was no longer maintained.

---

## 8. Conversation Caching Refactor

**PR #1480** | Merged March 23, 2026  
**Commits:** `76d87bdd`, `a97681c6`, `0b683f54`  
**Author:** adichaudhary

`Conversation` caching was significantly refactored for correctness and simplicity.

**Key changes:**
- Caching is **disabled by default** (`cache_enabled=False`) — opt-in only
- Caching logic moved entirely into `return_history_as_string()`
- `get_str()` is now a plain alias for `return_history_as_string()` — no separate logic
- `get_cache_stats()` simplified — redundant `total_tokens` field removed
- New examples directory: `examples/utils/conversation_cache_examples/`

```python
from swarms.structs.conversation import Conversation

# Opt-in to caching explicitly
conv = Conversation(cache_enabled=True)
conv.add("user", "What is the capital of France?")

# Caching happens inside return_history_as_string()
history = conv.return_history_as_string()

stats = conv.get_cache_stats()
# Returns: {"hits": int, "misses": int}  (total_tokens removed)
```

---

## 9. Dependency & Compatibility Fixes

### 9.1 Anthropic Temperature Compatibility

**PR #1523** | Merged April 11, 2026  
**Commit:** `c46f788e`  
**Author:** adichaudhary

Removed the `dynamic_temperature_enabled` parameter from agent initialization when using Anthropic models. Anthropic's API rejects this field, causing `BadRequestError` for all Anthropic-backed agents.

**Before:**
```python
# Broke with Anthropic models — API rejected the field
agent = Agent(
    model_name="claude-sonnet-4-6",
    dynamic_temperature_enabled=True,  # ❌ Anthropic API error
)
```

**After:**
```python
# Works correctly — field is stripped before Anthropic API calls
agent = Agent(
    model_name="claude-sonnet-4-6",
    # dynamic_temperature_enabled is ignored for Anthropic models
)
```

### 9.2 litellm Version Pin & Security Fix

**Commits:** `484eb5bc`, `dc60825f` | April 11 & March 24, 2026

- `litellm` pinned to a specific version in `pyproject.toml` to match `requirements.txt` and prevent transitive security issues
- Security vulnerability in litellm prevented by locking to a known-safe version
- `requests` added as an explicit dependency in `pyproject.toml` (was previously implicit)
- Unused packages removed from `requirements.txt`

---

## 10. Environment Loading Improvements

**Commit:** `9734cee2` | March 14, 2026 (v10.0.0 release)

Added `swarms/env.py` with a `load_swarms_env()` helper that uses `find_dotenv()` to search upward through parent directories for a `.env` file. This replaces bare `load_dotenv()` calls that only searched the current directory.

```python
# Before — only found .env in the current working directory
from dotenv import load_dotenv
load_dotenv()

# After — searches upward through parent directories
from swarms.env import load_swarms_env
load_swarms_env()
```

All entry points now use `load_swarms_env()`:
- `swarms/__init__.py`
- `swarms/cli/main.py`
- `swarms/agents/auto_generate_swarm_config.py`

---

## 11. Codebase Audit & Cleanup

### 11.1 Removed Deprecated Structs

**Commits:** `ae942df0`, `7d2bd1e7`, `59a00f38` | April 14, 2026

A major codebase audit removed unmaintained and deprecated structs. Total lines removed: **~13,000+**.

| Struct/Module | Reason for removal |
|---|---|
| `BoardOfDirectors` | Deprecated, replaced by HierarchicalSwarm |
| `SwarmTemplates` | Unmaintained, no active users |
| `MAKER` struct | Replaced by more composable patterns |
| `AgentGRPO` | Experimental, not production-ready |
| `OpenAIAssistant` / `OpenAIAssistantWrapper` | OpenAI Assistants API deprecated |
| `EuroSwarm` parliament simulation | Example-only, too large to maintain |
| `check_all_model_max_tokens` utility | Dead code |
| `data_to_text` utility | Dead code; replaced with stdlib `csv` where needed |

**Files removed totals:**
- `swarms/structs/board_of_directors_swarm.py` (1,829 lines)
- `swarms/structs/maker.py` (1,093 lines)
- `swarms/agents/openai_assistant.py` (336 lines)
- `swarms/structs/agent_grpo.py` (221 lines)
- `swarms/utils/data_to_text.py` (135 lines)
- `swarms/utils/check_all_model_max_tokens.py` (70 lines)
- `examples/multi_agent/euroswarm_parliament/euroswarm_parliament.py` (3,614 lines)
- Various related tests, docs, and examples

**Also added:**
- `initialize_logger` exported from `swarms.utils`
- `get_cpu_cores` utility added to `swarms/utils/`
- ReAct agent tutorial added to examples (`docs/swarms/examples/react_agent_tutorial.md`)
- `CODEBASE_AUDIT.md` tracking current deletion status

### 11.2 Dead Code Removal (uvloop/winloop)

**PR #1544, #1545** | April 19, 2026 (reverted in PR #1546, then re-landed)  
**Authors:** Steve-Dusty, MycCellium420

Removed `run_agents_concurrently_uvloop()` and related dead uvloop/winloop execution functions from `swarms/structs/multi_agent_exec.py`. These functions were no longer called anywhere in the codebase after the async execution refactor.

**Removed:**
- `run_agents_concurrently_uvloop()` — uvloop/winloop concurrent runner
- Platform-detection code (`sys` import, conditional uvloop/winloop setup)
- All associated examples and documentation sections

**Also removed:**
- `examples/guides/850_workshop/uvloop_example.py`
- `examples/multi_agent/exec_utilities/uvloop_example.py`
- `examples/multi_agent/exec_utilities/new_uvloop_example.py`
- `examples/concurrent_examples/uvloop/` directory (4 files)

> **Note:** PR #1545 was briefly reverted (PR #1546) due to a conflict, then the refactored version was re-landed via PR #1544.

---

## 12. HierarchicalSwarm Fixes

**PR #1471** | Merged March 19, 2026  
**Author:** Nishant-k-sagar

Fixed agent lookup performance in `HierarchicalSwarm` — O(n) scan replaced with O(1) dict lookup (same pattern as AgentRearrange fix in §5.2).

**Also in this area (commit `9734cee2`):**
- `print_director_task_distribution()` added to `Formatter` — renders a rich Tree panel showing director-to-agent task assignments
- Wired into `HierarchicalSwarm.run()` to display task distribution after order parsing
- HierarchicalSwarm example updated: `gpt-5.4`, `parallel_execution=True`, `agent_as_judge=True`, `director_temperature=1.0`, `planning_enabled=False`

```python
from swarms import HierarchicalSwarm

swarm = HierarchicalSwarm(
    name="ResearchDirector",
    director_model_name="gpt-5.4",
    agents=[researcher, analyst, writer],
    parallel_execution=True,
    agent_as_judge=True,
    director_temperature=1.0,
    planning_enabled=False,
)

result = swarm.run("Produce a competitive analysis of the top 5 cloud providers")
# Console prints a rich Tree panel showing which agent gets which subtask
```

---

## 13. Documentation Fixes

**PR #1467** | Merged March 17, 2026 — Fixed 6 broken documentation links  
**PR #1450** | Merged March 15, 2026 — Fixed broken links for HierarchicalSwarm and ConcurrentWorkflow in README  
**PR #1447** | Merged March 14, 2026 — Added async subagent documentation  
**PR #1468** | Merged March 18, 2026 — Added PlannerWorkerSwarm reference docs, examples, and mkdocs nav entries  
**Commit `46aa5843`** — Added v10 changelog to docs  
**Commit `fddbd25d`** — Added 16-agent Grok Heavy mode to HeavySwarm documentation

---

## 14. Dependency Updates

| Package | From | To | PR |
|---|---|---|---|
| `ruff` | `<0.15.6` | `<0.15.9` | #1454, #1496, #1508 |
| `types-pytz` | `<2026.0` | `<2027.0` | #1453 |
| `litellm` | unpinned | pinned | manual |
| `requests` | implicit | explicit in pyproject | manual |

---

## 15. Full PR List

| PR | Title | Merged | Author |
|---|---|---|---|
| #1443 | feat: PlannerWorkerSwarm | 2026-03-17 | Steve-Dusty |
| #1447 | docs: async subagent docs | 2026-03-14 | Steve-Dusty |
| #1450 | fix: README dead links | 2026-03-15 | Steve-Dusty |
| #1453 | dep: bump types-pytz | 2026-03-16 | dependabot |
| #1454 | dep: bump ruff | 2026-03-16 | dependabot |
| #1467 | fix: 6 broken documentation links | 2026-03-17 | Steve-Dusty |
| #1468 | docs: PlannerWorkerSwarm reference | 2026-03-18 | Steve-Dusty |
| #1469 | fix: machine ID stability | 2026-03-18 | — |
| #1471 | fix: HierarchicalSwarm agent lookup | 2026-03-19 | Nishant-k-sagar |
| #1473 | fix: test machine ID stability | 2026-03-18 | adichaudhary |
| #1474 | fix: missing try/except wrapper module | 2026-03-18 | adichaudhary |
| #1476 | fix: AgentRearrange exception propagation | 2026-03-19 | Steve-Dusty |
| #1478 | refactor: consolidate rearrange tests | 2026-03-19 | Steve-Dusty |
| #1479 | fix: repeated-agent flow awareness | 2026-04-04 | Steve-Dusty |
| #1480 | fix: conversation cache refactor | 2026-03-23 | adichaudhary |
| #1481 | fix: GraphWorkflow max_loops | — | Steve-Dusty |
| #1487 | feat: autoswarm generates Python file | — | Steve-Dusty |
| #1489 | feat: autoswarm file writer + CLI | 2026-04-08 | Steve-Dusty |
| #1490 | fix: GraphWorkflow max_loops (merged) | 2026-04-02 | Steve-Dusty |
| #1496 | dep: bump ruff | 2026-03-23 | dependabot |
| #1497 | feat: GraphWorkflow serialization | 2026-04-05 | adichaudhary |
| #1498 | feat: GraphWorkflow checkpoint dir | 2026-04-08 | adichaudhary |
| #1499 | feat: Grok Heavy agents | 2026-03-30 | Steve-Dusty |
| #1500 | feat: PlannerGeneratorEvaluator | — | Steve-Dusty |
| #1501 | feat: PlannerGeneratorEvaluator (merged) | 2026-03-29 | Steve-Dusty |
| #1502 | fix: issue #1049 | 2026-04-14 | idreesaziz |
| #1508 | dep: bump ruff | 2026-03-30 | dependabot |
| #1509 | feat: 16-agent Grok Heavy mode | 2026-04-02 | Steve-Dusty |
| #1512 | feat: concurrent batch_run pipeline | 2026-04-07 | adichaudhary |
| #1522 | feat: DriftDetectionAgent | 2026-04-14 | adichaudhary |
| #1523 | fix: Anthropic temperature compat | 2026-04-11 | adichaudhary |
| #1526 | (closed by) AdvisorSwarm | — | — |
| #1532 | feat: AdvisorSwarm | 2026-04-14 | Steve-Dusty |
| #1544 | refactor: remove uvloop dead code | 2026-04-19 | Steve-Dusty |
| #1545 | fix: remove uvloop dead code | 2026-04-19 | MycCellium420 |
| #1546 | revert: revert #1545 | 2026-04-19 | kyegomez |

---

## Conclusion

This release shipped 3 new multi-agent structs, 26 improvements and refactors, and 13 bug fixes across 120 commits. The new orchestration patterns give developers more tools for decomposing hard problems across agents — whether that means a cost-aware advisor pairing, parallel task queues, or iterative quality loops. Alongside the new capabilities, existing structs got faster, more reliable, and more secure, with five vulnerabilities patched and over 13,000 lines of dead code removed.

Every one of these changes moves toward a single goal: building the best multi-agent framework in the world. That means not just adding capabilities, but making the existing ones more reliable, more composable, and safer to run in production. There is more to come.

---

*Generated from git history: March 13 – April 19, 2026. 120 commits, 35 PRs.*
