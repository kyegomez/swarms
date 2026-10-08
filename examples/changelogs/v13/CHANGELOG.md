# Changelog

All notable changes to the **Swarms** framework from **May 5, 2026** to **June 12, 2026**, covering releases `12.0.1` (Jun 1), `12.0.2` (Jun 7), and `12.0.3` (Jun 11) — 49 commits in total.

---

## ✨ New Features

### GroupChat — async self-selecting chat (v12.0.3)

- **Complete GroupChat rewrite** — `GroupChat` is now a fully asynchronous, self-selecting chat. No rounds or speaker-selection functions: every agent listens in parallel and decides on its own whether to chime in via a forced `respond(score, message)` call. Replies scoring above `threshold` are broadcast; the chat ends after `max_loops` messages or `idle_timeout` seconds of silence. (`604613ff`)
- **`RESPOND_TOOL` export** — the respond tool schema is exported from `swarms.structs.groupchat`. (`604613ff`)
- **Auto-equipped respond tool** — agents passed to `GroupChat` automatically receive `RESPOND_TOOL`; no manual wiring needed. (`e307b20e`)

```python
from swarms import Agent, GroupChat

optimist = Agent(agent_name="Optimist", system_prompt="Argue the upside.", model_name="gpt-5.4", max_loops=1)
pessimist = Agent(agent_name="Pessimist", system_prompt="Argue the risks.", model_name="gpt-5.4", max_loops=1)

chat = GroupChat(
    agents=[optimist, pessimist],  # RESPOND_TOOL is auto-injected since 12.0.3
    max_loops=10,       # hard cap on total messages
    threshold=0.5,      # min desire-to-speak score (0..1) to publish
    idle_timeout=8.0,   # seconds of silence before the chat ends
)
result = chat.run("Should we adopt AI for medical diagnosis?")
```

### GraphWorkflow — composition and parallelism (PRs #1620, #1623, #1605 by @adichaudhary)

- **Nested subgraph composition** — embed a whole `GraphWorkflow` as a node inside another workflow, with spec serialisation and checkpoint nesting. (`30d05356`, `66285ab6`, `436ffa8b`, merged `9cde6643`)
- **`validate(raise_on_error)`** — compile-time structural validation of the graph. (`93232ab6`, merged `6063c332`)
- **`max_parallel_nodes` constructor parameter** — caps how many nodes execute concurrently. (`ddb7085b`, merged `56671fe2`)

```python
from swarms import Agent, GraphWorkflow, Node, Edge

# Inner workflow becomes a single node in the outer one
inner = GraphWorkflow(name="research")
inner.add_node(Node.from_agent(Agent(agent_name="Researcher", model_name="gpt-5.4", max_loops=1)))

outer = GraphWorkflow(max_parallel_nodes=4)        # at most 4 nodes run at once
outer.add_node(Node.from_subgraph(inner))          # nested subgraph node
outer.add_node(Node.from_agent(Agent(agent_name="Writer", model_name="gpt-5.4", max_loops=1)))
outer.add_edge(Edge(source="research", target="Writer"))

outer.validate(raise_on_error=True)                # fail fast on cycles, orphans, missing entry points
result = outer.run(task="Write a brief on AI chips.")
```

### Streaming across workflows

- **HierarchicalSwarm streaming** — `arun_stream` / `run_stream` with full token streaming across director, workers, and aggregator. (PR #1611 by @Steve-Dusty: `eb8e7ff6`, `d8d8563b`, merged `c83735a0`)
- **AgentRearrange streaming** — `arun_stream` and `run_stream` methods. (`f0e35550`)
- **SequentialWorkflow streaming** — stream tokens from each agent in sequence; `with_events=True` yields structured `agent_start` / `token` / `agent_end` events. (`f0e35550`, `b7913a0d`, `d0b60164`)

```python
from swarms import Agent, SequentialWorkflow

pipeline = SequentialWorkflow(agents=[
    Agent(agent_name="Researcher", model_name="gpt-5.4", max_loops=1),
    Agent(agent_name="Writer", model_name="gpt-5.4", max_loops=1),
])

# Plain token stream
for token in pipeline.run_stream("Summarise LLM research this year."):
    print(token, end="", flush=True)

# Structured events (for per-agent UI panels)
for event in pipeline.run_stream("Same task.", with_events=True):
    ...  # {"type": "agent_start" | "token" | "agent_end", ...}
```

### AgentRearrange — flow DSL upgrades (v12.0.1)

- **Pure concurrent flows** — flows may consist solely of comma-separated agents, no `->` required. (`cc9a35d9`)
- **`explain()` method** — prints the parsed execution plan. (`cc9a35d9`)
- **Team awareness** — agents are told who else is in the flow via the system prompt. (`cc9a35d9`)

```python
from swarms import Agent, AgentRearrange

a = Agent(agent_name="A", model_name="gpt-5.4", max_loops=1)
b = Agent(agent_name="B", model_name="gpt-5.4", max_loops=1)
c = Agent(agent_name="C", model_name="gpt-5.4", max_loops=1)

flow = AgentRearrange(agents=[a, b, c], flow="A, B, C")  # all three run in parallel — no "->" needed
flow.explain()                                            # print the execution plan
result = flow.run("Brainstorm product names.")
```

### RoundRobinSwarm — true rotation (v12.0.2)

- **True round-robin rewrite** — deterministic rotation replaces the previous shuffle-based ordering. (`2e258dfa`)
- **Turn awareness** — each agent receives previous/next speaker context. (`2e258dfa`)

```python
from swarms import Agent, RoundRobinSwarm

agents = [Agent(agent_name=f"Handler-{i}", model_name="gpt-5.4", max_loops=1) for i in range(3)]
rr = RoundRobinSwarm(agents=agents, max_loops=1)  # fixed order: Handler-0 -> Handler-1 -> Handler-2
result = rr.run("Review this proposal.")          # each agent knows who spoke before and who is next
```

### HeavySwarm — `variant` parameter

- **`variant` parameter** — `"default"`, `"medium"`, or `"heavy"`; replaces the old grok-specific flags. (`f599c2be`)

```python
from swarms import HeavySwarm

swarm = HeavySwarm(variant="heavy")  # was: grok-specific boolean flags
result = swarm.run("Deep analysis of the AI chip market.")
```

### CLI

- **Rotating tips command** and shared tips module; **model discovery** backed by LiteLLM; **typo correction** suggesting the closest command; **error hints** with recovery classification; **contextual next-step tip** after `init` and `setup-check`. (`517fb20a`)

```bash
swarms tips      # rotating usage tips
swarms models    # discover available models via LiteLLM
swarms inti      # typo -> "Did you mean: init?"
```

### Framework internals

- **Shared `find_agent_by_id` helper** — id-based lookup used across registry, base swarm, and utils. (`a6302e45`)
- **Cached name index for `find_agent_by_name`** — replaces repeated linear scans. (`1ff794f5`)
- **Reusable `SerializableMixin`** — `to_dict` extracted into a mixin shared across multi-agent structures (`03962415`), with a verbose log helper added (`f599c2be`).

```python
from swarms.structs.ma_blocks import find_agent_by_id, find_agent_by_name

agent = find_agent_by_id(agents, agent_id="agent-123")
agent = find_agent_by_name(agents, agent_name="Analyst")   # now backed by a cached index

# Any structure inheriting SerializableMixin (GroupChat, HeavySwarm, AgentRearrange, ...)
state = chat.to_dict()
```

### New examples & guides

- Dynamic groupchat and router-driven groupchat examples. (`604613ff`, `e307b20e`)
- Fable single-agent example. (`604613ff`)
- Round-robin scenario examples: ETF investment committee, medical tumor board, engineering design review. (`2e258dfa`)
- AgentRearrange topology examples: fan-out/fan-in, multi-stage, diamond flow with real agents, new-features showcase, dedicated examples folder. (`cc9a35d9`, `6c6f7980`)
- HeavySwarm research and variant examples; SwarmRouter concurrent-workflow example. (`f599c2be`)
- vLLM and o3 model examples with a models README. (`6f98a5f4`)
- Smoke-test example scripts for refactored modules. (`1ff794f5`)
- ContextCompressor example file. (`073623d7`)
- **`CLAUDE.md`** — repository guide for building agents with the framework, plus reflexion prompts. (`74d81fad`)

---

## 🔧 Improvements

### API & behaviour changes

- **Speaker-function API removed** from `SwarmRouter` and package exports — superseded by the self-selecting GroupChat. (`604613ff`)

  ```python
  # Before: SwarmRouter(..., speaker_function=round_robin_speaker)
  # After: just pick the swarm type — GroupChat agents decide for themselves
  router = SwarmRouter(agents=agents, swarm_type="GroupChat")
  ```

- **`Agent.tools_list_dictionary` defaults to an empty list** instead of `None`. (`604613ff`)

  ```python
  agent = Agent(agent_name="A", model_name="gpt-5.4")
  agent.tools_list_dictionary  # [] — append schemas without a None check
  ```

- **Auto-prompt-engineering (APE) logic removed** from SwarmRouter. (`6f98a5f4`)
- **Unused `rules` constructor argument removed.** (`03962415`, `f599c2be`)
- **Dead human-in-the-loop code removed.** (`f962f781`)
- **Unused stopping-condition helpers removed.** (`a6302e45`)
- **`get_all_agent_names` renamed to `return_all_agent_names`.** (`a6302e45`)

  ```python
  from swarms.structs.ma_blocks import return_all_agent_names  # was get_all_agent_names
  names = return_all_agent_names(agents)
  ```

- **`set_custom_flow` validates the flow immediately** on update instead of at run time. (`cc9a35d9`)
- **AgentRearrange output-type validation** — `output_type` checked against the allowed literal set. (`cc9a35d9`)

  ```python
  AgentRearrange(agents=[a, b], flow="A -> B", output_type="dict")   # ok
  AgentRearrange(agents=[a, b], flow="A -> B", output_type="bogus")  # raises immediately
  ```

### Performance & internals

- **Cached lookups everywhere** — `AgentRearrange` and `SwarmRouter` store agents as lists and reuse the shared cached lookup; `AgentRouter`'s dual linear scans replaced; social-algorithms duplicate agent map dropped. (`1ff794f5`)
- **Shared executor reuse** — inline thread pools replaced with the shared agent executor. (`f599c2be`)
- **Telemetry removed from the hot path** — `log_agent_data` calls dropped from `run`. (`03962415`)
- **Ineffective conversation string cache reverted.** (`03962415`)
- **HeavySwarm modularised** — question agents and dashboard extracted into their own modules. (`f599c2be`)
- **SwarmRouter cleanup** — unused params removed, message helpers extracted, class documented, inherits `SerializableMixin`. (`6f98a5f4`, `f599c2be`)
- **SelfMoA refactor** — inherits `SerializableMixin`, drops the tenacity retry decorator, moves parameter validation into setup, routes all logging through the inherited `_log`. (`a6302e45`)
- **RoundRobinSwarm refactor** — inherits `SerializableMixin`, drops its local `_log` and tenacity retry, deletes an unused callback hook, gates logs behind `verbose`, extracts prompts to module functions, consolidates init validation. (`2e258dfa`, `a6302e45`)

  ```python
  rr = RoundRobinSwarm(agents=agents, verbose=True)  # logs only when verbose=True
  ```

- **Hardened GraphWorkflow subgraph execution** — scoped dict flattening, checkpoint isolation, parameter forwarding, type-annotation cleanup. (`7ec0a274`)
- **Dead commented-out blocks removed from MixtureOfAgents.** (`a6302e45`)

### Documentation

- GroupChat docs and guides fully rewritten for the new async API. (`604613ff`)
- MixtureOfAgents docstrings rewritten. (`a6302e45`)
- Full docstrings added to `multi_agent_router` functions. (`1ff794f5`)
- GraphWorkflow subgraph docstrings and constructor docs. (`436ffa8b`)
- Performance audit document added, then relocated into `docs/`. (`f599c2be`, `1ff794f5`)
- README updates: hiring section, general refreshes, new examples listed in example READMEs. (`75c76d2f`, `be527575`, `069c2d47`, `e307b20e`)
- General docs improvements and docs-repo cleanup. (`ac59f952`, `9d07113c`)
- Multi-stage example output simplified; example task prompt refreshed. (`03962415`, `a6302e45`)

### Tests

- GroupChat test suite rewritten for the new async API. (`604613ff`)
- HeavySwarm tests updated for the `variant` API. (`f599c2be`)
- Run tests added covering every `SwarmRouter` swarm type; obsolete router tests removed. (`6f98a5f4`)
- **ContextCompressor test suite** (PR #1625 by @adichaudhary) — unit tests (`19b91b3f`), integration-test isolation fixed with stronger archive/`MEMORY.md` assertions (`72b59806`), tests switched to real agents (`073623d7`), merged `697760b0`.
- New tests for AgentRearrange flow and awareness behaviour. (`cc9a35d9`)
- Test-file consolidation — multi-agent-router, LiteLLM wrapper, agent-rearrange, artifact, and heavy-swarm test files each merged into single suites. (`64ccdc9e`, `f962f781`)

### Repository hygiene

- macOS `.DS_Store` files git-ignored; misspelled directories (`hiearchical`, `multii`, `swarmarrange`) renamed; council example folders merged; `hscf` folded into `hscf_examples`; `duo_agent` relocated; stale `tree_swarm_new_updates` file and byte-identical macOS duplicates deleted. (`7d6cc333`)
- Models and tools examples reorganised; stale UI and multi-MCP guide examples removed; `.gitignore` cleaned up. (`6f98a5f4`)
- Examples unified on top-level `from swarms import ...` imports. (`2e258dfa`)

  ```python
  # Before: from swarms.structs.agent import Agent
  # After:
  from swarms import Agent
  ```

- Code formatting passes on `swarms/` and agent-rearrange examples. (`0be1448d`, `57afc6be`)
- GitHub workflows cleaned up. (`7b3eedc7`)
- Unused images removed in three sweeps. (`919cee4d`, `b1e01830`, `79f459b2`)
- Interactive script moved into `examples/`. (`f962f781`)
- Default example model updated to `gpt-5.4-mini`. (`5861ab86`)
- Round-robin examples given diverse model names. (`24107e7a`)
- Branch syncs: upstream master merged into graph-workflow-composition. (`71ef5940`, `f5d7d7fe`)
- Version bumps: **12.0.1** (`6c6f7980`), **12.0.2** (`2e258dfa`), **12.0.3** (`e307b20e`).

---

## 🐛 Bug Fixes

- **SwarmRouter → GroupChat parameter forwarding** — `output_type` and `verbose` are now passed through to `GroupChat`. (`e307b20e`)

  ```python
  router = SwarmRouter(agents=agents, swarm_type="GroupChat", output_type="dict", verbose=True)
  # both settings now actually reach the underlying GroupChat
  ```

- **MajorityVoting streaming callback errors are re-raised** instead of being silently swallowed. (`a6302e45`)
- **Brittle `deepcopy` in AgentRearrange batch runs** replaced with a safe per-task clone. (`cc9a35d9`)
- **Thinking panel respects `print_on`** — thinking output no longer prints when `print_on=False`. (`f0e35550`)
- **`grok_schema` import path corrected.** (`f962f781`)
- **Byte-identical macOS duplicate files deleted** from the examples tree. (`7d6cc333`)
- **Integration-test isolation fixed** for the ContextCompressor suite. (`72b59806`)
- Extra blank line removed from reflexion prompts. (`f0e35550`)

---

## Commit index

| Date | Commit | Author | Summary |
| --- | --- | --- | --- |
| 2026-06-11 | `e307b20e` | Kye Gomez | GroupChat auto-equips respond tool; router forwards output_type/verbose; new examples; v12.0.3 |
| 2026-06-09 | `604613ff` | Kye Gomez | Async self-selecting GroupChat rewrite; RESPOND_TOOL export; speaker functions removed; docs/tests rewritten |
| 2026-06-09 | `ac59f952` | Kye Gomez | Docs improvements |
| 2026-06-08 | `069c2d47` | Kye Gomez | README update |
| 2026-06-08 | `a6302e45` | Kye Gomez | Shared id lookup; SelfMoA/RoundRobin mixin refactors; majority-voting callback fix |
| 2026-06-08 | `1ff794f5` | Kye Gomez | Cached agent-name lookup across rearrange/router/agent-router; router docstrings |
| 2026-06-08 | `f599c2be` | Kye Gomez | HeavySwarm modular refactor + variant param; SwarmRouter cleanup; perf audit |
| 2026-06-07 | `24107e7a` | Kye Gomez | Diverse model names in round-robin examples |
| 2026-06-07 | `2e258dfa` | Kye Gomez | True round-robin rewrite with turn awareness; committee/tumor-board/design-review examples; v12.0.2 |
| 2026-06-05 | `6f98a5f4` | Kye Gomez | SwarmRouter cleanup, APE removal, all-swarm-type tests; examples reorg |
| 2026-06-05 | `03962415` | Kye Gomez | Serializable mixin extraction; telemetry and cache removal |
| 2026-06-03 | `9cde6643` | Kye Gomez | Merge PR #1620 — GraphWorkflow nested subgraph composition |
| 2026-06-03 | `75c76d2f` | Kye Gomez | Hiring section in README |
| 2026-06-03 | `0be1448d` | Kye Gomez | Format `swarms/` folder |
| 2026-06-02 | `71ef5940` | adichaudhary | Merge master into graph-workflow-composition |
| 2026-06-02 | `6063c332` | Kye Gomez | Merge PR #1623 — GraphWorkflow validation |
| 2026-06-02 | `57afc6be` | Kye Gomez | Format agent-rearrange examples |
| 2026-06-02 | `be527575` | Kye Gomez | README update |
| 2026-06-01 | `7d6cc333` | Kye Gomez | Directory-typo renames, folder consolidation, duplicate-file cleanup |
| 2026-06-01 | `6c6f7980` | Kye Gomez | Agent-rearrange examples; v12.0.1 |
| 2026-06-01 | `cc9a35d9` | Kye Gomez | AgentRearrange: deepcopy fix, pure concurrent flows, team awareness, explain(), tests |
| 2026-05-27 | `c83735a0` | Kye Gomez | Merge PR #1611 — HierarchicalSwarm streaming |
| 2026-05-21 | `517fb20a` | Kye Gomez | CLI: tips, model discovery, typo correction, error hints |
| 2026-05-20 | `7b3eedc7` | Kye Gomez | Clean up workflows |
| 2026-05-20 | `56671fe2` | Kye Gomez | Merge PR #1605 — GraphWorkflow parallel execution |
| 2026-05-20 | `697760b0` | Kye Gomez | Merge PR #1625 — ContextCompressor tests |
| 2026-05-19 | `72b59806` | adichaudhary | Fix integration-test isolation; strengthen archive/MEMORY.md assertions |
| 2026-05-19 | `19b91b3f` | adichaudhary | ContextCompressor unit test suite |
| 2026-05-19 | `073623d7` | adichaudhary | ContextCompressor example; tests use real agents |
| 2026-05-18 | `93232ab6` | adichaudhary | GraphWorkflow `validate(raise_on_error)` + compile-time validation |
| 2026-05-18 | `7ec0a274` | adichaudhary | Harden subgraph execution (scoping, checkpoint isolation, param forwarding) |
| 2026-05-18 | `f5d7d7fe` | adichaudhary | Merge upstream master into graph-workflow-composition |
| 2026-05-18 | `30d05356` | adichaudhary | GraphWorkflow nested subgraph composition, spec serialisation, checkpoint nesting |
| 2026-05-14 | `d8d8563b` | Steve-Dusty | Full `arun_stream` with director/worker/aggregator streaming |
| 2026-05-14 | `64ccdc9e` | Kye Gomez | Merge router and LiteLLM-wrapper test files |
| 2026-05-14 | `f962f781` | Kye Gomez | Remove dead HITL code; merge test files; fix grok_schema import |
| 2026-05-13 | `eb8e7ff6` | Steve-Dusty | Add `arun_stream`/`run_stream` to HierarchicalSwarm |
| 2026-05-13 | `436ffa8b` | adichaudhary | Subgraph docstrings and constructor implementation |
| 2026-05-13 | `66285ab6` | adichaudhary | Sub-workflows created |
| 2026-05-12 | `ddb7085b` | adichaudhary | Add `max_parallel_nodes` to GraphWorkflow constructor |
| 2026-05-09 | `d0b60164` | Kye Gomez | Sequential workflow streaming |
| 2026-05-09 | `b7913a0d` | Kye Gomez | Streaming sequential workflow |
| 2026-05-08 | `5861ab86` | Kye Gomez | Change example model name to gpt-5.4-mini |
| 2026-05-07 | `919cee4d` | Kye Gomez | Clean up unused images |
| 2026-05-07 | `b1e01830` | Kye Gomez | Clean up unused images |
| 2026-05-07 | `79f459b2` | Kye Gomez | Clean up unused images |
| 2026-05-07 | `9d07113c` | Kye Gomez | Clean up docs repo |
| 2026-05-07 | `f0e35550` | Kye Gomez | Rearrange + sequential streaming methods; thinking-panel print fix |
| 2026-05-06 | `74d81fad` | Kye Gomez | Add CLAUDE.md; reflexion prompts |
