# Waste audit

What can come out of `swarms/` without losing capability.

Measured against `master` at `089e1f97`. Baseline: **77,856 lines** across **218 files**.

| directory | lines |
| --- | --- |
| `swarms/structs/` | 41,412 |
| `swarms/prompts/` | 8,551 |
| `swarms/tools/` | 7,979 |
| `swarms/agents/` | 7,641 |
| `swarms/utils/` | 6,684 |
| `swarms/cli/` | 3,380 |
| `swarms/telemetry/` | 1,030 |
| `swarms/schemas/` | 788 |
| `swarms/artifacts/` | 362 |

Two things worth saying up front, because they shape how to read the rest.

**Literal copy-paste is not the problem.** A cross-file duplicate-block scan (10-line
window, whitespace-normalised) finds exactly one duplicate of any size in the whole
package: `prompts/sales.py:1-42` is a byte-identical copy of
`prompts/sales_prompts.py:47-88`. The recent unification passes did their job. What
is left is *structural* duplication — the same function written N times with different
names, and one concept given four implementations.

**22% of the package is docstrings** (17,037 lines). That is not itself waste. But
`litellm_wrapper.py` is 42% docstring and `agent_rearrange.py` is 38%, which is worth
knowing when a line count looks alarming.

---

## Summary

| # | area | reclaimable | risk |
| --- | --- | --- | --- |
| 1 | Dead code — never imported, never called | ~3,600 | none |
| 2 | Same function written N times | ~1,170 | low |
| 3 | ~~`graph_workflow.py` — serialization, validation, edges, backends~~ | **done (−222)** | — |
| 4 | `base_tool.py` — provider logic written three times | ~400 | medium |
| 5 | `aop.py` — queue management written three times | ~220 | low |
| 6 | `Conversation` — alias sprawl | ~150 | low |
| 7 | `Agent` — serialization cluster | ~80 | low |
| | **total remaining** | **~5,470 (7%)** | |

Another ~1,500 is available if the three dashboards merge and the two
500-line methods (`graph_workflow.run`, `hiearchical_swarm.arun_stream`) get
decomposed, but those are refactors rather than deletions.

---

## 1. Dead code — ~3,600 lines, zero risk

### Orphaned prompt files: 3,205 lines, 36 files

Nothing imports these. Not `prompts/__init__.py` (which lists its imports
explicitly), not the package, not `examples/`, not `tests/`, not the docs.

```
python(286)                    autobloggen(276)               worker_prompt(211)
aga(185)                       programming(176)               agent_prompts(160)
xray_swarm_prompt(104)         agent_self_builder_prompt(103) self_operating_prompt(102)
multi_modal_prompts(101)       sales(98)                      support_agent_prompt(97)
tests(95)                      ai_research_team(91)           accountant_swarm_prompts(90)
sop_generator_agent_prompt(88) sales_prompts(88)              autoswarm(83)
project_manager(78)            agent_orchestration_prompt(75) prompt_generator(70)
code_spawner(62)               react(58)                      swarm_manager_agent(54)
meta_system_prompt(52)         multi_modal_visual_prompts(48) debate(44)
visual_cot(41)                 urban_planning(39)             personal_stylist(38)
education(34)                  security_team(28)              aot_prompt(23)
idea2img(14)                   task_assignment_prompt(13)     refiner_agent_prompt(0)
```

Watch for near-miss names when checking these: `worker_prompt.py` is *not*
`planner_worker_prompts.py` (which is live), and `prompt_generator.py` is not
`prompt_generator_optimizer.py` (also live).

### Orphaned modules: 306 lines

- `swarms/structs/various_alt_swarms.py` (240) — no references anywhere in the repo.
- `swarms/structs/collaborative_utils.py` (66) — same.
- ~~`swarms/structs/utils.py` (35)~~ — **removed.** Wrapped `ma_blocks.find_agent_by_id`
  and was imported by nothing; it aliased its own import to dodge a name collision
  with itself.

### Dead branches and parameters inside live files

- `agent.py:2722` — `_save_additional_components` saves `self.memory_manager`.
  Nothing ever sets that attribute. The branch cannot execute.
- `base_tool.py:2029-2119` — `get_schema_provider_format` and
  `convert_schema_between_providers`, 91 lines, zero callers in the package, in
  `examples/`, or in `tests/`.
- ~~`Agent.__init__` `retry_interval` and `tokenizer`~~ — **removed.** Both were
  assigned and never read anywhere in `swarms/`. Six lines out of `agent.py`
  (parameter, assignment, and docstring row for each), plus the call sites in
  `tests/structs/test_agent.py`, five example scripts, and one guide.

  Note for anyone auditing the removal: `Agent.__init__` ends in `**kwargs`, which it
  accepts and discards, so external code still passing `retry_interval=1` gets no
  `TypeError` — the argument is silently ignored. What disappears is the
  `agent.retry_interval` / `agent.tokenizer` attribute. That is the right trade here
  since neither attribute did anything, but it is a silent behaviour change rather
  than a loud one.

---

## 2. Same function written N times — ~1,170 lines

### `reliability_check` × 19 — 562 lines

`swarm_router.py:411` · `agent.py:2559` · `heavy_swarm.py:219` ·
`hiearchical_swarm.py:614` · `agent_rearrange.py:209` · `sequential_workflow.py:180` ·
`cron_job.py:132` · `concurrent_workflow.py:219` · `auto_swarm_builder.py:314` ·
`spreadsheet_swarm.py:98` · `advisor_swarm.py:125` ·
`planner_generator_evaluator.py:274` · `mixture_of_agents.py:132` ·
`reasoning_agent_router.py:116` · `majority_voting.py:164` · `swarm_rearrange.py:97` ·
`council_as_judge.py:294` · `agent_judge.py:229` · `planner_worker_swarm.py:594`

Three spellings for one method — `reliability_check`, `reliability_checks`,
`_reliability_checks` — and most bodies assert the same three things: agents
non-empty, `max_loops != 0`, optionally append `AGENT_COLLAB_PROMPT`.

The user-visible cost is inconsistency. The same mistake — constructing a swarm with
no agents — raises `"Agents list cannot be None or empty"`,
`"No agents provided."`, or `"Agents list is empty"` depending on which class you
happened to reach for.

A shared `validate_swarm(agents, max_loops, ...)` plus per-class extras cuts ~350
lines and makes one failure produce one message.

### The async-to-sync generator bridge × 3 — ~150 lines

`agent_rearrange.py:1332` · `hiearchical_swarm.py:2263` · `agent.py:3464`

Identical each time: daemon thread, `queue.Queue`, `DONE` sentinel, `exc_holder` list,
re-raise after join. One `sync_iter_from_async(agen_factory)` helper of about 25 lines
replaces all three.

### Batch runners that never moved to the shared helper

`execution_utils.batched_run` exists and 18 files use it. Eight did not convert:

| file | method |
| --- | --- |
| `agent.py:3655` | `run_batched` — reimplements the img-pairing logic verbatim |
| `agent.py:2455` | `run_concurrent_tasks` — is `batched_run(max_workers=...)` |
| `agent.py:2489` | `bulk_run` |
| `agent_judge.py:380` | `run_batched` |
| `multi_agent_router.py:461` | `batch_run` |
| `cron_job.py:334` | `batched_run` |
| `sequential_workflow.py:453` | `run_batched` |
| `concurrent_workflow.py:639` | `batch_run` |
| `hiearchical_swarm.py:1693` | `batched_run` |
| `model_router.py:286` | `batch_run` |

Three public spellings survive for one concept: `batch_run`, `batched_run`,
`run_batched`. Worth picking one and aliasing the others.

### Three dashboards, plus a fourth inline

`formatter.py:660-741` · `utils/hierarchical_swarm_dashboard.py` (563) ·
`utils/heavy_swarm_dashboard.py` (313) · inline at `concurrent_workflow.py:248`.
The spinner frame list alone is copy-pasted — `formatter.py:316` is
`hierarchical_swarm_dashboard.py:54`.

---

## 3. `graph_workflow.py` — 3,997 → 3,776 lines — **done**

Consolidated. Executable code dropped 106 lines (2,674 → 2,568) while every public
name kept working, and two latent bugs fell out in the process.

**What replaced what.** There were never four serialization systems — there were two
*shapes* (shallow topology vs. deep export with agents embedded) each reachable three
ways (dict, string, file), plus a duplicated validator. Now:

| new | absorbs |
| --- | --- |
| `to_dict(shallow=...)` | the node/edge/agent builders that `to_spec` and `to_json` each had their own copy of |
| `_write_json(...)` | the makedirs / open / dump boilerplate in `save_spec` and `save_to_file` |
| `save(path, shallow=...)` / `load(path, agent_registry=...)` | one obvious pair; `load` sniffs the shape so one call reads both |
| `_structural_checks(...)` | the six checks `validate` and `_fast_validate` each implemented separately |
| `_fan_patterns()` / `_branching()` | the fan-out/fan-in grouping, written **three** times across `visualize` and `visualize_simple` |
| `_attach_edge()` / `_add_edge_pairs()` | the validate-append-register block, written **four** times across `add_edge`, `add_edges_from_source`, `add_edges_to_target`, `add_parallel_chain` |
| `_task_key()` / `_checkpoint_path()` / `_safe_output()` | duplicated checkpoint-key derivation, path building, and per-agent error handling inside the 480-line `run` |

Separately, 90 lines of docstring came out of `NetworkXBackend` and `RustworkxBackend`
where they restated the `GraphBackend` ABC **verbatim**. Only byte-identical ones were
dropped; the eleven that add implementation detail stayed. `inspect.getdoc` walks the
MRO, so `help()` and Sphinx still show the contract.

`to_spec`, `save_spec`, `from_topology_spec`, `to_json`, `from_json`, `save_to_file`
and `load_from_file` all still exist as thin wrappers — each has 11–20 call sites
across `examples/` and the docs, so deleting them would have been a breaking change
for no benefit.

**Two bugs this surfaced**, both in the deep path, both previously untested:

1. `from_json` passed deserialized agent *dicts* into `from_spec`, which derives node
   ids from live `Agent` objects. Ids came out wrong, so every edge failed with
   `Source node 'X' does not exist`. `to_json` → `from_json` had never round-tripped,
   and neither had `save_to_file` → `load_from_file`.
2. `to_json` wrote node types as `str(node.type)` — `"NodeType.AGENT"` — which
   `NodeType()` refuses to parse. Loading would have failed on the next line anyway.

Both are fixed; `_parse_node_type` accepts the legacy spelling so files written by
older versions still load. Twelve characterization tests now pin the round-trips
(`tests/structs/test_graph_workflow.py`), up from zero.

**Still open in this file:**

- `run` is still ~500 lines. The duplicated pieces are out, but the layer loop
  itself would want splitting into `_run_layer` / `_resume_layer` to go further.
- `visualize` is ~260 lines of graphviz styling.
- `export_summary` (87 lines) overlaps `to_dict(shallow=False)`, but its key names
  differ (`from`/`to` vs `source`/`target`); routing it through `to_dict` would cost
  more in remapping than it saves.
- **`RustworkxBackend` deserves a deliberate decision rather than a deletion.** It is a
  documented opt-in backend (`SKILL.md:368`), so it is not dead. But `rustworkx` is in
  neither `pyproject.toml` nor `requirements.txt`, so no default install and no CI run
  ever executes those 315 lines — including the hand-rolled Kahn's algorithm at `:549`.
  Either it becomes a real extra with a CI job, or it goes.

## 4. `base_tool.py` — 3,077 lines, provider logic written three times

Every operation forks into openai / anthropic / generic variants that differ by a
dict key:

- `_validate_{openai,anthropic,generic,json}_schema` — 320 lines, `:1709-2028`
- `_extract_{openai,anthropic,generic}_function_calls` — 275 lines, `:2392-2666`
- `_build_{openai,anthropic,generic}_schema` plus extractors — 74 lines, `:2120-2193`

A per-provider field-map table collapses these to roughly 250 lines total.

Context that matters before touching this file: of `BaseTool`'s 27 public methods, the
package itself calls **four** — `base_model_to_dict`,
`execute_function_calls_from_api_response`, `convert_tool_into_openai_schema`,
`multi_base_models_to_dict`. The other 23 are user-facing API exercised only from
`examples/`. That is legitimate, but it means ~2,000 lines carry no internal test
pressure, so a refactor there needs tests written first.

---

## 5. `aop.py` — 2,975 lines

Currently **has no test coverage at all**. Its test file was removed in #2016 because
it imported `mcp.server.fastmcp`, which does not exist in the pinned `mcp` release, and
the import error aborted the entire `test` CI job at collection time. It is also absent
from `swarms/structs/__init__.py`, so `from swarms import AOP` does not work today.
Both are worth settling before any refactor here.

The duplication itself is mechanical:

- `pause_agent_queue` / `resume_agent_queue` / `clear_agent_queue` — `:1503-1594`,
  92 lines. Same guard, same lookup, same try/except; one differing call.
- `pause_all_queues` / `resume_all_queues` / `clear_all_queues` — `:1688-1751`,
  64 lines. The same loop three times.
- `_register_queue_management_tools` — `:2106-2291`, 186 lines, wrapping each of the
  above in a near-identical MCP tool registration.

One `_queue_op(name, action)` plus a registration table takes ~340 lines to ~120.

---

## 6. `Conversation` — 1,586 lines, alias sprawl

- `search` (`:685`) and `search_keyword_in_conversation` (`:1065`) are the same list
  comprehension. The second crashes on non-str content where the first does not.
- `add_multiple_messages` (`:619`) calls `add_multiple` (`:629`), which returns
  `None` — so the wrapper's `added` is always `None` and its return value is
  meaningless. `add_multiple` also spins up a `ThreadPoolExecutor` sized to 25% of
  CPUs in order to append to a Python list: pure overhead, and unsynchronised appends
  to `conversation_history`.
- Ten `return_*` / `to_*` accessors (`:1269`, `:1280`, `:1304`, `:1314`, `:1324`,
  `:1335`, `:1411`, `:1418`, `:1208`, `:1224`) exist only to feed the 15-branch
  `if/elif` in `utils/history_output_formatter.py`. A `{output_type: lambda}` table
  deletes most of them.

`clear_memory`, which was byte-identical to `clear`, was removed in #2016.

**Two live bugs are visible in this cluster** and should be fixed whether or not the
refactor happens:

- `return_dict_final` (`:1411`) returns a *tuple of the same value twice*, despite the
  name and its `-> dict` contract in `history_output_formatter`.
- `return_all_except_first` (`:1335`) slices `[2:]`, dropping two messages, not one.

---

## 7. `Agent` — 4,473 lines

**Nine ways to serialize:** `to_dict`, `to_json`, `to_yaml`, `to_toml`,
`model_dump_json`, `model_dump_yaml`, `save_to_yaml`, `save`, `get_saveable_state`.
`model_dump_json` and `model_dump_yaml` (`:3081-3124`) are the same 20-line body with
the extension swapped, and both names falsely promise Pydantic semantics while
actually writing files to disk.

**`__init__` is 406 lines: 90 parameters, 124 attribute assignments.** Several
clusters are the same knob more than once:

| cluster | parameters |
| --- | --- |
| MCP | `mcp_url`, `mcp_urls`, `mcp_config`, `mcp_configs`, `mcp_api_key`, `mcp_authorization_token`, `mcp_oauth`, `mcp_headers`, `mcp_transport`, `mcp_timeout` — 10, where `MCPConnection` already exists in `swarms/schemas/` |
| output | `streaming_on`, `stream`, `streaming_callback`, `print_on`, `verbose`, `dashboard` |
| stopping | `stopping_condition`, `stopping_token`, `preset_stopping_token`, `stopping_func`, `custom_exit_command` |
| tools | `tools`, `tool_schema`, `tools_list_dictionary`, `list_base_models`, `dynamic_tools`, `selected_tools`, `think_tool` |
| model | `llm`, `model_name`, `llm_args`, `llm_base_url`, `llm_api_key`, `fallback_model_name`, `fallback_models`, `random_models_on` |

This is the largest single design-level cost in the codebase, and the least safe to
change — every one of those parameters is public API. Grouping them behind config
objects while keeping the flat kwargs as deprecated aliases is the only realistic path.

---

## Suggested order

1. **Section 1 in one pass.** Pure deletion, mechanically verifiable, no behaviour
   change. ~3,600 lines, and it shrinks the surface everything else has to be checked
   against.
2. **The two `Conversation` bugs** (`return_dict_final`, `return_all_except_first`).
   Small, and they are wrong today regardless of any refactor.
3. **`reliability_check` unification.** Touches the most files, so it wants a quiet
   moment, but it is the change users would actually feel — one error message instead
   of three for the same mistake.
4. **The async-to-sync bridge helper.** Small, self-contained, three call sites.
5. **`aop.py`** — but first decide whether it is supported. It has no tests and no
   export; refactoring an untested 2,975-line module is not worth doing until that
   question is answered.
6. ~~**`graph_workflow` serialization**~~ — done. **`base_tool` providers** remain,
   and want tests written before the refactor, not after.
