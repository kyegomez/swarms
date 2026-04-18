# Swarms Codebase Audit — Deletion & Cleanup Candidates

> Updated: 2026-04-14
> Scope: `swarms/agents/`, `swarms/structs/`, `swarms/utils/`, `swarms/tools/`, `swarms/prompts/`

---

## Already Deleted

These have been removed from the codebase since the initial audit.

| File | Notes |
|---|---|
| `swarms/structs/board_of_directors_swarm.py` | Deleted — source, tests, docs, examples, and simulation files all removed |
| `swarms/structs/swarm_templates.py` | Deleted — zero callers, doc removed |
| `swarms/structs/maker.py` | Deleted — source, tests, docs, examples removed |
| `swarms/agents/openai_assistant.py` | Deleted — source and example removed |
| `swarms/agents/react_agent.py` | Deleted — moved to examples as tutorial |
| `swarms/structs/agent_grpo.py` | Deleted — source, tests, examples removed |
| `swarms/utils/check_all_model_max_tokens.py` | Deleted — only re-exported, never consumed |
| `swarms/utils/data_to_text.py` | Deleted — callers updated to use stdlib |
| `swarms/utils/index.py` | Deleted then restored — actively used by `agent.py` and `mcp_client_tools.py` |

---

## 1. DEAD CODE — Safe to Delete

Zero callers anywhere in the codebase, not exported, no docs.

| File | Lines | Contents | Why Delete |
|---|---|---|---|
| `swarms/structs/agent_roles.py` | 35 | Single `Literal` type with ~30 role strings | Never imported, not exported, zero callers |
| `swarms/structs/concat.py` | 28 | `concat_strings()` — wraps `str.join()` | Trivially replaced by stdlib, zero callers |
| `swarms/structs/swarm_id.py` | 6 | `f"swarm-{uuid4().hex}"` wrapper | Zero callers, one-liner not worth a module |
| `swarms/utils/xml_utils.py` | 77 | XML parsing helpers | No direct callers outside utils |
| `swarms/tools/func_to_str.py` | 43 | `function_to_str()`, `functions_to_str()` | Zero imports found anywhere |
| `swarms/tools/tool_type.py` | 7 | `ToolType` union type | Only used inside `base_tool.py` — inline it there |
| `swarms/prompts/tests.py` | 95 | `TEST_WRITER_SOP_PROMPT()` string | Zero imports, not exported |

---

## 2. EXPERIMENTAL / ABANDONED — Review Before Deleting

Unfinished or CLI-only experiments never promoted to the public API.

| File | Lines | Contents | Signal |
|---|---|---|---|
| `swarms/structs/various_alt_swarms.py` | 1103 | Alternative swarm implementations | Only 1 import (`base_swarm.py`), no docs |
| `swarms/structs/collaborative_utils.py` | 77 | Agent communication helpers | Only 2 imports, not exported |
| `swarms/agents/ape_agent.py` | 36 | `auto_generate_prompt()` | Only used in CLI, not public API |
| `swarms/agents/auto_chat_agent.py` | 52 | REPL-style interactive chat loop | Only used in CLI, appears incomplete |
| `swarms/agents/auto_generate_swarm_config.py` | 447 | Parse markdown YAML into swarm config | Only used in CLI, experimental |

---

## 3. UTILS — Low Usage

| File | Lines | Status |
|---|---|---|
| `swarms/utils/get_cpu_cores.py` | 50 | Only 2 import sites; could be inlined |
| `swarms/utils/types.py` | 13 | Used in 3-4 files but not exported; consolidate or inline |

---

## 4. PROMPTS — Unverified Usage

The `swarms/prompts/` directory has 60+ files. Most are domain-specific and fine, but these stand out:

| File | Contents | Why Flag |
|---|---|---|
| `swarms/prompts/tests.py` | Test-writing SOP prompt | Zero imports anywhere |
| `swarms/prompts/tools.py` | Tool-use prompts | Verify import count — likely unused |

---

## 5. LARGE FILES WITH LOW USAGE

| File | Lines | Import Count | Notes |
|---|---|---|---|
| `swarms/structs/various_alt_swarms.py` | 1103 | 1 | Single caller in `base_swarm.py` |
| `swarms/agents/auto_generate_swarm_config.py` | 447 | 1 | CLI only |

---

## 6. NOT EXPORTED BUT SHOULD BE

Heavily used internally but missing from `__init__.py`. Not deletion candidates.

| File | Import Count | Action |
|---|---|---|
| `swarms/utils/loguru_logger.py` | 70+ files | DONE — now exported |
| `swarms/utils/formatter.py` | 70+ files | Add to `utils/__init__.py` |
| `swarms/utils/workspace_utils.py` | 15+ files | Add to `utils/__init__.py` |
| `swarms/structs/planner_worker_swarm.py` | multiple | Add to `structs/__init__.py` |
| `swarms/structs/tree_swarm.py` | 5+ examples | Add to `structs/__init__.py` |
| `swarms/structs/transforms.py` | docs + examples | Add to `structs/__init__.py` |
| `swarms/structs/deep_discussion.py` | 3 callers | Add to `structs/__init__.py` |
| `swarms/structs/image_batch_processor.py` | example + test | Add to `structs/__init__.py` |
| `swarms/structs/csv_to_agent.py` | example | Add to `structs/__init__.py` |
| `swarms/structs/hierarchical_structured_communication_framework.py` | 16+ matches | Add to `structs/__init__.py` |

---

## Retracted (Found to be Active)

| File | Reason |
|---|---|
| `swarms/structs/safe_loading.py` | Actively used — `SafeLoaderUtils` and `SafeStateManager` called in `agent.py` |
| `swarms/tools/function_util.py` | Actively used — `process_tool_docs()` imported and called in `base_tool.py:524` |

---

## Remaining Deletion Queue

### Phase 1 — Safe, delete now

```
swarms/structs/agent_roles.py
swarms/structs/concat.py
swarms/structs/swarm_id.py
swarms/tools/func_to_str.py
swarms/tools/tool_type.py
swarms/prompts/tests.py
```

### Phase 2 — Verify then delete

```
swarms/utils/xml_utils.py              (confirm no callers outside utils)
swarms/utils/get_cpu_cores.py          (inline at 2 call sites)
swarms/utils/types.py                  (consolidate or inline)
swarms/structs/collaborative_utils.py  (confirm only 2 callers, inline logic)
```

### Phase 3 — Investigate before deciding

```
swarms/structs/various_alt_swarms.py        (1103 lines, 1 import)
swarms/agents/ape_agent.py                  (CLI only)
swarms/agents/auto_chat_agent.py            (CLI only)
swarms/agents/auto_generate_swarm_config.py (CLI only)
```

---

## Summary

| Status | Files | Est. Lines |
|---|---|---|
| Already deleted | 8 | ~5,500+ |
| Phase 1 — delete now | 6 | ~215 |
| Phase 2 — verify + delete | 4 | ~175 |
| Phase 3 — investigate | 4 | ~1,600+ |
| Retracted (keep) | 2 | — |
