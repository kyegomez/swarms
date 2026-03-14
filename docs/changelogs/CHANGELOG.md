# Changelog

**February 2, 2026 → March 13, 2026** Swarms v10

---

## Introduction

Over the past month and a half, the Swarms team has been heads down building toward the premier release of Swarms v10. What started as a series of targeted improvements quickly grew into one of the most comprehensive updates the framework has seen. Across dozens of commits and multiple pull requests, we rebuilt core systems, introduced powerful new orchestration primitives, and made the entire framework faster, smarter, and easier to work with.

The sub-agent system received a full overhaul. Agents can now autonomously spawn sub-agents, assign them tasks, monitor their progress in real time, and cancel them on demand — all backed by a persistent async task registry that handles retries and failure recovery without any manual wiring on your end.

The `HierarchicalSwarm` took a major step forward with two standout additions. Worker agents can now execute in parallel, cutting down task completion time significantly on complex workflows. A built-in judge agent can also evaluate every worker's output after each cycle, scoring performance across five dimensions and surfacing actionable feedback — so your swarm gets smarter with every run.

We also introduced `SkillOrchestra`, a brand new routing primitive that lets you define agent skills explicitly and route tasks to the most qualified agent automatically based on what each agent knows how to do. Instead of hardcoding which agent handles what, `SkillOrchestra` makes skill-based delegation a first-class feature of the framework.

On top of these headline features, the CLI was modernized from the ground up, a range of long-standing bugs were resolved, and internal defaults and cleanup work has made the entire codebase more stable and predictable. This changelog covers everything shipped during this period, organized by category.

---

## New Features

- **Sub-Agent Status & Cancel Tools** — Added `check_sub_agent_status` and `cancel_sub_agent_tasks` as new autonomous planning tools with full schema definitions and registry-backed handler implementations, wired into Agent's tool dispatch map. *(2026-03-12)*
- **Async Sub-Agent Registry Refactor** — Replaced raw asyncio dispatching in `assign_task_tool` with `SubagentRegistry`-based spawning for consistent task tracking, retries, and cancellation; added `_find_registry` and `_resolve_sub_agents_by_name` helpers. *(2026-03-12)*
- **Sub-Agent Print Visibility** — Enabled `print_on=True` for sub-agents created via `create_sub_agent_tool` to surface their output during execution. *(2026-03-12)*
- **SkillOrchestra** — Added skill-aware agent routing via skill transfer, including a full usage example. *(2026-03-10)*
- **Async Sub-Agent Execution** — Added background task registry with async sub-agent support. *(2026-03-08)*
- **Parallel Agent Execution** — Concurrent agent execution via `ThreadPoolExecutor` with `as_completed()`; outputs written to conversation in original submission order to avoid race conditions. *(2026-03-05)*
- **Agent-as-Judge** — Structured per-agent scoring after each cycle using a new `JudgeReport` Pydantic schema with `AgentScore` entries covering task adherence, accuracy, depth, clarity, and swarm contribution weighted into a 0–10 composite; takes priority over `director_feedback_on` when both are enabled. *(2026-03-05)*
- **`arun()` Async Entry Point** — Added async entry point wrapping `run()` in `asyncio.to_thread()` for non-blocking use in FastAPI and other async frameworks. *(2026-03-05)*
- **`HIERARCHICAL_SWARM_JUDGE_PROMPT`** — Extensive judge system prompt with five weighted scoring dimensions, rubrics for every score band, mandatory citation standards, and adversarial honesty guardrails. *(2026-03-05)*
- **CLI UI Overhaul** — Replaced ASCII art banner with a compact Claude Code-style header featuring active provider detection, rotating startup tips, and a simplified red-and-white color scheme. *(2026-03-03)*
- **Plain-Text Help Output** — Replaced rich-formatted help with plain-text argparse-style reference listing all commands, options, and examples. *(2026-03-03)*
- **Sub-Agent Task Creation** — Enabled agents to create and assign tasks to sub-agents via a new autonomous tool. *(2026-02-02)*
- **Base64 Image Processing** — New guide and support for agents processing images through base64 encoding. *(2026-02-02)*

---

## Improvements

- **PDF Utility Removal** — Removed `pdf_to_text` as a core utility from `utils/__init__.py` and `data_to_text.py`; inlined a local implementation directly in the mem0 example script. *(2026-03-12)*
- **Agent Class Cleanup** — Removed duplicated async sub-agent methods from the Agent class body, now superseded by registry integration. *(2026-03-12)*
- **`top_p` Default Fix** — Removed hardcoded `top_p=None` from `auto_chat_agent` and set `Agent.top_p` default to `None` to avoid unintended sampling parameter injection. *(2026-03-12)*
- **Model & Version Update** — Bumped swarms to `9.0.3` and updated default model references from `gpt-4.1`/`claude-sonnet-4-5` to `gpt-5.4`/`gpt-5.1` across example files and CLI. *(2026-03-12)*
- **Async Sub-Agents Code Cleanup** — Code cleanup and logic refactoring for async sub-agent execution. *(2026-03-09)*
- **Hierarchical Swarm Docs Overhaul** — Overhauled `hierarchical_swarm.md` with updated Mermaid diagram, new constructor parameter table, dedicated sections for all new features, `arun()` docs with FastAPI integration example, and updated best practices. *(2026-03-05)*
- **CLI Cleanup** — Removed deprecated `features` command, `show_features` function, redundant `help` subcommand, and dead CLI table utilities (`create_command_table`, `create_detailed_command_table`). *(2026-03-03)*
- **Agent Defaults** — Agent now defaults to `gpt-4.1`, uses styled formatter console input prompts in interactive mode, and always returns the final summary from the autonomous loop. *(2026-03-03)*
- **CLI Documentation** — Updated CLI documentation to reflect recent CLI overhaul. *(2026-03-03)*
- **CLI Parameters** — Improved CLI with fewer parameters and fixed the `swarms chat` feature. *(2026-02-02)*
- **Sub-Agent Tutorial Docs** — Improved sub-agent tutorial documentation. *(2026-02-02)*
- **Examples Section** — Improved examples section with updated references. *(2026-02-05)*
- **Bash Tool for `agent.auto`** — Added bash tool support for agents running in auto mode. *(2026-02-02)*
- **Auto-Save Examples** — Added examples for auto-saving structures. *(2026-02-02)*
- **Bash Command Security Guardrails** — Added security validation in `autonomous_loop_utils` to block dangerous shell patterns including recursive deletion, pipe-to-shell, disk writes, fork bombs, and privilege escalation; enforces a 512-character command length limit. *(2026-03-03)*
- **Dynamic Timestamps in Agent Prompts** — Introduced a `get_time` helper to dynamically inject the current date and time into prompts at runtime. *(2026-03-03)*
- **Auto Chat Loop Refactor** — Refactored `auto_chat_agent` into a persistent while-loop with graceful exit-command handling and an exposed `model_name` parameter. *(2026-03-03)*
- **Agent Selected Tools Method** — Added method for running agents with selected tools by name. *(2026-02-05)*
---

## Bug Fixes

- Fixed README and auto-chat issue on CLI. *(2026-03-11)*
- **Welcome Workflow** — Fixed unexpected inputs in welcome workflow. *(2026-02-28)*
- **LLMCouncil** — Fixed incompatibility between `LLMCouncil`, `SwarmRouter`, and the API server. *(2026-02-24)*
- **Agent Identity** — Fixed system prompt injection for agent identity when no system prompt is provided. *(2026-02-17)*
- **DebateWithJudge** — Fixed API field for `DebateWithJudge` to work correctly with the API server. *(2026-02-15)*
- **RoundRobin** — Fixed `RoundRobin` swarm; removed useless parameter. *(2026-02-15)*
- **LiteLLM Empty System Prompt Fix** — Fixed Anthropic API rejections by stripping whitespace-only system prompt blocks and normalizing orchestrator-style `System:/Human:` prompts before message construction. *(2026-03-12)*
- Fixed typo: `seperatedly` → `separately`. *(2026-02-10)*

---

## Dependency Updates

- Bumped `aquasecurity/trivy-action` from `0.34.1` to `0.35.0`. *(2026-03-09)*
- Updated `ruff` requirement to `>=0.5.1,<0.15.6`. *(2026-03-09)*
- Updated `types-pytz` requirement to `>=2023.3,<2027.0`. *(2026-03-09)*
- Updated `ruff` requirement to `>=0.5.1,<0.15.5`. *(2026-03-03)*
- Bumped `actions/upload-artifact` from `6` to `7`. *(2026-03-03)*
- Updated `ruff` requirement to `>=0.5.1,<0.15.3`. *(2026-02-23)*
- Bumped `aquasecurity/trivy-action` from `0.34.0` to `0.34.1`. *(2026-02-23)*
- Updated `pymdown-extensions` requirement to `~=10.21`. *(2026-02-16)*
- Updated `ruff` requirement to `>=0.5.1,<0.15.2`. *(2026-02-16)*
- Bumped `aquasecurity/trivy-action` from `0.33.1` to `0.34.0`. *(2026-02-16)*
- Updated `ruff` requirement to `>=0.5.1,<0.15.1`. *(2026-02-12)*

---

## Conclusion

This release represents one of the most significant leaps forward in the Swarms framework to date. Over the past six weeks, nearly every major system — sub-agent orchestration, the hierarchical swarm, the CLI, and core agent infrastructure — has been meaningfully improved. Multi-agent workflows are faster with parallel execution, smarter with the judge agent, and more autonomous than ever with a fully async sub-agent registry that handles task tracking, retries, and cancellation out of the box. The CLI is cleaner, the defaults are more sensible, and a wide range of bugs that affected real-world usage have been resolved. Whether you are building simple pipelines or complex multi-layered swarms, the framework is more capable, more reliable, and easier to work with than it has ever been. We are grateful for every contributor, bug reporter, and community member who helped shape this release — and this is only the beginning of what is planned for v10.
