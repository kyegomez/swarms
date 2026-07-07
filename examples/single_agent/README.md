# Single Agent Examples

Runnable examples for building and configuring a single Swarms `Agent`. Examples are grouped into a small number of top-level buckets so the directory is easy to scan.

```
single_agent/
├── getting_started/   First runs: minimal agent, interactive mode, onboarding
├── capabilities/      What an agent can do: tools, skills, RAG, vision, streaming, prompt caching
├── reasoning/         Reasoning agents: judges, consistency, iterative, reasoning duo/router
├── autonomy/          Autonomous loops and agent-to-agent handoffs
├── integrations/      Marketplace prompts and external agent bridges
├── demos/             End-to-end applied demos
└── utils/             Configuration, output formats, and misc helpers
```

Set the relevant API key (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) before running any example.

---

## getting_started/
First-run examples — start here.

- [simple_agent.py](getting_started/simple_agent.py) — Minimal single agent
- [interactive.py](getting_started/interactive.py) — Interactive REPL-style agent
- **onboarding/**
  - [onboard-basic.py](getting_started/onboarding/onboard-basic.py) — Basic onboarding flow
  - [agents.yaml](getting_started/onboarding/agents.yaml) — Agent configuration file

## capabilities/
Feature-specific examples — what an agent can do.

### tools/
- [agent_mcp.py](capabilities/tools/agent_mcp.py) — MCP server tools
- [agent_with_exa.py](capabilities/tools/agent_with_exa.py) / [exa_search_agent.py](capabilities/tools/exa_search_agent.py) — Exa search
- [multi_tool_usage_agent.py](capabilities/tools/multi_tool_usage_agent.py) — Multiple tools
- [litellm_tool_example.py](capabilities/tools/litellm_tool_example.py) — LiteLLM tool calling
- [omni_modal_agent.py](capabilities/tools/omni_modal_agent.py) — Omni-modal agent
- [new_tools_examples.py](capabilities/tools/new_tools_examples.py) — Latest tool patterns
- [example_async_vs_multithread.py](capabilities/tools/example_async_vs_multithread.py) — Async vs multithread
- [swarms_tools_example.py](capabilities/tools/swarms_tools_example.py) — swarms-tools integration
- [swarms_of_browser_agents.py](capabilities/tools/swarms_of_browser_agents.py) — Browser automation
- [together_deepseek_agent.py](capabilities/tools/together_deepseek_agent.py) — Together AI DeepSeek
- **solana_tool/** — [solana_tool.py](capabilities/tools/solana_tool/solana_tool.py), [solana_tool_test.py](capabilities/tools/solana_tool/solana_tool_test.py)
- **structured_outputs/** — [structured_outputs_example.py](capabilities/tools/structured_outputs/structured_outputs_example.py), [example_meaning_of_life_agents.py](capabilities/tools/structured_outputs/example_meaning_of_life_agents.py)
- **tools_examples/** — [simple_tool_example.py](capabilities/tools/tools_examples/simple_tool_example.py), [dex_screener.py](capabilities/tools/tools_examples/dex_screener.py), [financial_news_agent.py](capabilities/tools/tools_examples/financial_news_agent.py), [swarms_tool_example_simple.py](capabilities/tools/tools_examples/swarms_tool_example_simple.py)

### skills/
- [agent_with_skills.py](capabilities/skills/agent_with_skills.py) — Agent with skills
- [agent_with_multiple_skills.py](capabilities/skills/agent_with_multiple_skills.py) — Multiple skills
- [agent_with_dynamic_skills.py](capabilities/skills/agent_with_dynamic_skills.py) — Dynamic skill loading
- [agent_with_custom_skill.py](capabilities/skills/agent_with_custom_skill.py) — Custom skill
- `code-review/`, `data-visualization/`, `financial-analysis/` — example `SKILL.md` definitions

### rag/
- [simple_example.py](capabilities/rag/simple_example.py) — Minimal RAG
- [full_agent_rag_example.py](capabilities/rag/full_agent_rag_example.py) — Complete RAG
- [pinecone_example.py](capabilities/rag/pinecone_example.py) — Pinecone
- [qdrant_agent.py](capabilities/rag/qdrant_agent.py) / [qdrant_rag_example.py](capabilities/rag/qdrant_rag_example.py) — Qdrant

### vision/
- [multimodal_example.py](capabilities/vision/multimodal_example.py) — Multimodal agent
- [anthropic_vision_test.py](capabilities/vision/anthropic_vision_test.py) — Anthropic vision
- [image_batch_example.py](capabilities/vision/image_batch_example.py) / [multiple_image_processing.py](capabilities/vision/multiple_image_processing.py) — Batch images
- [vision_tools.py](capabilities/vision/vision_tools.py) / [vision_test.py](capabilities/vision/vision_test.py) — Vision tools
- **base_64_images/** — [test_base64_image.py](capabilities/vision/base_64_images/test_base64_image.py)

### streaming/
- [streaming_example.py](capabilities/streaming/streaming_example.py) — Token streaming
- [agent_streaming.py](capabilities/streaming/agent_streaming.py) — Streaming agent
- [example_streaming_tools.py](capabilities/streaming/example_streaming_tools.py) — Streaming with tools
- [test_agent_streaming_and_loop.py](capabilities/streaming/test_agent_streaming_and_loop.py) — Streaming + loops

### prompt_caching/
- [prompt_caching_example.py](capabilities/prompt_caching/prompt_caching_example.py) — Overview
- [1_basic_anthropic.py](capabilities/prompt_caching/1_basic_anthropic.py) — Basic Anthropic caching
- [2_one_hour_ttl.py](capabilities/prompt_caching/2_one_hour_ttl.py) — 1-hour TTL
- [3_cache_tools.py](capabilities/prompt_caching/3_cache_tools.py) — Cache tool definitions
- [4_system_only.py](capabilities/prompt_caching/4_system_only.py) — System-prompt only
- [5_openai_caching.py](capabilities/prompt_caching/5_openai_caching.py) — OpenAI automatic caching
- [6_all_options.py](capabilities/prompt_caching/6_all_options.py) / [cache_config_example.py](capabilities/prompt_caching/cache_config_example.py) — Full `cache_config`

## reasoning/
Reasoning-agent patterns.

- [reasoning_duo.py](reasoning/reasoning_duo.py) / [reasoning_duo_example.py](reasoning/reasoning_duo_example.py) / [reasoning_duo_test.py](reasoning/reasoning_duo_test.py) — Two-agent reasoning
- [reasoning_agent_router.py](reasoning/reasoning_agent_router.py) / [reasoning_agent_router_now.py](reasoning/reasoning_agent_router_now.py) — Reasoning router
- [agent_judge_example.py](reasoning/agent_judge_example.py) / [agent_judge_evaluation_criteria_example.py](reasoning/agent_judge_evaluation_criteria_example.py) — Agent-as-judge
- [consistency_agent.py](reasoning/consistency_agent.py) / [consistency_example.py](reasoning/consistency_example.py) — Self-consistency
- [iterative_agent.py](reasoning/iterative_agent.py) — Iterative reasoning
- [gpk_agent.py](reasoning/gpk_agent.py) — Generated-knowledge prompting

## autonomy/
Autonomous loops and handoffs.

### autonomous_agents/
- [auto_agent.py](autonomy/autonomous_agents/auto_agent.py) — `max_loops="auto"` agent
- [autonomous_agent.py](autonomy/autonomous_agents/autonomous_agent.py) — Autonomous agent
- [example_autonomous_looper_run_bash.py](autonomy/autonomous_agents/example_autonomous_looper_run_bash.py) — Autonomous loop with bash
- [sub_agent_test.py](autonomy/autonomous_agents/sub_agent_test.py) — Spawning sub-agents
- [marketplace_and_auto_agent.py](autonomy/autonomous_agents/marketplace_and_auto_agent.py) — Autonomous + marketplace

### handoffs/
- [handoff_example.py](autonomy/handoffs/handoff_example.py) / [handoffs_example.py](autonomy/handoffs/handoffs_example.py) — Basic handoffs
- [duo_agent.py](autonomy/handoffs/duo_agent.py) — Two-agent handoff
- [autonomous_agent_with_handoffs_example.py](autonomy/handoffs/autonomous_agent_with_handoffs_example.py) — Autonomous + handoffs

## integrations/
Marketplace and external agent bridges.

### marketplace/
- [marketplace_prompt_example.py](integrations/marketplace/marketplace_prompt_example.py) — Marketplace prompts
- [quant_trader_agent.py](integrations/marketplace/quant_trader_agent.py) / [zia_agent.py](integrations/marketplace/zia_agent.py) — Marketplace-backed agents
- [medical_agent_add_to_marketplace.py](integrations/marketplace/medical_agent_add_to_marketplace.py) — Publish an agent to the marketplace

### external_agents/
- See [external_agents/README.md](integrations/external_agents/README.md) — bridging non-Swarms agents

## demos/
End-to-end applied demos.

- [insurance_agent.py](demos/insurance_agent.py) — Insurance processing
- [persistent_legal_agent.py](demos/persistent_legal_agent.py) — Legal agent with persistent memory

## utils/
Configuration, output formats, and misc helpers.

- [async_agent.py](utils/async_agent.py) — Async agent
- [custom_agent_base_url.py](utils/custom_agent_base_url.py) — Custom base URL
- [dynamic_context_window.py](utils/dynamic_context_window.py) — Dynamic context window
- [fallback_test.py](utils/fallback_test.py) — Model fallback
- [grok_4_agent.py](utils/grok_4_agent.py) — Grok 4 agent
- [list_agent_output_types.py](utils/list_agent_output_types.py) — Output types
- [markdown_agent.py](utils/markdown_agent.py) — Markdown output
- [xml_output_example.py](utils/xml_output_example.py) — XML output
- **autosaving_examples/** — [autosave_basic_example.py](utils/autosaving_examples/autosave_basic_example.py), [autosave_config_access_example.py](utils/autosaving_examples/autosave_config_access_example.py), [autosave_directory_structure_example.py](utils/autosaving_examples/autosave_directory_structure_example.py), [autosave_recovery_example.py](utils/autosaving_examples/autosave_recovery_example.py)
- **transform_prompts/** — [transforms_agent_example.py](utils/transform_prompts/transforms_agent_example.py), [transforms_examples.py](utils/transform_prompts/transforms_examples.py)
