# Single Agent Examples

This directory contains examples demonstrating single agent patterns, configurations, and use cases in Swarms.

## Agent Skills
- [agent_with_skills.py](agent_skill_examples/agent_with_skills.py) - Agent with skills integration
- [agent_with_multiple_skills.py](agent_skill_examples/agent_with_multiple_skills.py) - Agent with multiple skills
- [agent_with_dynamic_skills.py](agent_skill_examples/agent_with_dynamic_skills.py) - Agent with dynamic skills loader
- [agent_with_custom_skill.py](agent_skill_examples/agent_with_custom_skill.py) - Agent with custom skill
- [code-review/](agent_skill_examples/code-review/) - Code review skill example
- [data-visualization/](agent_skill_examples/data-visualization/) - Data visualization skill example
- [financial-analysis/](agent_skill_examples/financial-analysis/) - Financial analysis skill example

## Demos
- [insurance_agent.py](demos/insurance_agent.py) - Insurance processing agent
- [persistent_legal_agent.py](demos/persistent_legal_agent.py) - Legal document processing agent

## External Agents
- [openai_assistant_wrapper.py](external_agents/openai_assistant_wrapper.py) - OpenAI Assistant integration

## LLM Integrations

### Azure
- [azure_agent_api_verison.py](llms/azure_agent_api_verison.py) - Azure API version handling
- [azure_agent.py](llms/azure_agent.py) - Azure OpenAI integration
- [azure_model_support.py](llms/azure_model_support.py) - Azure model support

### Claude
- [claude_4_example.py](llms/claude_examples/claude_4_example.py) - Claude 4 integration
- [claude_4.py](llms/claude_examples/claude_4.py) - Claude 4 implementation
- [swarms_claude_example.py](llms/claude_examples/swarms_claude_example.py) - Swarms Claude integration

### DeepSeek
- [deepseek_r1.py](llms/deepseek_examples/deepseek_r1.py) - DeepSeek R1 model
- [fast_r1_groq.py](llms/deepseek_examples/fast_r1_groq.py) - Fast R1 with Groq
- [groq_deepseek_agent.py](llms/deepseek_examples/groq_deepseek_agent.py) - Groq DeepSeek integration

### Mistral
- [mistral_example.py](llms/mistral_example.py) - Mistral model integration

### OpenAI
- [4o_mini_demo.py](llms/openai_examples/4o_mini_demo.py) - GPT-4o Mini demonstration
- [reasoning_duo_batched.py](llms/openai_examples/reasoning_duo_batched.py) - Batched reasoning with OpenAI
- [test_async_litellm.py](llms/openai_examples/test_async_litellm.py) - Async LiteLLM testing

### O3
- [o3_agent.py](llms/o3_agent.py) - O3 model integration

### Qwen
- [qwen_3_base.py](llms/qwen_3_base.py) - Qwen 3 base model

## Full Autonomy
- [autonomous_agent.py](full_autonomy/autonomous_agent.py) - Autonomous agent implementation
- [marketplace_and_auto_agent.py](full_autonomy/marketplace_and_auto_agent.py) - Marketplace integration with autonomous agent

## Handoffs
- [handoff_example.py](handoffs/handoff_example.py) - Basic agent handoff example
- [autonomous_agent_with_handoffs_example.py](handoffs/autonomous_agent_with_handoffs_example.py) - Autonomous agent with handoffs

## Marketplace
- [marketplace_prompt_example.py](marketplace/marketplace_prompt_example.py) - Marketplace prompt integration example
- [quant_trader_agent.py](marketplace/quant_trader_agent.py) - Quantitative trader agent using marketplace prompts

## Onboarding
- [agents.yaml](onboard/agents.yaml) - Agent configuration file
- [onboard-basic.py](onboard/onboard-basic.py) - Basic onboarding example

## RAG (Retrieval Augmented Generation)
- [full_agent_rag_example.py](rag/full_agent_rag_example.py) - Complete RAG implementation
- [pinecone_example.py](rag/pinecone_example.py) - Pinecone vector database integration
- [qdrant_agent.py](rag/qdrant_agent.py) - Qdrant vector database agent
- [qdrant_rag_example.py](rag/qdrant_rag_example.py) - Qdrant RAG implementation
- [simple_example.py](rag/simple_example.py) - Simple RAG example
- [README.md](rag/README.md) - RAG documentation

## Reasoning Agents
- [agent_judge_evaluation_criteria_example.py](reasoning_agent_examples/agent_judge_evaluation_criteria_example.py) - Evaluation criteria for agent judging
- [agent_judge_example.py](reasoning_agent_examples/agent_judge_example.py) - Agent judging system
- [consistency_agent.py](reasoning_agent_examples/consistency_agent.py) - Consistency checking agent
- [consistency_example.py](reasoning_agent_examples/consistency_example.py) - Consistency example
- [gpk_agent.py](reasoning_agent_examples/gpk_agent.py) - GPK reasoning agent
- [iterative_agent.py](reasoning_agent_examples/iterative_agent.py) - Iterative reasoning agent
- [reasoning_agent_router_now.py](reasoning_agent_examples/reasoning_agent_router_now.py) - Current reasoning router
- [reasoning_agent_router.py](reasoning_agent_examples/reasoning_agent_router.py) - Reasoning agent router
- [reasoning_duo_example.py](reasoning_agent_examples/reasoning_duo_example.py) - Two-agent reasoning
- [reasoning_duo_test.py](reasoning_agent_examples/reasoning_duo_test.py) - Reasoning duo testing
- [reasoning_duo.py](reasoning_agent_examples/reasoning_duo.py) - Reasoning duo implementation

## Tools Integration
- [agent_with_exa.py](tools/agent_with_exa.py) - Agent with Exa search integration
- [exa_search_agent.py](tools/exa_search_agent.py) - Exa search integration
- [example_async_vs_multithread.py](tools/example_async_vs_multithread.py) - Async vs multithreading comparison
- [litellm_tool_example.py](tools/litellm_tool_example.py) - LiteLLM tool integration
- [multi_tool_usage_agent.py](tools/multi_tool_usage_agent.py) - Multi-tool agent
- [new_tools_examples.py](tools/new_tools_examples.py) - Latest tool examples
- [omni_modal_agent.py](tools/omni_modal_agent.py) - Omni-modal agent
- [swarms_of_browser_agents.py](tools/swarms_of_browser_agents.py) - Browser automation swarms
- [swarms_tools_example.py](tools/swarms_tools_example.py) - Swarms tools integration
- [together_deepseek_agent.py](tools/together_deepseek_agent.py) - Together AI DeepSeek integration

### Solana Tools
- [solana_tool.py](tools/solana_tool/solana_tool.py) - Solana blockchain integration
- [solana_tool_test.py](tools/solana_tool/solana_tool_test.py) - Solana tool testing

### Structured Outputs
- [example_meaning_of_life_agents.py](tools/structured_outputs/example_meaning_of_life_agents.py) - Meaning of life example
- [structured_outputs_example.py](tools/structured_outputs/structured_outputs_example.py) - Structured output examples

### Tools Examples
- [dex_screener.py](tools/tools_examples/dex_screener.py) - DEX screener tool
- [financial_news_agent.py](tools/tools_examples/financial_news_agent.py) - Financial news agent
- [simple_tool_example.py](tools/tools_examples/simple_tool_example.py) - Simple tool usage
- [swarms_tool_example_simple.py](tools/tools_examples/swarms_tool_example_simple.py) - Simple Swarms tool

## Utils
- [async_agent.py](utils/async_agent.py) - Async agent implementation
- [dynamic_context_window.py](utils/dynamic_context_window.py) - Dynamic context window management
- [fallback_test.py](utils/fallback_test.py) - Fallback mechanism testing
- [grok_4_agent.py](utils/grok_4_agent.py) - Grok 4 agent implementation
- [handoffs_example.py](utils/handoffs_example.py) - Agent handoff examples
- [list_agent_output_types.py](utils/list_agent_output_types.py) - Output type listing
- [markdown_agent.py](utils/markdown_agent.py) - Markdown processing agent
- [medical_agent_add_to_marketplace.py](utils/medical_agent_add_to_marketplace.py) - Add medical agent to marketplace
- [xml_output_example.py](utils/xml_output_example.py) - XML output example

### Autosaving
- [autosave_basic_example.py](utils/autosaving_examples/autosave_basic_example.py) - Basic autosave
- [autosave_config_access_example.py](utils/autosaving_examples/autosave_config_access_example.py) - Config access
- [autosave_directory_structure_example.py](utils/autosaving_examples/autosave_directory_structure_example.py) - Directory structure
- [autosave_recovery_example.py](utils/autosaving_examples/autosave_recovery_example.py) - Recovery example

### Transform Prompts
- [transforms_agent_example.py](utils/transform_prompts/transforms_agent_example.py) - Prompt transformation agent
- [transforms_examples.py](utils/transform_prompts/transforms_examples.py) - Prompt transformation examples

## Vision
- [anthropic_vision_test.py](vision/anthropic_vision_test.py) - Anthropic vision testing
- [image_batch_example.py](vision/image_batch_example.py) - Batch image processing
- [multimodal_example.py](vision/multimodal_example.py) - Multimodal agent example
- [multiple_image_processing.py](vision/multiple_image_processing.py) - Multiple image processing
- [vision_test.py](vision/vision_test.py) - Vision testing
- [vision_tools.py](vision/vision_tools.py) - Vision tools integration

## Main Files
- [simple_agent.py](simple_agent.py) - Basic single agent example
