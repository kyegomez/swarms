# Swarms Examples

This directory contains comprehensive examples demonstrating various capabilities and use cases of the Swarms framework. Each subdirectory focuses on specific aspects of multi-agent systems, single agents, tools, and integrations.

## Directory Overview

### Multi-Agent Systems

| Example | Description |
|---------|-------------|
| [multi_agent/](multi_agent/) | Advanced multi-agent patterns including agent rearrangement, auto swarm builder (ASB), batched workflows, board of directors, caching, concurrent processing, councils, debates, elections, forest swarms, graph workflows, group chats, heavy swarms, hierarchical swarms, LLM council, majority voting, orchestration examples, paper implementations, sequential workflows, social algorithms, simulations, spreadsheet examples, swarm routing, and utilities |
| [README.md](multi_agent/README.md) | Complete multi-agent examples documentation |
| [duo_agent.py](multi_agent/duo_agent.py) | Two-agent collaboration example |
| [tree_swarm_new_updates.py](multi_agent/tree_swarm_new_updates.py) | Tree swarm implementation with latest updates |
| [agent_rearrange_examples/](multi_agent/agent_rearrange_examples/) | Agent rearrangement and reorganization examples |
| [agent_router_examples/](multi_agent/agent_router_examples/) | Agent routing and routing patterns |
| [asb/](multi_agent/asb/) | Auto Swarm Builder (ASB) examples and implementations |
| [batched_grid_workflow/](multi_agent/batched_grid_workflow/) | Batched grid workflow examples |
| [board_of_directors/](multi_agent/board_of_directors/) | Board of directors pattern examples |
| [caching_examples/](multi_agent/caching_examples/) | Agent caching examples |
| [concurrent_examples/](multi_agent/concurrent_examples/) | Concurrent processing and parallel execution examples |
| [council/](multi_agent/council/) | Council pattern examples with judge evaluation |
| [council_of_judges/](multi_agent/council_of_judges/) | Council of judges pattern implementations |
| [debate_examples/](multi_agent/debate_examples/) | Multi-agent debate and discussion examples |
| [election_swarm_examples/](multi_agent/election_swarm_examples/) | Election and voting swarm examples |
| [exec_utilities/](multi_agent/exec_utilities/) | Execution utilities including uvloop examples |
| [forest_swarm_examples/](multi_agent/forest_swarm_examples/) | Forest swarm and tree-based swarm examples |
| [graphworkflow_examples/](multi_agent/graphworkflow_examples/) | Graph workflow and graph-based processing examples |
| [groupchat/](multi_agent/groupchat/) | Group chat and multi-agent conversation examples |
| [heavy_swarm_examples/](multi_agent/heavy_swarm_examples/) | Heavy swarm implementations for complex tasks |
| [hiearchical_swarm/](multi_agent/hiearchical_swarm/) | Hierarchical swarm structures and patterns |
| [hscf/](multi_agent/hscf/) | Hierarchical Swarm Control Framework examples |
| [llm_council_examples/](multi_agent/llm_council_examples/) | LLM Council collaboration patterns |
| [majority_voting/](multi_agent/majority_voting/) | Majority voting and consensus examples |
| [mar/](multi_agent/mar/) | Multi-agent rearrangement examples |
| [moa_examples/](multi_agent/moa_examples/) | Mixture of Agents (MOA) examples |
| [orchestration_examples/](multi_agent/orchestration_examples/) | Agent orchestration and coordination examples |
| [paper_implementations/](multi_agent/paper_implementations/) | Research paper implementations |
| [sequential_workflow/](multi_agent/sequential_workflow/) | Sequential workflow examples |
| [simulations/](multi_agent/simulations/) | Multi-agent simulation examples |
| [social_algorithms_examples/](multi_agent/social_algorithms_examples/) | Social algorithm patterns and implementations |
| [spreadsheet_examples/](multi_agent/spreadsheet_examples/) | Spreadsheet-based agent examples |
| [swarm_router/](multi_agent/swarm_router/) | Swarm routing and routing patterns |
| [swarmarrange/](multi_agent/swarmarrange/) | Swarm arrangement examples |
| [swarms_api_examples/](multi_agent/swarms_api_examples/) | Swarms API integration examples |
| [utils/](multi_agent/utils/) | Multi-agent utility functions |
| [agent_grpo_examples/](multi_agent/agent_grpo_examples/) | Agent GRPO (Group Reinforcement Policy Optimization) examples for math and physics problems |

### Single Agent Systems

| Example | Description |
|---------|-------------|
| [single_agent/](single_agent/) | Single agent implementations including demos, external agent integrations, LLM integrations (Azure, Claude, DeepSeek, Mistral, OpenAI, Qwen), onboarding, RAG, reasoning agents, tools integration, utils, vision capabilities, and MCP integration |
| [README.md](single_agent/README.md) | Complete single agent examples documentation |
| [simple_agent.py](single_agent/simple_agent.py) | Basic single agent example |
| [auto_agent.py](single_agent/auto_agent.py) | Auto agent implementation |
| [agent_mcp.py](single_agent/agent_mcp.py) | MCP integration example |
| [agent_skill_examples/](single_agent/agent_skill_examples/) | Agent skill examples including dynamic skills loader, custom skills, and pre-built skills (code review, data visualization, financial analysis) |
| [demos/](single_agent/demos/) | Single agent demonstration examples including insurance and legal agents |
| [full_autonomy/](single_agent/full_autonomy/) | Autonomous agent examples with marketplace integration and full autonomy capabilities |
| [example_autonomous_looper_run_bash.py](single_agent/full_autonomy/example_autonomous_looper_run_bash.py) | Autonomous agent with `run_bash` tool for terminal access |
| [handoffs/](single_agent/handoffs/) | Agent handoff examples including autonomous agents with handoffs |
| [external_agents/](single_agent/external_agents/) | External agent integrations including OpenAI Assistant wrapper |
| [llms/](single_agent/llms/) | LLM integration examples for Azure, Claude, DeepSeek, Mistral, OpenAI, O3, and Qwen |
| [azure_agent_api_verison.py](single_agent/llms/azure_agent_api_verison.py) | Azure agent API version example |
| [azure_agent.py](single_agent/llms/azure_agent.py) | Azure agent example |
| [azure_model_support.py](single_agent/llms/azure_model_support.py) | Azure model support |
| [base_llm.py](single_agent/llms/base_llm.py) | Base LLM example |
| [mistral_example.py](single_agent/llms/mistral_example.py) | Mistral example |
| [o3_agent.py](single_agent/llms/o3_agent.py) | O3 agent example |
| [qwen_3_base.py](single_agent/llms/qwen_3_base.py) | Qwen 3 base example |
| [claude_examples/](single_agent/llms/claude_examples/) | Claude model examples |
| [claude_4_example.py](single_agent/llms/claude_examples/claude_4_example.py) | Claude 4 example |
| [claude_4.py](single_agent/llms/claude_examples/claude_4.py) | Claude 4 implementation |
| [swarms_claude_example.py](single_agent/llms/claude_examples/swarms_claude_example.py) | Swarms Claude example |
| [deepseek_examples/](single_agent/llms/deepseek_examples/) | DeepSeek model examples |
| [deepseek_r1.py](single_agent/llms/deepseek_examples/deepseek_r1.py) | DeepSeek R1 example |
| [fast_r1_groq.py](single_agent/llms/deepseek_examples/fast_r1_groq.py) | Fast R1 Groq example |
| [groq_deepseek_agent.py](single_agent/llms/deepseek_examples/groq_deepseek_agent.py) | Groq DeepSeek agent |
| [openai_examples/](single_agent/llms/openai_examples/) | OpenAI model examples |
| [4o_mini_demo.py](single_agent/llms/openai_examples/4o_mini_demo.py) | GPT-4o mini demo |
| [reasoning_duo_batched.py](single_agent/llms/openai_examples/reasoning_duo_batched.py) | Reasoning duo batched example |
| [test_async_litellm.py](single_agent/llms/openai_examples/test_async_litellm.py) | Async LiteLLM test |
| [onboard/](single_agent/onboard/) | Agent onboarding examples and configurations |
| [marketplace/](single_agent/marketplace/) | Marketplace prompt integration examples including quant trader agent |
| [rag/](single_agent/rag/) | Retrieval Augmented Generation (RAG) implementations with vector database integrations |
| [full_agent_rag_example.py](single_agent/rag/full_agent_rag_example.py) | Full agent RAG example |
| [pinecone_example.py](single_agent/rag/pinecone_example.py) | Pinecone vector database example |
| [qdrant_agent.py](single_agent/rag/qdrant_agent.py) | Qdrant agent example |
| [qdrant_rag_example.py](single_agent/rag/qdrant_rag_example.py) | Qdrant RAG example |
| [simple_example.py](single_agent/rag/simple_example.py) | Simple RAG example |
| [reasoning_agent_examples/](single_agent/reasoning_agent_examples/) | Reasoning agent patterns including consistency, GPK, iterative, and reasoning duo |
| [agent_judge_evaluation_criteria_example.py](single_agent/reasoning_agent_examples/agent_judge_evaluation_criteria_example.py) | Agent judge evaluation criteria example |
| [agent_judge_example.py](single_agent/reasoning_agent_examples/agent_judge_example.py) | Agent judge example |
| [consistency_agent.py](single_agent/reasoning_agent_examples/consistency_agent.py) | Consistency agent |
| [consistency_example.py](single_agent/reasoning_agent_examples/consistency_example.py) | Consistency example |
| [gpk_agent.py](single_agent/reasoning_agent_examples/gpk_agent.py) | GPK agent |
| [iterative_agent.py](single_agent/reasoning_agent_examples/iterative_agent.py) | Iterative agent |
| [reasoning_agent_router_now.py](single_agent/reasoning_agent_examples/reasoning_agent_router_now.py) | Reasoning agent router (current) |
| [reasoning_agent_router.py](single_agent/reasoning_agent_examples/reasoning_agent_router.py) | Reasoning agent router |
| [reasoning_duo_example.py](single_agent/reasoning_agent_examples/reasoning_duo_example.py) | Reasoning duo example |
| [reasoning_duo_test.py](single_agent/reasoning_agent_examples/reasoning_duo_test.py) | Reasoning duo test |
| [reasoning_duo.py](single_agent/reasoning_agent_examples/reasoning_duo.py) | Reasoning duo implementation |
| [tools/](single_agent/tools/) | Tool integration examples including Exa search, LiteLLM, multi-tool usage, Omni modal, Solana, structured outputs, and browser agents |
| [agent_with_exa.py](single_agent/tools/agent_with_exa.py) | Agent with Exa search integration |
| [exa_search_agent.py](single_agent/tools/exa_search_agent.py) | Exa search agent example |
| [example_async_vs_multithread.py](single_agent/tools/example_async_vs_multithread.py) | Async vs multithread example |
| [litellm_tool_example.py](single_agent/tools/litellm_tool_example.py) | LiteLLM tool example |
| [multi_tool_usage_agent.py](single_agent/tools/multi_tool_usage_agent.py) | Multi-tool usage agent |
| [new_tools_examples.py](single_agent/tools/new_tools_examples.py) | New tools examples |
| [omni_modal_agent.py](single_agent/tools/omni_modal_agent.py) | Omni modal agent |
| [swarms_of_browser_agents.py](single_agent/tools/swarms_of_browser_agents.py) | Swarms of browser agents |
| [swarms_tools_example.py](single_agent/tools/swarms_tools_example.py) | Swarms tools example |
| [together_deepseek_agent.py](single_agent/tools/together_deepseek_agent.py) | Together DeepSeek agent |
| [solana_tool/](single_agent/tools/solana_tool/) | Solana tool integration |
| [solana_tool.py](single_agent/tools/solana_tool/solana_tool.py) | Solana tool implementation |
| [solana_tool_test.py](single_agent/tools/solana_tool/solana_tool_test.py) | Solana tool test |
| [structured_outputs/](single_agent/tools/structured_outputs/) | Structured outputs examples |
| [example_meaning_of_life_agents.py](single_agent/tools/structured_outputs/example_meaning_of_life_agents.py) | Meaning of life agents example |
| [structured_outputs_example.py](single_agent/tools/structured_outputs/structured_outputs_example.py) | Structured outputs example |
| [tools_examples/](single_agent/tools/tools_examples/) | Additional tool examples |
| [dex_screener.py](single_agent/tools/tools_examples/dex_screener.py) | DEX screener tool |
| [financial_news_agent.py](single_agent/tools/tools_examples/financial_news_agent.py) | Financial news agent |
| [simple_tool_example.py](single_agent/tools/tools_examples/simple_tool_example.py) | Simple tool example |
| [swarms_tool_example_simple.py](single_agent/tools/tools_examples/swarms_tool_example_simple.py) | Simple Swarms tool example |
| [utils/](single_agent/utils/) | Single agent utility functions including async agents, custom base URLs, dynamic context windows, fallback tests, handoffs, markdown agents, and XML output |
| [async_agent.py](single_agent/utils/async_agent.py) | Async agent example |
| [custom_agent_base_url.py](single_agent/utils/custom_agent_base_url.py) | Custom agent base URL |
| [dynamic_context_window.py](single_agent/utils/dynamic_context_window.py) | Dynamic context window example |
| [fallback_test.py](single_agent/utils/fallback_test.py) | Fallback test example |
| [grok_4_agent.py](single_agent/utils/grok_4_agent.py) | Grok 4 agent example |
| [handoffs_example.py](single_agent/utils/handoffs_example.py) | Handoffs example |
| [list_agent_output_types.py](single_agent/utils/list_agent_output_types.py) | List agent output types |
| [markdown_agent.py](single_agent/utils/markdown_agent.py) | Markdown agent example |
| [medical_agent_add_to_marketplace.py](single_agent/utils/medical_agent_add_to_marketplace.py) | Medical agent marketplace example |
| [xml_output_example.py](single_agent/utils/xml_output_example.py) | XML output example |
| [autosaving_examples/](single_agent/utils/autosaving_examples/) | Autosaving examples |
| [autosave_basic_example.py](single_agent/utils/autosaving_examples/autosave_basic_example.py) | Basic autosave example |
| [autosave_config_access_example.py](single_agent/utils/autosaving_examples/autosave_config_access_example.py) | Autosave config access example |
| [autosave_directory_structure_example.py](single_agent/utils/autosaving_examples/autosave_directory_structure_example.py) | Autosave directory structure example |
| [autosave_recovery_example.py](single_agent/utils/autosaving_examples/autosave_recovery_example.py) | Autosave recovery example |
| [transform_prompts/](single_agent/utils/transform_prompts/) | Prompt transformation examples |
| [transforms_agent_example.py](single_agent/utils/transform_prompts/transforms_agent_example.py) | Transform agent example |
| [transforms_examples.py](single_agent/utils/transform_prompts/transforms_examples.py) | Transform examples |
| [vision/](single_agent/vision/) | Vision and multimodal agent examples including image processing and batch image examples |
| [anthropic_vision_test.py](single_agent/vision/anthropic_vision_test.py) | Anthropic vision test |
| [image_batch_example.py](single_agent/vision/image_batch_example.py) | Image batch processing example |
| [multimodal_example.py](single_agent/vision/multimodal_example.py) | Multimodal example |
| [multiple_image_processing.py](single_agent/vision/multiple_image_processing.py) | Multiple image processing |
| [vision_test.py](single_agent/vision/vision_test.py) | Vision test |
| [vision_tools.py](single_agent/vision/vision_tools.py) | Vision tools |

### Tools & Integrations

| Example | Description |
|---------|-------------|
| [tools/](tools/) | Tool integration examples including agent-as-tools, base tool implementations, browser automation, Claude integration, Exa search, Firecrawl, multi-tool usage, and Stagehand integration |
| [README.md](tools/README.md) | Complete tools examples documentation |
| [agent_as_tools.py](tools/agent_as_tools.py) | Using agents as tools |
| [browser_use_as_tool.py](tools/browser_use_as_tool.py) | Browser automation tool |
| [browser_use_demo.py](tools/browser_use_demo.py) | Browser automation demonstration |
| [claude_as_a_tool.py](tools/claude_as_a_tool.py) | Claude model integration as a tool |
| [exa_search_agent.py](tools/exa_search_agent.py) | Exa search integration |
| [exa_search_agent_quant.py](tools/exa_search_agent_quant.py) | Exa search with quantitative analysis |
| [exa_search_test.py](tools/exa_search_test.py) | Exa search testing examples |
| [firecrawl_agents_example.py](tools/firecrawl_agents_example.py) | Firecrawl integration |
| [base_tool_examples/](tools/base_tool_examples/) | Base tool implementation examples |
| [base_tool_examples.py](tools/base_tool_examples/base_tool_examples.py) | Base tool examples |
| [conver_funcs_to_schema.py](tools/base_tool_examples/conver_funcs_to_schema.py) | Convert functions to schema |
| [convert_basemodels.py](tools/base_tool_examples/convert_basemodels.py) | Convert base models |
| [exa_search_test.py](tools/base_tool_examples/exa_search_test.py) | Exa search test |
| [example_usage.py](tools/base_tool_examples/example_usage.py) | Base tool usage example |
| [schema_validation_example.py](tools/base_tool_examples/schema_validation_example.py) | Schema validation example |
| [test_anthropic_specific.py](tools/base_tool_examples/test_anthropic_specific.py) | Anthropic-specific test |
| [test_base_tool_comprehensive_fixed.py](tools/base_tool_examples/test_base_tool_comprehensive_fixed.py) | Comprehensive base tool test (fixed) |
| [test_base_tool_comprehensive.py](tools/base_tool_examples/test_base_tool_comprehensive.py) | Comprehensive base tool test |
| [test_function_calls_anthropic.py](tools/base_tool_examples/test_function_calls_anthropic.py) | Function calls test (Anthropic) |
| [test_function_calls.py](tools/base_tool_examples/test_function_calls.py) | Function calls test |
| [multii_tool_use/](tools/multii_tool_use/) | Multi-tool usage examples |
| [stagehand/](tools/stagehand/) | Stagehand UI automation |
| [1_stagehand_wrapper_agent.py](tools/stagehand/1_stagehand_wrapper_agent.py) | Stagehand wrapper agent |
| [2_stagehand_tools_agent.py](tools/stagehand/2_stagehand_tools_agent.py) | Stagehand tools agent |
| [3_stagehand_mcp_agent.py](tools/stagehand/3_stagehand_mcp_agent.py) | Stagehand MCP agent |
| [4_stagehand_multi_agent_workflow.py](tools/stagehand/4_stagehand_multi_agent_workflow.py) | Stagehand multi-agent workflow |
| [tests/](tools/stagehand/tests/) | Stagehand tests |
| [test_stagehand_integration.py](tools/stagehand/tests/test_stagehand_integration.py) | Stagehand integration test |
| [test_stagehand_simple.py](tools/stagehand/tests/test_stagehand_simple.py) | Simple Stagehand test |

### API & Protocols

#### Swarms API

| Example | Description |
|---------|-------------|
| [swarms_api/](swarms_api/) | Swarms API usage examples including agent overview, batch processing, client integration, team examples, analysis, and rate limiting |
| [README.md](swarms_api/README.md) | API examples documentation |
| [agent_overview.py](swarms_api/agent_overview.py) | Agent overview and listing examples |
| [client_example.py](swarms_api/client_example.py) | API client example |
| [batch_example.py](swarms_api/batch_example.py) | Batch processing example |
| [hospital_team.py](swarms_api/hospital_team.py) | Hospital management team simulation |
| [legal_team.py](swarms_api/legal_team.py) | Legal team collaboration example |
| [icd_ten_analysis.py](swarms_api/icd_ten_analysis.py) | ICD-10 medical code analysis |
| [rate_limits.py](swarms_api/rate_limits.py) | Rate limiting and throttling examples |

#### Model Context Protocol (MCP)

| Example | Description |
|---------|-------------|
| [mcp/](mcp/) | Model Context Protocol (MCP) integration examples including agent implementations, multi-connection setups, server configurations, utility functions, and multi-MCP guides |
| [README.md](mcp/README.md) | MCP examples documentation |
| [multi_mcp_example.py](mcp/multi_mcp_example.py) | Multi-MCP connection example |
| [agent_examples/](mcp/agent_examples/) | Agent-based MCP examples |
| [agent_mcp_old.py](mcp/agent_examples/agent_mcp_old.py) | Legacy agent MCP example |
| [agent_multi_mcp_connections.py](mcp/agent_examples/agent_multi_mcp_connections.py) | Agent with multiple MCP connections |
| [agent_tools_dict_example.py](mcp/agent_examples/agent_tools_dict_example.py) | Agent tools dictionary example |
| [mcp_exampler.py](mcp/agent_examples/mcp_exampler.py) | MCP example implementation |
| [servers/](mcp/servers/) | MCP server implementations |
| [mcp_agent_tool.py](mcp/servers/mcp_agent_tool.py) | MCP agent tool server |
| [mcp_test.py](mcp/servers/mcp_test.py) | MCP server testing |
| [okx_crypto_server.py](mcp/servers/okx_crypto_server.py) | OKX crypto MCP server |
| [test.py](mcp/servers/test.py) | MCP server test |
| [mcp_utils/](mcp/mcp_utils/) | MCP utility functions |
| [client.py](mcp/mcp_utils/client.py) | MCP client utility |
| [mcp_client_call.py](mcp/mcp_utils/mcp_client_call.py) | MCP client call example |
| [mcp_multiple_servers_example.py](mcp/mcp_utils/mcp_multiple_servers_example.py) | Multiple MCP servers example |
| [mcp_multiple_tool_test.py](mcp/mcp_utils/mcp_multiple_tool_test.py) | Multiple MCP tools test |
| [multiagent_client.py](mcp/mcp_utils/multiagent_client.py) | Multi-agent MCP client |
| [singleagent_client.py](mcp/mcp_utils/singleagent_client.py) | Single agent MCP client |
| [test_multiple_mcp_servers.py](mcp/mcp_utils/test_multiple_mcp_servers.py) | Test multiple MCP servers |
| [utils/](mcp/mcp_utils/utils/) | MCP utility subdirectory |
| [find_tools_on_mcp.py](mcp/mcp_utils/utils/find_tools_on_mcp.py) | Find tools on MCP |
| [mcp_execute_example.py](mcp/mcp_utils/utils/mcp_execute_example.py) | MCP execute example |
| [mcp_load_tools_example.py](mcp/mcp_utils/utils/mcp_load_tools_example.py) | MCP load tools example |
| [mcp_multiserver_tool_fetch.py](mcp/mcp_utils/utils/mcp_multiserver_tool_fetch.py) | Multi-server tool fetch |
| [utils.py](mcp/mcp_utils/utils.py) | MCP utilities |
| [multi_mcp_guide/](mcp/multi_mcp_guide/) | Multi-MCP setup guides |
| [agent_mcp.py](mcp/multi_mcp_guide/agent_mcp.py) | Agent MCP guide |
| [mcp_agent_tool.py](mcp/multi_mcp_guide/mcp_agent_tool.py) | MCP agent tool guide |
| [okx_crypto_server.py](mcp/multi_mcp_guide/okx_crypto_server.py) | OKX crypto server guide |

#### Agents over Protocol (AOP)

| Example | Description |
|---------|-------------|
| [aop_examples/](aop_examples/) | Agents over Protocol (AOP) examples demonstrating MCP server setup, agent discovery, client interactions, queue-based task submission, medical AOP implementations, and utility functions |
| [README.md](aop_examples/README.md) | AOP examples documentation |
| [server.py](aop_examples/server.py) | AOP server implementation |
| [client/](aop_examples/client/) | AOP client examples and agent discovery including cluster, queue, raw client, and task examples |
| [aop_cluster_example.py](aop_examples/client/aop_cluster_example.py) | AOP cluster example |
| [aop_queue_example.py](aop_examples/client/aop_queue_example.py) | AOP queue-based task submission |
| [aop_raw_client_code.py](aop_examples/client/aop_raw_client_code.py) | Raw AOP client implementation |
| [aop_raw_task_example.py](aop_examples/client/aop_raw_task_example.py) | Raw AOP task example |
| [example_new_agent_tools.py](aop_examples/client/example_new_agent_tools.py) | New agent tools example |
| [get_all_agents.py](aop_examples/client/get_all_agents.py) | Get all available agents |
| [list_agents_and_call_them.py](aop_examples/client/list_agents_and_call_them.py) | List and call agents example |
| [discovery/](aop_examples/discovery/) | Agent discovery examples including communication and discovery testing |
| [example_agent_communication.py](aop_examples/discovery/example_agent_communication.py) | Agent communication example |
| [test_aop_discovery.py](aop_examples/discovery/test_aop_discovery.py) | AOP discovery testing |
| [simple_discovery_example.py](aop_examples/discovery/simple_discovery_example.py) | Simple discovery example |
| [example_aop_discovery.py](aop_examples/discovery/example_aop_discovery.py) | AOP discovery example |
| [medical_aop/](aop_examples/medical_aop/) | Medical AOP implementations |
| [server.py](aop_examples/medical_aop/server.py) | Medical AOP server |
| [client.py](aop_examples/medical_aop/client.py) | Medical AOP client |
| [utils/](aop_examples/utils/) | AOP utility functions |
| [network_management_example.py](aop_examples/utils/network_management_example.py) | Network management example |
| [comprehensive_aop_example.py](aop_examples/utils/comprehensive_aop_example.py) | Comprehensive AOP example |
| [persistence_management_example.py](aop_examples/utils/persistence_management_example.py) | Persistence management example |
| [network_error_example.py](aop_examples/utils/network_error_example.py) | Network error handling example |
| [persistence_example.py](aop_examples/utils/persistence_example.py) | Persistence example |

### Advanced Capabilities

| Example | Description |
|---------|-------------|
| [reasoning_agents/](reasoning_agents/) | Advanced reasoning capabilities including agent judge evaluation systems, O3 model integration, mixture of agents (MOA) sequential examples, and reasoning agent router examples |
| [README.md](reasoning_agents/README.md) | Reasoning agents documentation |
| [moa_seq_example.py](reasoning_agents/moa_seq_example.py) | MOA sequential example |
| [agent_judge_examples/](reasoning_agents/agent_judge_examples/) | Agent judge evaluation systems |
| [example1_basic_evaluation.py](reasoning_agents/agent_judge_examples/example1_basic_evaluation.py) | Basic evaluation example |
| [example2_technical_evaluation.py](reasoning_agents/agent_judge_examples/example2_technical_evaluation.py) | Technical evaluation example |
| [example3_creative_evaluation.py](reasoning_agents/agent_judge_examples/example3_creative_evaluation.py) | Creative evaluation example |
| [reasoning_agent_router_examples/](reasoning_agents/reasoning_agent_router_examples/) | Reasoning agent router examples |
| [agent_judge_example.py](reasoning_agents/reasoning_agent_router_examples/agent_judge_example.py) | Agent judge example |
| [gkp_agent_example.py](reasoning_agents/reasoning_agent_router_examples/gkp_agent_example.py) | GKP agent example |
| [ire_example.py](reasoning_agents/reasoning_agent_router_examples/ire_example.py) | IRE example |
| [reasoning_duo_example.py](reasoning_agents/reasoning_agent_router_examples/reasoning_duo_example.py) | Reasoning duo example |
| [reflexion_agent_example.py](reasoning_agents/reasoning_agent_router_examples/reflexion_agent_example.py) | Reflexion agent example |
| [self_consistency_example.py](reasoning_agents/reasoning_agent_router_examples/self_consistency_example.py) | Self-consistency example |
| [voice_agents/](voice_agents/) | Voice and speech-enabled agent examples including agent speech, agent with speech, debate with speech, Google Calendar integration, hierarchical speech swarm, and autonomous agent with speech |
| [README.md](voice_agents/README.md) | Voice agents documentation |
| [agent_speech.py](voice_agents/agent_speech.py) | Agent with speech capabilities |
| [agent_with_speech.py](voice_agents/agent_with_speech.py) | Speech-enabled agent implementation |
| [debate_with_speech.py](voice_agents/debate_with_speech.py) | Multi-agent debate with speech capabilities |
| [google_calendar_agent.py](voice_agents/google_calendar_agent.py) | Google Calendar integration with voice agent |
| [hiearchical_speech_swarm.py](voice_agents/hiearchical_speech_swarm.py) | Hierarchical speech swarm implementation |
| [run_auto_agent_with_speech.py](voice_agents/run_auto_agent_with_speech.py) | Autonomous agent with terminal access and streaming TTS |

### Marketplace

| Example | Description |
|---------|-------------|
| [marketplace/](marketplace/) | Swarms marketplace prompt integration examples for using pre-built prompts from the marketplace |
| [zia_agent.py](marketplace/zia_agent.py) | Zia agent implementation using marketplace prompts |
| [single_agent/marketplace/](single_agent/marketplace/) | Single agent marketplace examples including marketplace prompt integration and quant trader agent |
| [marketplace_prompt_example.py](single_agent/marketplace/marketplace_prompt_example.py) | Example of using marketplace prompts with agents |
| [quant_trader_agent.py](single_agent/marketplace/quant_trader_agent.py) | Quantitative trader agent using marketplace prompts |

### Guides & Tutorials

| Example | Description |
|---------|-------------|
| [guides/](guides/) | Comprehensive guides and tutorials including demos, generation length blog, geo guesser agent, graph workflow guide, hackathon examples, hierarchical marketing team, nano banana Jarvis agent, smart database, web scraper agents, workshops, x402 examples, and workshop examples (840_update, 850_workshop) |
| [README.md](guides/README.md) | Guides documentation |
| [hiearchical_marketing_team.py](guides/hiearchical_marketing_team.py) | Hierarchical marketing team example |
| [840_update/](guides/840_update/) | Update examples from version 8.4.0 including agent rearrange, auto swarm builder, and fallback examples |
| [850_workshop/](guides/850_workshop/) | Workshop examples from version 8.5.0 including AOP, MOA, peer review, and concurrent examples |
| [880_update_changelog_examples/](guides/880_update_changelog_examples/) | Changelog examples showcasing new features including marketplace integration, multi-agent structures, workflow orchestration, voice agents, evaluation & debate, routing, and autosaving |
| [changelog_890/](guides/changelog_890/) | Changelog examples from January 2026 release including dynamic skills loader, autonomous agent loop, agent handoffs, API key validation, max loops parameter, multi-tool agent tutorial, hierarchical voice agent, and agent rearrange patterns |
| [mem0/](guides/mem0/) | Mem0 integration examples for memory management with Swarms |
| [demos/](guides/demos/) | Various demonstration examples including apps, crypto, finance, insurance, legal, medical, real estate, science, and synthetic data |
| [fairy_swarm/](guides/fairy_swarm/) | Fairy swarm examples and implementations |
| [generation_length_blog/](guides/generation_length_blog/) | Long-form content generation examples |
| [geo_guesser_agent/](guides/geo_guesser_agent/) | Geographic guessing agent examples |
| [graphworkflow_guide/](guides/graphworkflow_guide/) | Comprehensive graph workflow guide and examples |
| [hackathon_judge_agent/](guides/hackathon_judge_agent/) | Hackathon judging agent examples |
| [hackathons/](guides/hackathons/) | Hackathon project examples |
| [nano_banana_jarvis_agent/](guides/nano_banana_jarvis_agent/) | Nano banana Jarvis agent with image generation |
| [smart_database/](guides/smart_database/) | Smart database examples |
| [voice_agents/](guides/voice_agents/) | Voice agent examples with speech capabilities |
| [web_scraper_agents/](guides/web_scraper_agents/) | Web scraper agent examples |
| [workshops/](guides/workshops/) | Workshop examples |
| [x402_examples/](guides/x402_examples/) | X402 protocol examples |
| [deployment/](guides/deployment/) | Deployment strategies: cron jobs and FastAPI examples |
| [v9_examples/](guides/v9_examples/) | V9 autonomous looper examples (autonomous_looper, tools, sub-agents) |

### Deployment

| Example | Description |
|---------|-------------|
| [guides/deployment/](guides/deployment/) | Deployment strategies and patterns including cron job implementations and FastAPI deployment examples |
| [README.md](guides/deployment/README.md) | Deployment documentation |
| [fastapi/](guides/deployment/fastapi/) | FastAPI deployment examples |
| [cron_job_examples/](guides/deployment/cron_job_examples/) | Cron job examples |

### Utilities

| Example | Description |
|---------|-------------|
| [utils/](utils/) | Utility functions and helper implementations including agent loader, communication examples, concurrent wrappers, miscellaneous utilities, and telemetry |
| [README.md](utils/README.md) | Utils documentation |
| [agent_loader/](utils/agent_loader/) | Agent loading utilities |
| [communication_examples/](utils/communication_examples/) | Agent communication patterns including DuckDB, Pulsar, Redis, and SQLite |
| [fetch_prompt.py](utils/fetch_prompt.py) | Prompt fetching utilities |
| [litellm_connect_issue.py](utils/litellm_connect_issue.py) | LiteLLM connection issue examples |
| [litellm_network_error_handling.py](utils/litellm_network_error_handling.py) | LiteLLM network error handling |
| [misc/](utils/misc/) | Miscellaneous utility functions including AOP, conversation, CSV agents, and visualization |
| [agent_map_test.py](utils/misc/agent_map_test.py) | Agent map test |
| [conversation_simple.py](utils/misc/conversation_simple.py) | Simple conversation example |
| [conversation_test_truncate.py](utils/misc/conversation_test_truncate.py) | Conversation truncate test |
| [conversation_test.py](utils/misc/conversation_test.py) | Conversation test |
| [csvagent_example.py](utils/misc/csvagent_example.py) | CSV agent example |
| [dict_to_table.py](utils/misc/dict_to_table.py) | Dictionary to table conversion |
| [swarm_matcher_example.py](utils/misc/swarm_matcher_example.py) | Swarm matcher example |
| [test_load_conversation.py](utils/misc/test_load_conversation.py) | Load conversation test |
| [visualizer_test.py](utils/misc/visualizer_test.py) | Visualizer test |
| [aop/](utils/misc/aop/) | AOP utility examples |
| [client.py](utils/misc/aop/client.py) | AOP client utility |
| [test_aop.py](utils/misc/aop/test_aop.py) | AOP test |
| [telemetry/](utils/telemetry/) | Telemetry and monitoring utilities |

### User Interface

| Example | Description |
|---------|-------------|
| [ui/](ui/) | User interface examples and implementations including chat interfaces |
| [README.md](ui/README.md) | UI examples documentation |
| [chat.py](ui/chat.py) | Chat interface example |

### Command Line Interface

| Example | Description |
|---------|-------------|
| [cli/](cli/) | CLI command examples demonstrating all available Swarms CLI features including setup, agent management, multi-agent architectures, and utilities |
| [README.md](cli/README.md) | CLI examples documentation |
| [01_setup_check.sh](cli/01_setup_check.sh) | Environment setup verification |
| [02_onboarding.sh](cli/02_onboarding.sh) | User onboarding process |
| [03_get_api_key.sh](cli/03_get_api_key.sh) | API key retrieval |
| [04_check_login.sh](cli/04_check_login.sh) | Login status verification |
| [05_create_agent.sh](cli/05_create_agent.sh) | Create custom agents |
| [06_run_agents_yaml.sh](cli/06_run_agents_yaml.sh) | Run agents from YAML configuration |
| [07_load_markdown.sh](cli/07_load_markdown.sh) | Load markdown configurations |
| [08_llm_council.sh](cli/08_llm_council.sh) | LLM Council collaboration |
| [09_heavy_swarm.sh](cli/09_heavy_swarm.sh) | HeavySwarm complex analysis |
| [10_autoswarm.sh](cli/10_autoswarm.sh) | Auto swarm builder examples |
| [11_features.sh](cli/11_features.sh) | Feature demonstration |
| [12_help.sh](cli/12_help.sh) | Help and documentation |
| [13_auto_upgrade.sh](cli/13_auto_upgrade.sh) | Automatic upgrade process |
| [14_book_call.sh](cli/14_book_call.sh) | Book a call functionality |
| [run_all_examples.sh](cli/run_all_examples.sh) | Run all CLI examples |

## Quick Start

| Use Case | Example |
|----------|---------|
| New to Swarms? | Start with [single_agent/simple_agent.py](single_agent/simple_agent.py) for basic concepts |
| Want to use the CLI? | Check out [cli/](cli/) for all CLI command examples |
| Want multi-agent workflows? | Check out [multi_agent/duo_agent.py](multi_agent/duo_agent.py) |
| Need tool integration? | Explore [tools/agent_as_tools.py](tools/agent_as_tools.py) |
| Interested in AOP? | Try [aop_examples/client/example_new_agent_tools.py](aop_examples/client/example_new_agent_tools.py) for agent discovery |
| Want to see social algorithms? | Check out [multi_agent/social_algorithms_examples/](multi_agent/social_algorithms_examples/) |
| Looking for guides? | Visit [guides/](guides/) for comprehensive tutorials |
| Need RAG? | Try [single_agent/rag/](single_agent/rag/) for RAG examples |
| Want reasoning agents? | Check out [reasoning_agents/](reasoning_agents/) for reasoning agent examples |
| Interested in marketplace prompts? | Explore [marketplace/](marketplace/) for marketplace prompt integration examples |
| Want voice/speech capabilities? | Check out [voice_agents/](voice_agents/) for speech-enabled agent examples |
| Interested in agent skills? | Explore [single_agent/agent_skill_examples/](single_agent/agent_skill_examples/) for dynamic skills |
| Want autonomous agents? | Check out [single_agent/full_autonomy/](single_agent/full_autonomy/) for full autonomy examples |
| Need agent handoffs? | See [single_agent/handoffs/](single_agent/handoffs/) for handoff patterns |
| Looking for latest features? | Visit [guides/changelog_890/](guides/changelog_890/) for January 2026 release examples |

## Key Examples by Category

### Multi-Agent Patterns

| Example | Description |
|---------|-------------|
| [Duo Agent](multi_agent/duo_agent.py) | Two-agent collaboration |
| [Tree Swarm](multi_agent/tree_swarm_new_updates.py) | Tree swarm with latest updates |
| [Agent Rearrange](multi_agent/agent_rearrange_examples/) | Agent rearrangement patterns |
| [Agent Router](multi_agent/agent_router_examples/) | Agent routing implementations |
| [Agent GRPO](multi_agent/agent_grpo_examples/) | Agent GRPO examples for math and physics problems |
| [Auto Swarm Builder](multi_agent/asb/) | Auto Swarm Builder (ASB) examples |
| [Batched Grid Workflow](multi_agent/batched_grid_workflow/) | Batched grid workflow patterns |
| [Board of Directors](multi_agent/board_of_directors/) | Board of directors pattern |
| [Concurrent Processing](multi_agent/concurrent_examples/) | Concurrent and parallel execution |
| [Council](multi_agent/council/) | Council pattern with judge evaluation |
| [Council of Judges](multi_agent/council_of_judges/) | Council of judges implementations |
| [Debates](multi_agent/debate_examples/) | Multi-agent debate examples |
| [Election Swarm](multi_agent/election_swarm_examples/) | Election and voting swarms |
| [Forest Swarm](multi_agent/forest_swarm_examples/) | Forest and tree-based swarms |
| [Hierarchical Swarm](multi_agent/hiearchical_swarm/hierarchical_swarm_example.py) | Hierarchical agent structures |
| [Heavy Swarm](multi_agent/heavy_swarm_examples/) | Heavy swarm for complex tasks |
| [Group Chat](multi_agent/groupchat/) | Multi-agent conversations and interactive group chat implementations |
| [Graph Workflow](multi_agent/graphworkflow_examples/graph_workflow_example.py) | Graph-based workflows |
| [LLM Council](multi_agent/llm_council_examples/) | LLM Council collaboration |
| [Majority Voting](multi_agent/majority_voting/) | Majority voting and consensus |
| [Mixture of Agents](multi_agent/moa_examples/) | Mixture of Agents (MOA) examples |
| [Orchestration](multi_agent/orchestration_examples/) | Agent orchestration patterns |
| [Sequential Workflow](multi_agent/sequential_workflow/) | Sequential workflow examples |
| [Simulations](multi_agent/simulations/) | Multi-agent simulations |
| [Social Algorithms](multi_agent/social_algorithms_examples/) | Various social algorithm patterns |
| [Spreadsheet Examples](multi_agent/spreadsheet_examples/) | Spreadsheet-based agents |
| [Swarm Router](multi_agent/swarm_router/) | Swarm routing patterns |

### Single Agent Examples

| Example | Description |
|---------|-------------|
| [Simple Agent](single_agent/simple_agent.py) | Basic agent setup |
| [Auto Agent](single_agent/auto_agent.py) | Auto agent implementation |
| [Agent Skills](single_agent/agent_skill_examples/) | Dynamic skills loader and custom skill examples |
| [Autonomous Agents](single_agent/full_autonomy/) | Full autonomy examples with marketplace integration |
| [Agent Handoffs](single_agent/handoffs/) | Agent handoff patterns and examples |
| [Marketplace Integration](single_agent/marketplace/) | Marketplace prompt integration examples |
| [Reasoning Agents](single_agent/reasoning_agent_examples/) | Advanced reasoning patterns including consistency, GPK, iterative, and reasoning duo |
| [Vision Agents](single_agent/vision/multimodal_example.py) | Vision and multimodal capabilities |
| [RAG Agents](single_agent/rag/) | Retrieval augmented generation |
| [External Agents](single_agent/external_agents/) | External agent integrations |
| [Onboarding](single_agent/onboard/) | Agent onboarding examples |
| [Tools Integration](single_agent/tools/) | Comprehensive tool integration examples |
| [Voice Agents](voice_agents/) | Speech-enabled agent examples |

### Tool Integrations

| Example | Description |
|---------|-------------|
| [Agent as Tools](tools/agent_as_tools.py) | Using agents as tools |
| [Browser Automation](tools/browser_use_as_tool.py) | Browser control |
| [Browser Demo](tools/browser_use_demo.py) | Browser automation demonstration |
| [Claude as Tool](tools/claude_as_a_tool.py) | Claude model as a tool |
| [Exa Search](tools/exa_search_agent.py) | Search integration |
| [Exa Search Quant](tools/exa_search_agent_quant.py) | Exa search with quantitative analysis |
| [Firecrawl](tools/firecrawl_agents_example.py) | Firecrawl web scraping integration |
| [Stagehand](tools/stagehand/) | UI automation |
| [Base Tools](tools/base_tool_examples/) | Base tool implementation examples |
| [Multi-Tool Use](tools/multii_tool_use/) | Multi-tool usage examples |

### Model Integration Examples

| Example | Description |
|---------|-------------|
| [OpenAI](single_agent/llms/openai_examples/4o_mini_demo.py) | OpenAI models |
| [Claude](single_agent/llms/claude_examples/claude_4_example.py) | Claude models |
| [DeepSeek](single_agent/llms/deepseek_examples/deepseek_r1.py) | DeepSeek models |
| [DeepSeek Groq](single_agent/llms/deepseek_examples/groq_deepseek_agent.py) | DeepSeek with Groq integration |
| [Azure](single_agent/llms/azure_agent.py) | Azure OpenAI |
| [Mistral](single_agent/llms/mistral_example.py) | Mistral models |
| [O3](single_agent/llms/o3_agent.py) | O3 model integration |
| [Qwen](single_agent/llms/qwen_3_base.py) | Qwen model integration |
| [Ollama](https://docs.swarms.world) | Local Ollama and other providers via LiteLLM—see docs and [single_agent/llms/](single_agent/llms/) |

### Marketplace Examples

| Example | Description |
|---------|-------------|
| [Marketplace Prompt](single_agent/marketplace/marketplace_prompt_example.py) | Using marketplace prompts with agents |
| [Quant Trader Agent](single_agent/marketplace/quant_trader_agent.py) | Quantitative trader agent with marketplace prompts |
| [Zia Agent](marketplace/zia_agent.py) | Zia agent with marketplace prompt integration |

### CLI Examples

| Example | Description |
|---------|-------------|
| [Setup Check](cli/01_setup_check.sh) | Verify environment setup |
| [Onboarding](cli/02_onboarding.sh) | User onboarding process |
| [Get API Key](cli/03_get_api_key.sh) | API key retrieval |
| [Check Login](cli/04_check_login.sh) | Login status verification |
| [Create Agent](cli/05_create_agent.sh) | Create custom agents via CLI |
| [Run Agents YAML](cli/06_run_agents_yaml.sh) | Run agents from YAML configuration |
| [Load Markdown](cli/07_load_markdown.sh) | Load markdown configurations |
| [LLM Council](cli/08_llm_council.sh) | Run LLM Council collaboration |
| [HeavySwarm](cli/09_heavy_swarm.sh) | Run HeavySwarm for complex tasks |
| [AutoSwarm](cli/10_autoswarm.sh) | Auto swarm builder examples |
| [Features](cli/11_features.sh) | Feature demonstration |
| [Help](cli/12_help.sh) | Help and documentation |
| [Auto Upgrade](cli/13_auto_upgrade.sh) | Automatic upgrade process |
| [Book Call](cli/14_book_call.sh) | Book a call functionality |
| [Research Agent](cli/research_agent_example.sh) | Research agent example |
| [Run All Examples](cli/run_all_examples.sh) | Run all CLI examples |

## Documentation

Each subdirectory contains its own README.md file with detailed descriptions and links to all available examples. Click on any folder above to explore its specific examples and use cases.

## Related Resources

- [Main Swarms Documentation](../docs/)
- [API Reference](../swarms/)
- [Contributing Guidelines](../CONTRIBUTING.md)

## Contributing

Found an interesting example or want to add your own? Check out our [contributing guidelines](../CONTRIBUTING.md) and feel free to submit pull requests with new examples or improvements to existing ones.

---

*This examples directory is continuously updated with new patterns, integrations, and use cases. Check back regularly for the latest examples!*
