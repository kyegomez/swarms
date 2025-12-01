# Swarms Examples

This directory contains comprehensive examples demonstrating various capabilities and use cases of the Swarms framework. Each subdirectory focuses on specific aspects of multi-agent systems, single agents, tools, and integrations.

## Directory Overview

### Multi-Agent Systems

- **[multi_agent/](multi_agent/)** - Advanced multi-agent patterns including agent rearrangement, auto swarm builder (ASB), batched workflows, board of directors, caching, concurrent processing, councils, debates, elections, forest swarms, graph workflows, group chats, heavy swarms, hierarchical swarms, LLM council, majority voting, orchestration examples, paper implementations, sequential workflows, social algorithms, simulations, spreadsheet examples, swarm routing, and utilities.
  - [README.md](multi_agent/README.md) - Complete multi-agent examples documentation
  - [duo_agent.py](multi_agent/duo_agent.py) - Two-agent collaboration example
  - [llm_council_examples/](multi_agent/llm_council_examples/) - LLM Council collaboration patterns
  - [caching_examples/](multi_agent/caching_examples/) - Agent caching examples

### Single Agent Systems

- **[single_agent/](single_agent/)** - Single agent implementations including demos, external agent integrations, LLM integrations (Azure, Claude, DeepSeek, Mistral, OpenAI, Qwen), onboarding, RAG, reasoning agents, tools integration, utils, vision capabilities, and MCP integration.
  - [README.md](single_agent/README.md) - Complete single agent examples documentation
  - [simple_agent.py](single_agent/simple_agent.py) - Basic single agent example
  - [agent_mcp.py](single_agent/agent_mcp.py) - MCP integration example
  - [rag/](single_agent/rag/) - Retrieval Augmented Generation (RAG) implementations with vector database integrations

### Tools & Integrations

- **[tools/](tools/)** - Tool integration examples including agent-as-tools, base tool implementations, browser automation, Claude integration, Exa search, Firecrawl, multi-tool usage, and Stagehand integration.
  - [README.md](tools/README.md) - Complete tools examples documentation
  - [agent_as_tools.py](tools/agent_as_tools.py) - Using agents as tools
  - [browser_use_as_tool.py](tools/browser_use_as_tool.py) - Browser automation tool
  - [exa_search_agent.py](tools/exa_search_agent.py) - Exa search integration
  - [firecrawl_agents_example.py](tools/firecrawl_agents_example.py) - Firecrawl integration
  - [base_tool_examples/](tools/base_tool_examples/) - Base tool implementation examples
  - [multii_tool_use/](tools/multii_tool_use/) - Multi-tool usage examples
  - [stagehand/](tools/stagehand/) - Stagehand UI automation

### Model Integrations

- **[models/](models/)** - Various model integrations including Cerebras, GPT-5, GPT-OSS, Llama 4, Lumo, O3, Ollama, and vLLM implementations with concurrent processing examples and provider-specific configurations.
  - [README.md](models/README.md) - Model integration documentation
  - [simple_example_ollama.py](models/simple_example_ollama.py) - Ollama integration example
  - [cerebas_example.py](models/cerebas_example.py) - Cerebras model example
  - [lumo_example.py](models/lumo_example.py) - Lumo model example
  - [example_o3.py](models/example_o3.py) - O3 model example
  - [gpt_5/](models/gpt_5/) - GPT-5 model examples
  - [gpt_oss_examples/](models/gpt_oss_examples/) - GPT-OSS examples
  - [llama4_examples/](models/llama4_examples/) - Llama 4 examples
  - [main_providers/](models/main_providers/) - Main provider configurations
  - [vllm/](models/vllm/) - vLLM integration examples

### API & Protocols

- **[swarms_api/](swarms_api/)** - Swarms API usage examples including agent overview, batch processing, client integration, team examples, analysis, and rate limiting.
  - [README.md](swarms_api/README.md) - API examples documentation
  - [client_example.py](swarms_api/client_example.py) - API client example
  - [batch_example.py](swarms_api/batch_example.py) - Batch processing example
  - [hospital_team.py](swarms_api/hospital_team.py) - Hospital management team simulation
  - [legal_team.py](swarms_api/legal_team.py) - Legal team collaboration example
  - [icd_ten_analysis.py](swarms_api/icd_ten_analysis.py) - ICD-10 medical code analysis
  - [rate_limits.py](swarms_api/rate_limits.py) - Rate limiting and throttling examples

- **[mcp/](mcp/)** - Model Context Protocol (MCP) integration examples including agent implementations, multi-connection setups, server configurations, utility functions, and multi-MCP guides.
  - [README.md](mcp/README.md) - MCP examples documentation
  - [multi_mcp_example.py](mcp/multi_mcp_example.py) - Multi-MCP connection example
  - [agent_examples/](mcp/agent_examples/) - Agent-based MCP examples
  - [servers/](mcp/servers/) - MCP server implementations
  - [mcp_utils/](mcp/mcp_utils/) - MCP utility functions
  - [multi_mcp_guide/](mcp/multi_mcp_guide/) - Multi-MCP setup guides

- **[aop_examples/](aop_examples/)** - Agents over Protocol (AOP) examples demonstrating MCP server setup, agent discovery, client interactions, queue-based task submission, medical AOP implementations, and utility functions.
  - [README.md](aop_examples/README.md) - AOP examples documentation
  - [server.py](aop_examples/server.py) - AOP server implementation
  - [client/](aop_examples/client/) - AOP client examples and agent discovery
  - [discovery/](aop_examples/discovery/) - Agent discovery examples
  - [medical_aop/](aop_examples/medical_aop/) - Medical AOP implementations
  - [utils/](aop_examples/utils/) - AOP utility functions

### Advanced Capabilities

- **[reasoning_agents/](reasoning_agents/)** - Advanced reasoning capabilities including agent judge evaluation systems, O3 model integration, mixture of agents (MOA) sequential examples, and reasoning agent router examples.
  - [README.md](reasoning_agents/README.md) - Reasoning agents documentation
  - [moa_seq_example.py](reasoning_agents/moa_seq_example.py) - MOA sequential example
  - [agent_judge_examples/](reasoning_agents/agent_judge_examples/) - Agent judge evaluation systems
  - [reasoning_agent_router_examples/](reasoning_agents/reasoning_agent_router_examples/) - Reasoning agent router examples

### Guides & Tutorials

- **[guides/](guides/)** - Comprehensive guides and tutorials including demos, generation length blog, geo guesser agent, graph workflow guide, hackathon examples, hierarchical marketing team, nano banana Jarvis agent, smart database, web scraper agents, workshops, x402 examples, and workshop examples (840_update, 850_workshop).
  - [README.md](guides/README.md) - Guides documentation
  - [hiearchical_marketing_team.py](guides/hiearchical_marketing_team.py) - Hierarchical marketing team example
  - [demos/](guides/demos/) - Various demonstration examples
  - [hackathons/](guides/hackathons/) - Hackathon project examples
  - [workshops/](guides/workshops/) - Workshop examples
  - [x402_examples/](guides/x402_examples/) - X402 protocol examples

### Deployment

- **[deployment/](deployment/)** - Deployment strategies and patterns including cron job implementations and FastAPI deployment examples.
  - [README.md](deployment/README.md) - Deployment documentation
  - [fastapi/](deployment/fastapi/) - FastAPI deployment examples
  - [cron_job_examples/](deployment/cron_job_examples/) - Cron job examples

### Utilities

- **[utils/](utils/)** - Utility functions and helper implementations including agent loader, communication examples, concurrent wrappers, miscellaneous utilities, and telemetry.
  - [README.md](utils/README.md) - Utils documentation
  - [agent_loader/](utils/agent_loader/) - Agent loading utilities
  - [communication_examples/](utils/communication_examples/) - Agent communication patterns
  - [concurrent_wrapper_examples.py](utils/concurrent_wrapper_examples.py) - Concurrent processing wrappers
  - [misc/](utils/misc/) - Miscellaneous utility functions
  - [telemetry/](utils/telemetry/) - Telemetry and monitoring utilities

### User Interface

- **[ui/](ui/)** - User interface examples and implementations including chat interfaces.
  - [README.md](ui/README.md) - UI examples documentation
  - [chat.py](ui/chat.py) - Chat interface example

### Command Line Interface

- **[cli/](cli/)** - CLI command examples demonstrating all available Swarms CLI features including setup, agent management, multi-agent architectures, and utilities.
  - [README.md](cli/README.md) - CLI examples documentation
  - [01_setup_check.sh](cli/01_setup_check.sh) - Environment setup verification
  - [05_create_agent.sh](cli/05_create_agent.sh) - Create custom agents
  - [08_llm_council.sh](cli/08_llm_council.sh) - LLM Council collaboration
  - [09_heavy_swarm.sh](cli/09_heavy_swarm.sh) - HeavySwarm complex analysis

## Quick Start

1. **New to Swarms?** Start with [single_agent/simple_agent.py](single_agent/simple_agent.py) for basic concepts
2. **Want to use the CLI?** Check out [cli/](cli/) for all CLI command examples
3. **Want multi-agent workflows?** Check out [multi_agent/duo_agent.py](multi_agent/duo_agent.py)
4. **Need tool integration?** Explore [tools/agent_as_tools.py](tools/agent_as_tools.py)
5. **Interested in AOP?** Try [aop_examples/client/example_new_agent_tools.py](aop_examples/client/example_new_agent_tools.py) for agent discovery
6. **Want to see social algorithms?** Check out [multi_agent/social_algorithms_examples/](multi_agent/social_algorithms_examples/)
7. **Looking for guides?** Visit [guides/](guides/) for comprehensive tutorials
8. **Need RAG?** Try [single_agent/rag/](single_agent/rag/) for RAG examples
9. **Want reasoning agents?** Check out [reasoning_agents/](reasoning_agents/) for reasoning agent examples

## Key Examples by Category

### Multi-Agent Patterns

- [Duo Agent](multi_agent/duo_agent.py) - Two-agent collaboration
- [Hierarchical Swarm](multi_agent/hiearchical_swarm/hierarchical_swarm_example.py) - Hierarchical agent structures
- [Group Chat](multi_agent/groupchat/interactive_groupchat_example.py) - Multi-agent conversations
- [Graph Workflow](multi_agent/graphworkflow_examples/graph_workflow_example.py) - Graph-based workflows
- [Social Algorithms](multi_agent/social_algorithms_examples/) - Various social algorithm patterns

### Single Agent Examples

- [Simple Agent](single_agent/simple_agent.py) - Basic agent setup
- [Reasoning Agents](single_agent/reasoning_agent_examples/) - Advanced reasoning patterns
- [Vision Agents](single_agent/vision/multimodal_example.py) - Vision and multimodal capabilities
- [RAG Agents](single_agent/rag/) - Retrieval augmented generation

### Tool Integrations

- [Agent as Tools](tools/agent_as_tools.py) - Using agents as tools
- [Browser Automation](tools/browser_use_as_tool.py) - Browser control
- [Exa Search](tools/exa_search_agent.py) - Search integration
- [Stagehand](tools/stagehand/) - UI automation

### Model Integrations

- [OpenAI](single_agent/llms/openai_examples/4o_mini_demo.py) - OpenAI models
- [Claude](single_agent/llms/claude_examples/claude_4_example.py) - Claude models
- [DeepSeek](single_agent/llms/deepseek_examples/deepseek_r1.py) - DeepSeek models
- [Azure](single_agent/llms/azure_agent.py) - Azure OpenAI
- [Ollama](models/simple_example_ollama.py) - Local Ollama models

### CLI Examples

- [Setup Check](cli/01_setup_check.sh) - Verify environment setup
- [Create Agent](cli/05_create_agent.sh) - Create custom agents via CLI
- [LLM Council](cli/08_llm_council.sh) - Run LLM Council collaboration
- [HeavySwarm](cli/09_heavy_swarm.sh) - Run HeavySwarm for complex tasks
- [All CLI Examples](cli/) - Complete CLI examples directory

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
