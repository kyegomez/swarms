# Swarms Examples

This directory contains comprehensive examples demonstrating various capabilities and use cases of the Swarms framework. Each subdirectory focuses on specific aspects of multi-agent systems, single agents, tools, and integrations.

## Directory Overview

### Multi-Agent Systems

- **[multi_agent/](multi_agent/)** - Advanced multi-agent patterns including agent rearrangement, auto swarm builder (ASB), batched workflows, board of directors, caching, concurrent processing, councils, debates, elections, forest swarms, graph workflows, group chats, heavy swarms, hierarchical swarms, majority voting, orchestration examples, social algorithms, simulations, spreadsheet examples, and swarm routing.
  - [README.md](multi_agent/README.md) - Complete multi-agent examples documentation

### Single Agent Systems

- **[single_agent/](single_agent/)** - Single agent implementations including demos, external agent integrations, LLM integrations (Azure, Claude, DeepSeek, Mistral, OpenAI, Qwen), onboarding, RAG, reasoning agents, tools integration, utils, and vision capabilities.
  - [README.md](single_agent/README.md) - Complete single agent examples documentation
  - [simple_agent.py](single_agent/simple_agent.py) - Basic single agent example

### Tools & Integrations

- **[tools/](tools/)** - Tool integration examples including agent-as-tools, base tool implementations, browser automation, Claude integration, Exa search, Firecrawl, multi-tool usage, and Stagehand integration.
  - [README.md](tools/README.md) - Complete tools examples documentation
  - [agent_as_tools.py](tools/agent_as_tools.py) - Using agents as tools

### Model Integrations

- **[models/](models/)** - Various model integrations including Cerebras, GPT-5, GPT-OSS, Llama 4, Lumo, and Ollama implementations with concurrent processing examples and provider-specific configurations.
  - [README.md](models/README.md) - Model integration documentation
  - [simple_example_ollama.py](models/simple_example_ollama.py) - Ollama integration example
  - [cerebas_example.py](models/cerebas_example.py) - Cerebras model example
  - [lumo_example.py](models/lumo_example.py) - Lumo model example

### API & Protocols

- **[swarms_api_examples/](swarms_api_examples/)** - Swarms API usage examples including agent overview, batch processing, client integration, team examples, analysis, and rate limiting.
  - [README.md](swarms_api_examples/README.md) - API examples documentation
  - [client_example.py](swarms_api_examples/client_example.py) - API client example
  - [batch_example.py](swarms_api_examples/batch_example.py) - Batch processing example

- **[mcp/](mcp/)** - Model Context Protocol (MCP) integration examples including agent implementations, multi-connection setups, server configurations, and utility functions.
  - [README.md](mcp/README.md) - MCP examples documentation
  - [multi_mcp_example.py](mcp/multi_mcp_example.py) - Multi-MCP connection example

- **[aop_examples/](aop_examples/)** - Agents over Protocol (AOP) examples demonstrating MCP server setup, agent discovery, client interactions, queue-based task submission, and medical AOP implementations.
  - [README.md](aop_examples/README.md) - AOP examples documentation
  - [server.py](aop_examples/server.py) - AOP server implementation

### Advanced Capabilities

- **[reasoning_agents/](reasoning_agents/)** - Advanced reasoning capabilities including agent judge evaluation systems, O3 model integration, and mixture of agents (MOA) sequential examples.
  - [README.md](reasoning_agents/README.md) - Reasoning agents documentation
  - [example_o3.py](reasoning_agents/example_o3.py) - O3 model example
  - [moa_seq_example.py](reasoning_agents/moa_seq_example.py) - MOA sequential example

- **[rag/](rag/)** - Retrieval Augmented Generation (RAG) implementations with vector database integrations including Qdrant examples.
  - [README.md](rag/README.md) - RAG documentation
  - [qdrant_rag_example.py](rag/qdrant_rag_example.py) - Qdrant RAG example

### Guides & Tutorials

- **[guides/](guides/)** - Comprehensive guides and tutorials including generation length blog, geo guesser agent, graph workflow guide, hierarchical marketing team, nano banana Jarvis agent, smart database, web scraper agents, and workshop examples (840_update, 850_workshop).
  - [README.md](guides/README.md) - Guides documentation
  - [hiearchical_marketing_team.py](guides/hiearchical_marketing_team.py) - Hierarchical marketing team example

### Deployment

- **[deployment/](deployment/)** - Deployment strategies and patterns including cron job implementations and FastAPI deployment examples.
  - [README.md](deployment/README.md) - Deployment documentation
  - [fastapi/](deployment/fastapi/) - FastAPI deployment examples
  - [cron_job_examples/](deployment/cron_job_examples/) - Cron job examples

### Utilities

- **[utils/](utils/)** - Utility functions and helper implementations including agent loader, communication examples, concurrent wrappers, miscellaneous utilities, and telemetry.
  - [README.md](utils/README.md) - Utils documentation

### User Interface

- **[ui/](ui/)** - User interface examples and implementations including chat interfaces.
  - [README.md](ui/README.md) - UI examples documentation
  - [chat.py](ui/chat.py) - Chat interface example

## Quick Start

1. **New to Swarms?** Start with [single_agent/simple_agent.py](single_agent/simple_agent.py) for basic concepts
2. **Want multi-agent workflows?** Check out [multi_agent/duo_agent.py](multi_agent/duo_agent.py)
3. **Need tool integration?** Explore [tools/agent_as_tools.py](tools/agent_as_tools.py)
4. **Interested in AOP?** Try [aop_examples/client/example_new_agent_tools.py](aop_examples/client/example_new_agent_tools.py) for agent discovery
5. **Want to see social algorithms?** Check out [multi_agent/social_algorithms_examples/](multi_agent/social_algorithms_examples/)
6. **Looking for guides?** Visit [guides/](guides/) for comprehensive tutorials
7. **Need RAG?** Try [rag/qdrant_rag_example.py](rag/qdrant_rag_example.py)
8. **Want reasoning agents?** Check out [reasoning_agents/example_o3.py](reasoning_agents/example_o3.py)

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
- [RAG Agents](single_agent/rag/qdrant_rag_example.py) - Retrieval augmented generation

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
