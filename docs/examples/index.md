# Swarms Examples Index

Welcome to the comprehensive Swarms Examples Index! This curated collection showcases the power and versatility of the Swarms framework for building intelligent multi-agent systems. Whether you're a beginner looking to get started or an advanced developer seeking complex implementations, you'll find practical examples to accelerate your AI development journey.

## What is Swarms?

Swarms is a cutting-edge framework for creating sophisticated multi-agent AI systems that can collaborate, reason, and solve complex problems together. From single intelligent agents to coordinated swarms of specialized AI workers, Swarms provides the tools and patterns you need to build the next generation of AI applications.

## What You'll Find Here

This index organizes **100+ production-ready examples** from our [Swarms Examples Repository](https://github.com/The-Swarm-Corporation/swarms-examples) and the main Swarms repository, covering:


- **Single Agent Systems**: From basic implementations to advanced reasoning agents

- **Multi-Agent Architectures**: Collaborative swarms, hierarchical systems, and experimental topologies

- **Industry Applications**: Real-world use cases across finance, healthcare, security, and more

- **Integration Examples**: Connect with popular AI models, tools, and frameworks

- **Advanced Patterns**: RAG systems, function calling, MCP integration, and more

## Getting Started

**New to Swarms?** Start with the [Easy Example](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/easy_example.py) under Single Agent Examples → Core Agents.

**Looking for comprehensive tutorials?** Check out [The Swarms Cookbook](https://github.com/The-Swarm-Corporation/Cookbook) for detailed walkthroughs and advanced patterns.

**Want to see real-world applications?** Explore the Industry Applications section to see how Swarms solves practical problems.

## Quick Navigation


- [Single Agent Examples](#single-agent-examples) - Individual AI agents with various capabilities

- [Multi-Agent Examples](#multi-agent-examples) - Collaborative systems and swarm architectures

- [Additional Resources](#additional-resources) - Community links and support channels

## Single Agent Examples

### Core Agents
| Category | Example | Description |
|----------|---------|-------------|
| Basic | [Easy Example](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/easy_example.py) | Basic agent implementation demonstrating core functionality and setup |
| Settings | [Agent Settings](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/agent_settings.py) | Comprehensive configuration options for customizing agent behavior and capabilities |
| YAML | [Agents from YAML](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/agents_from_yaml_example.py) | Creating and configuring agents using YAML configuration files for easy deployment |
| Memory | [Agent with Long-term Memory](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/memory/agents_and_memory/agent_with_longterm_memory.py) | Implementation of persistent memory capabilities for maintaining context across sessions |

### Model Integrations
| Category | Example | Description |
|----------|---------|-------------|
| Azure | [Azure OpenAI Agent](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/settings/various_models/basic_agent_with_azure_openai.py) | Integration with Azure OpenAI services for enterprise-grade AI capabilities |
| Groq | [Groq Agent](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/settings/various_models/groq_agent.py) | High-performance inference using Groq's accelerated computing platform |
| Custom | [Custom Model Agent](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/settings/various_models/custom_model_with_agent.py) | Framework for integrating custom ML models into the agent architecture |
| Cerebras | [Cerebras Example](https://github.com/kyegomez/swarms/blob/master/examples/models/cerebas_example.py) | Integration with Cerebras AI platform for high-performance model inference |
| Claude | [Claude 4 Example](https://github.com/kyegomez/swarms/blob/master/examples/models/claude_4_example.py) | Anthropic Claude 4 model integration for advanced reasoning capabilities |
| Swarms Claude | [Swarms Claude Example](https://github.com/kyegomez/swarms/blob/master/examples/models/swarms_claude_example.py) | Optimized Claude integration within the Swarms framework |
| Lumo | [Lumo Example](https://github.com/kyegomez/swarms/blob/master/examples/models/lumo_example.py) | Lumo AI model integration for specialized tasks |
| Llama4 | [LiteLLM Example](https://github.com/kyegomez/swarms/blob/master/examples/models/llama4_examples/litellm_example.py) | Llama4 model integration using LiteLLM for efficient inference |

### Tools and Function Calling
| Category | Example | Description |
|----------|---------|-------------|
| Basic Tools | [Tool Agent](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/tools/tool_agent.py) | Basic tool-using agent demonstrating external tool integration capabilities |
| Advanced Tools | [Agent with Many Tools](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/tools/agent_with_many_tools.py) | Advanced agent utilizing multiple tools for complex task execution |
| OpenAI Functions | [OpenAI Function Caller](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/tools/function_calling/openai_function_caller_example.py) | Integration with OpenAI's function calling API for structured outputs |
| Command Line | [Command Tool Agent](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/tools/tool_agent/command_r_tool_agent.py) | Command-line interface tool integration |
| Jamba | [Jamba Tool Agent](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/tools/tool_agent/jamba_tool_agent.py) | Integration with Jamba framework for enhanced tool capabilities |
| Pydantic | [Pydantic Tool Agent](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/tools/tool_agent/tool_agent_pydantic.py) | Tool validation and schema enforcement using Pydantic |
| Function Caller | [Function Caller Example](https://github.com/kyegomez/swarms/blob/master/examples/demos/spike/function_caller_example.py) | Advanced function calling capabilities with dynamic tool execution |
| LiteLLM Tools | [LiteLLM Tool Example](https://github.com/kyegomez/swarms/blob/master/examples/single_agent/tools/litellm_tool_example.py) | Tool integration using LiteLLM for model-agnostic function calling |
| Swarms Tools | [Swarms Tools Example](https://github.com/kyegomez/swarms/blob/master/examples/single_agent/tools/swarms_tools_example.py) | Native Swarms tool ecosystem integration |
| Structured Outputs | [Structured Outputs Example](https://github.com/kyegomez/swarms/blob/master/examples/single_agent/tools/structured_outputs/structured_outputs_example.py) | Structured data output capabilities for consistent responses |
| Schema Validation | [Schema Validation Example](https://github.com/kyegomez/swarms/blob/master/examples/tools/base_tool_examples/schema_validation_example.py) | Tool schema validation and error handling |

### MCP (Model Context Protocol) Integration
| Category | Example | Description |
|----------|---------|-------------|
| Agent Tools | [Agent Tools Dict Example](https://github.com/kyegomez/swarms/blob/master/examples/mcp/mcp_examples/agent_use/agent_tools_dict_example.py) | MCP integration for dynamic tool management |
| MCP Execute | [MCP Execute Example](https://github.com/kyegomez/swarms/blob/master/examples/mcp/mcp_examples/utils/mcp_execute_example.py) | MCP command execution and response handling |
| MCP Load Tools | [MCP Load Tools Example](https://github.com/kyegomez/swarms/blob/master/examples/mcp/mcp_examples/utils/mcp_load_tools_example.py) | Dynamic tool loading through MCP protocol |
| Multiple Servers | [MCP Multiple Servers Example](https://github.com/kyegomez/swarms/blob/master/examples/mcp/mcp_utils/mcp_multiple_servers_example.py) | Multi-server MCP configuration and management |

### RAG and Memory
| Category | Example | Description |
|----------|---------|-------------|
| Full RAG | [Full Agent RAG Example](https://github.com/kyegomez/swarms/blob/master/examples/single_agent/rag/full_agent_rag_example.py) | Complete RAG implementation with retrieval and generation |
| Pinecone | [Pinecone Example](https://github.com/kyegomez/swarms/blob/master/examples/single_agent/rag/pinecone_example.py) | Vector database integration using Pinecone for semantic search |

### Reasoning and Decision Making
| Category | Example | Description |
|----------|---------|-------------|
| Agent Judge | [Agent Judge Example](https://github.com/kyegomez/swarms/blob/master/examples/single_agent/reasoning_agent_examples/agent_judge_example.py) | Agent-based decision making and evaluation system |
| Reasoning Duo | [Reasoning Duo Example](https://github.com/kyegomez/swarms/blob/master/examples/single_agent/reasoning_agent_examples/reasoning_duo_example.py) | Collaborative reasoning between two specialized agents |

### Vision and Multimodal
| Category | Example | Description |
|----------|---------|-------------|
| Image Batch | [Image Batch Example](https://github.com/kyegomez/swarms/blob/master/examples/single_agent/vision/image_batch_example.py) | Batch processing of multiple images with vision capabilities |
| Multimodal | [Multimodal Example](https://github.com/kyegomez/swarms/blob/master/examples/single_agent/vision/multimodal_example.py) | Multi-modal agent supporting text, image, and audio inputs |

### Utilities and Output Formats
| Category | Example | Description |
|----------|---------|-------------|
| XML Output | [XML Output Example](https://github.com/kyegomez/swarms/blob/master/examples/single_agent/utils/xml_output_example.py) | Structured XML output formatting for agent responses |
| CSV Agent | [CSV Agent Example](https://github.com/kyegomez/swarms/blob/master/examples/misc/csvagent_example.py) | CSV data processing and manipulation agent |
| Swarm Matcher | [Swarm Matcher Example](https://github.com/kyegomez/swarms/blob/master/examples/misc/swarm_matcher_example.py) | Agent matching and selection system |

### Third-Party Integrations
| Category | Example | Description |
|----------|---------|-------------|
| Microsoft | [AutoGen Integration](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/3rd_party_agents/auto_gen.py) | Integration with Microsoft's AutoGen framework for autonomous agents |
| LangChain | [LangChain Integration](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/3rd_party_agents/langchain.py) | Combining LangChain's capabilities with Swarms for enhanced functionality |
| Browser | [Multion Integration](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/3rd_party_agents/multion_agent.py) | Web automation and browsing capabilities using Multion |
| Team AI | [Crew AI](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/3rd_party_agents/crew_ai.py) | Team-based AI collaboration using Crew AI framework |
| Development | [Griptape](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/3rd_party_agents/griptape.py) | Integration with Griptape for structured AI application development |

### Industry-Specific Agents
| Category | Example | Description |
|----------|---------|-------------|
| Finance | [401k Agent](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/use_cases/finance/401k_agent.py) | Retirement planning assistant with investment strategy recommendations |
| Finance | [Estate Planning](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/use_cases/finance/estate_planning_agent.py) | Comprehensive estate planning and wealth management assistant |
| Security | [Perimeter Defense](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/use_cases/security/perimeter_defense_agent.py) | Security monitoring and threat detection system |
| Research | [Perplexity Agent](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/use_cases/research/perplexity_agent.py) | Advanced research automation using Perplexity AI integration |
| Legal | [Alberto Agent](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/use_cases/law/alberto_agent.py) | Legal research and document analysis assistant |
| Healthcare | [Pharma Agent](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/agents/use_cases/pharma/pharma_agent_two.py) | Pharmaceutical research and drug interaction analysis |

## Multi-Agent Examples

### Core Architectures
| Category | Example | Description |
|----------|---------|-------------|
| Basic | [Build a Swarm](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/structs/swarms/base_swarm/build_a_swarm.py) | Foundation for creating custom swarm architectures with multiple agents |
| Auto Swarm | [Auto Swarm](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/structs/swarms/auto_swarm/auto_swarm_example.py) | Self-organizing swarm with automatic task distribution and management |
| Concurrent | [Concurrent Swarm](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/structs/swarms/concurrent_swarm/concurrent_swarm_example.py) | Parallel execution of tasks across multiple agents for improved performance |
| Star | [Star Swarm](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/structs/swarms/different_architectures/star_swarm.py) | Centralized architecture with a hub agent coordinating peripheral agents |
| Circular | [Circular Swarm](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/structs/swarms/different_architectures/circular_swarm.py) | Ring topology for cyclic information flow between agents |
| Graph Workflow | [Graph Workflow Basic](https://github.com/kyegomez/swarms/blob/main/examples/structs/graph_workflow_basic.py) | Minimal graph workflow with two agents and one task |

### Concurrent and Parallel Processing
| Category | Example | Description |
|----------|---------|-------------|
| Concurrent | [Concurrent Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/concurrent_examples/concurrent_example.py) | Basic concurrent execution of multiple agents |
| Concurrent Swarm | [Concurrent Swarm Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/concurrent_examples/concurrent_swarm_example.py) | Advanced concurrent swarm with parallel task processing |

### Hierarchical and Sequential Workflows
| Category | Example | Description |
|----------|---------|-------------|
| Hierarchical | [Hierarchical Swarm Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/hiearchical_swarm/hiearchical_examples/hierarchical_swarm_example.py) | Multi-level hierarchical agent organization |
| Hierarchical Basic | [Hierarchical Swarm Basic](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/hiearchical_swarm/hiearchical_swarm-example.py) | Simplified hierarchical swarm implementation |
| Hierarchical Advanced | [Hierarchical Advanced](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/hiearchical_swarm/hierarchical_swarm_example.py) | Advanced hierarchical swarm with complex agent relationships |
| Sequential Workflow | [Sequential Workflow Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/sequential_workflow/sequential_workflow_example.py) | Linear workflow with agents processing tasks in sequence |
| Sequential Swarm | [Sequential Swarm Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/sequential_workflow/sequential_swarm_example.py) | Sequential swarm with coordinated task execution |

### Group Chat and Interactive Systems
| Category | Example | Description |
|----------|---------|-------------|
| Group Chat | [Group Chat Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/groupchat/groupchat_examples/group_chat_example.py) | Multi-agent group chat system with turn-based communication |
| Group Chat Advanced | [Group Chat Advanced](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/groupchat/groupchat_examples/groupchat_example.py) | Advanced group chat with enhanced interaction capabilities |
| Mortgage Panel | [Mortgage Tax Panel](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/groupchat/groupchat_examples/mortgage_tax_panel_example.py) | Specialized panel for mortgage and tax discussions |
| Interactive Group Chat | [Interactive Group Chat](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/groupchat/interactive_groupchat_example.py) | Interactive group chat with real-time user participation |
| Dynamic Speaker | [Random Dynamic Speaker](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/groupchat/random_dynamic_speaker_example.py) | Dynamic speaker selection in group conversations |
| Interactive Speaker | [Interactive Speaker Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/interactive_groupchat_examples/interactive_groupchat_speaker_example.py) | Interactive speaker management in group chats |
| Medical Panel | [Medical Panel Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/interactive_groupchat_examples/medical_panel_example.py) | Medical expert panel for healthcare discussions |
| Stream Example | [Stream Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/interactive_groupchat_examples/stream_example.py) | Streaming capabilities in interactive group chats |

### Research and Deep Analysis
| Category | Example | Description |
|----------|---------|-------------|
| Advanced Research | [Advanced Research System](https://github.com/The-Swarm-Corporation/AdvancedResearch) | Multi-agent research system inspired by Anthropic's research methodology with orchestrator-worker architecture |
| Deep Research | [Deep Research Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/deep_research_examples/deep_research_example.py) | Comprehensive research system with multiple specialized agents |
| Deep Research Swarm | [Deep Research Swarm](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/deep_research_examples/deep_research_swarm_example.py) | Swarm-based deep research with collaborative analysis |
| Scientific Agents | [Deep Research Swarm Example](https://github.com/kyegomez/swarms/blob/master/examples/demos/scient_agents/deep_research_swarm_example.py) | Scientific research swarm for academic and research applications |

### Routing and Decision Making
| Category | Example | Description |
|----------|---------|-------------|
| Model Router | [Model Router Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/mar/model_router_example.py) | Intelligent routing of tasks to appropriate model agents |
| Multi-Agent Router | [Multi-Agent Router Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/mar/multi_agent_router_example.py) | Advanced routing system for multi-agent task distribution |
| Swarm Router | [Swarm Router Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/swarm_router/swarm_router_example.py) | Swarm-specific routing and load balancing |
| Majority Voting | [Majority Voting Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/majority_voting/majority_voting_example.py) | Consensus-based decision making using majority voting |

### Council and Collaborative Systems
| Category | Example | Description |
|----------|---------|-------------|
| Council Judge | [Council Judge Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/council/council_judge_example.py) | Council-based decision making with expert judgment |

### Advanced Collaboration
| Category | Example | Description |
|----------|---------|-------------|
| Enhanced Collaboration | [Enhanced Collaboration Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/enhanced_collaboration_example.py) | Advanced collaboration patterns between multiple agents |
| Mixture of Agents | [Mixture of Agents Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/mixture_of_agents_example.py) | Heterogeneous agent mixture for diverse task handling |
| Aggregate | [Aggregate Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/aggregate_example.py) | Aggregation of results from multiple agents |

### API and Integration
| Category | Example | Description |
|----------|---------|-------------|
| Swarms API | [Swarms API Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/swarms_api_examples/swarms_api_example.py) | API integration for Swarms multi-agent systems |

### Utilities and Batch Processing
| Category | Example | Description |
|----------|---------|-------------|
| Batch Agent | [Batch Agent Example](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/utils/batch_agent_example.py) | Batch processing capabilities for multiple agents |

### Experimental Architectures
| Category | Example | Description |
|----------|---------|-------------|
| Monte Carlo | [Monte Carlo Swarm](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/structs/swarms/experimental/monte_carlo_swarm.py) | Probabilistic decision-making using Monte Carlo simulation across agents |
| Federated | [Federated Swarm](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/structs/swarms/experimental/federated_swarm.py) | Distributed learning system with privacy-preserving agent collaboration |
| Ant Colony | [Ant Swarm](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/structs/swarms/experimental/ant_swarm.py) | Bio-inspired optimization using ant colony algorithms for agent coordination |
| Matrix | [Agent Matrix](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/structs/swarms/experimental/agent_matrix.py) | Grid-based agent organization for complex problem-solving |
| DFS | [DFS Search Swarm](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/structs/swarms/experimental/dfs_search_swarm.py) | Depth-first search swarm for complex problem exploration |
| Pulsar | [Pulsar Swarm](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/structs/swarms/experimental/pulsar_swarm.py) | Pulsar-based coordination for synchronized agent behavior |

### Collaboration Patterns
| Category | Example | Description |
|----------|---------|-------------|
| Delegation | [Agent Delegation](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/structs/swarms/multi_agent_collaboration/agent_delegation.py) | Task delegation and management system |
| Communication | [Message Pool](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/structs/swarms/multi_agent_collaboration/message_pool.py) | Shared communication system for efficient agent interaction |
| Scheduling | [Round Robin](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/structs/swarms/multi_agent_collaboration/round_robin_example.py) | Round-robin task scheduling and execution |
| Load Balancing | [Load Balancer](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/structs/swarms/multi_agent_collaboration/load_balancer_example.py) | Dynamic task distribution system for optimal resource utilization |
| Consensus | [Majority Voting](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/structs/swarms/multi_agent_collaboration/majority_voting.py) | Consensus-building system using democratic voting among agents |

### Industry Applications
| Category | Example | Description |
|----------|---------|-------------|
| Finance | [Accountant Team](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/applications/demos/accountant_team/account_team2_example.py) | Multi-agent system for financial analysis, bookkeeping, and tax planning |
| Marketing | [Ad Generation](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/applications/demos/ad_gen/ad_gen_example.py) | Collaborative ad creation with copywriting and design agents |
| Aerospace | [Space Traffic Control](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/applications/demos/agentic_space_traffic_control/game.py) | Complex simulation of space traffic management with multiple coordinating agents |
| Agriculture | [Plant Biology](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/applications/demos/plant_biologist_swarm/agricultural_swarm.py) | Agricultural analysis and optimization using specialized biology agents |
| Urban Dev | [Urban Planning](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/applications/demos/urban_planning/urban_planning_example.py) | City development planning with multiple specialized urban development agents |
| Education | [Education System](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/applications/demos/education/education_example.py) | Personalized learning system with multiple teaching and assessment agents |
| Security | [Email Phishing Detection](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/applications/demos/email_phiser/email_swarm.py) | Multi-agent security analysis and threat detection |
| Fashion | [Personal Stylist](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/applications/demos/personal_stylist/personal_stylist_example.py) | Fashion recommendation system with style analysis and matching agents |
| Healthcare | [Healthcare Assistant](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/applications/demos/positive_med/positive_med_example.py) | Medical diagnosis and treatment planning with specialist consultation agents |
| Security Ops | [Security Team](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/applications/demos/security_team/security_team_example.py) | Comprehensive security operations with threat detection and response agents |
| Medical | [X-Ray Analysis](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/applications/demos/xray/xray_example.py) | Multi-agent medical imaging analysis and diagnosis |
| Business | [Business Strategy](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/applications/business_strategy/business_strategy_graph/growth_agent.py) | Strategic planning and business development swarm |
| Research | [Astronomy Research](https://github.com/The-Swarm-Corporation/swarms-examples/blob/main/examples/applications/astronomy/multiversal_detection/test.py) | Collaborative space research and astronomical analysis |


-------

## Connect With Us

Join our community of agent engineers and researchers for technical support, cutting-edge updates, and exclusive access to world-class agent engineering insights!

| Platform | Description | Link |
|----------|-------------|------|
| 📚 Documentation | Official documentation and guides | [docs.swarms.world](https://docs.swarms.world) |
| 📝 Blog | Latest updates and technical articles | [Medium](https://medium.com/@kyeg) |
| 💬 Discord | Live chat and community support | [Join Discord](https://discord.gg/EamjgSaEQf) |
| 🐦 Twitter | Latest news and announcements | [@swarms_corp](https://twitter.com/swarms_corp) |
| 👥 LinkedIn | Professional network and updates | [The Swarm Corporation](https://www.linkedin.com/company/the-swarm-corporation) |
| 📺 YouTube | Tutorials and demos | [Swarms Channel](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ) |
| 🎫 Events | Join our community events | [Sign up here](https://lu.ma/swarms_calendar) |
| 🚀 Onboarding Session | Get onboarded with Kye Gomez, creator and lead maintainer of Swarms | [Book Session](https://cal.com/swarms/swarms-onboarding-session) |
