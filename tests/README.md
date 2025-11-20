# Swarms Test Suite

This directory contains comprehensive tests for the Swarms framework, covering all major components including agents, workflows, tools, utilities, and more.

## 📁 Directory Structure

### Core Test Files
- **`test_comprehensive_test.py`** - Main comprehensive test suite that runs all major Swarms components
- **`test___init__.py`** - Package initialization tests
- **`requirements.txt`** - Test dependencies (swarms, pytest, matplotlib, loguru)

### Test Categories

#### 🤖 Agent Tests (`/agent/`)
Tests for individual agent functionality and behavior:

**`/agents/`** - Core agent functionality
- `test_agent_logging.py` - Agent logging and monitoring capabilities
- `test_create_agents_from_yaml.py` - YAML-based agent creation
- `test_litellm_args_kwargs.py` - LiteLLM argument handling
- `test_llm_args.py` - LLM argument processing
- `test_llm_handling_args.py` - LLM argument management
- `test_tool_agent.py` - Tool-enabled agent functionality

**`/benchmark_agent/`** - Agent performance and benchmarking
- `test_agent_benchmark_init.py` - Agent benchmark initialization
- `test_agent_exec_benchmark.py` - Agent execution benchmarking
- `test_auto_test_eval.py` - Automated test evaluation
- `test_github_summarizer_agent.py` - GitHub summarization agent
- `test_profiling_agent.py` - Agent performance profiling

#### 🏗️ Structure Tests (`/structs/`)
Tests for Swarms structural components and workflows:

- `test_agent.py` - Core Agent class functionality
- `test_agent_features.py` - Agent feature testing
- `test_agent_rearrange.py` - Agent rearrangement capabilities
- `test_agentrearrange.py` - Alternative agent rearrangement tests
- `test_airflow_swarm.py` - Airflow integration
- `test_auto_swarm_builder_fix.py` - Auto swarm builder fixes
- `test_auto_swarms_builder.py` - Automated swarm construction
- `test_base_workflow.py` - Base workflow functionality
- `test_base.py` - Base class implementations
- `test_board_of_directors_swarm.py` - Board of directors swarm pattern
- `test_concurrent_workflow.py` - Concurrent workflow execution
- `test_conversation.py` - Conversation management
- `test_forest_swarm.py` - Forest swarm architecture
- `test_graph_workflow_comprehensive.py` - Graph-based workflows
- `test_groupchat.py` - Group chat functionality
- `test_majority_voting.py` - Majority voting mechanisms
- `test_moa.py` - Mixture of Agents (MoA) testing
- `test_multi_agent_collab.py` - Multi-agent collaboration
- `test_multi_agent_orchestrator.py` - Multi-agent orchestration
- `test_reasoning_agent_router_all.py` - Reasoning agent routing
- `test_recursive_workflow.py` - Recursive workflow patterns
- `test_round_robin_swarm.py` - Round-robin swarm scheduling
- `test_sequential_workflow.py` - Sequential workflow execution
- `test_spreadsheet.py` - Spreadsheet swarm functionality
- `test_swarm_architectures.py` - Various swarm architectures
- `test_yaml_model.py` - YAML model configuration

#### 🔧 Tools Tests (`/tools/`)
Tests for tool integration and functionality:

- `test_base_tool.py` - Base tool class functionality
- `test_output_str_fix.py` - Output string formatting fixes
- `test_parse_tools.py` - Tool parsing and execution
- `test_support_mcp.py` - MCP (Model Context Protocol) support

#### 🛠️ Utilities Tests (`/utils/`)
Tests for utility functions and helpers:

- `test_acompletions.py` - Async completion handling
- `test_auto_check_download.py` - Automatic download checking
- `test_display_markdown_message.py` - Markdown message display
- `test_docstring_parser.py` - Docstring parsing utilities
- `test_extract_code_from_markdown.py` - Code extraction from markdown
- `test_formatter.py` - Text formatting utilities
- `test_litellm_wrapper.py` - LiteLLM wrapper functionality
- `test_math_eval.py` - Mathematical expression evaluation
- `test_md_output.py` - Markdown output handling
- `test_metrics_decorator.py` - Metrics collection decorators
- `test_pdf_to_text.py` - PDF to text conversion
- `test_try_except_wrapper.py` - Error handling wrappers

#### 🎨 Artifacts Tests (`/artifacts/`)
Tests for artifact management and versioning:

- `test_artifact_main.py` - Core artifact functionality
- `test_artifact_output_types.py` - Artifact output type handling

#### 💬 Communication Tests (`/communication/`)
Tests for communication and conversation management:

- `test_conversation.py` - Conversation handling and persistence

#### 📊 AOP (Aspect-Oriented Programming) Tests (`/aop/`)
Advanced testing with benchmarking and performance analysis:

- `aop_benchmark.py` - Comprehensive AOP benchmarking suite
- `test_data/` - Benchmark data and results
  - `aop_benchmark_data/` - Benchmark results and visualizations
  - `image1.jpg`, `image2.png` - Test images

#### 📈 Telemetry Tests (`/telemetry/`)
Tests for telemetry and monitoring:

- `test_user_utils.py` - User utility telemetry

## 🚀 Running Tests

### Prerequisites
Install test dependencies:
```bash
pip install -r requirements.txt
```

### Running All Tests
```bash
pytest
```

### Running Specific Test Categories
```bash
# Run agent tests
pytest agent/

# Run structure tests
pytest structs/

# Run utility tests
pytest utils/

# Run tool tests
pytest tools/
```

### Running Individual Test Files
```bash
# Run comprehensive test suite
pytest test_comprehensive_test.py

# Run specific test file
pytest structs/test_agent.py
```

### Running with Coverage
```bash
pytest --cov=swarms --cov-report=html
```

## 📋 Test Features

### Comprehensive Testing
- **Agent Functionality**: Complete testing of agent creation, execution, and management
- **Workflow Testing**: Various workflow patterns including sequential, concurrent, and recursive
- **Tool Integration**: Testing of tool parsing, execution, and MCP support
- **Performance Benchmarking**: AOP benchmarking with multiple LLM providers
- **Error Handling**: Comprehensive error handling and recovery testing

### Test Data
- Benchmark results with CSV and Excel exports
- Performance visualizations (PNG charts)
- Test images for multimodal testing
- Conversation cache files for persistence testing

### Supported LLM Providers
The AOP benchmark tests support multiple LLM providers:
- OpenAI (GPT-4o, GPT-4o-mini, GPT-4-turbo)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Haiku, Claude 3 Sonnet)
- Google (Gemini 1.5 Pro, Gemini 1.5 Flash)
- Meta (Llama 3.1 8B, Llama 3.1 70B)

## 🔧 Configuration

### Environment Variables
Tests require the following environment variables:
- `OPENAI_API_KEY` - OpenAI API key for testing
- Additional API keys for other providers (optional)

### Test Configuration
- Maximum agents: 20 (configurable in AOP benchmark)
- Requests per test: 20
- Concurrent requests: 5
- Timeout settings: Configurable per test type

## 📊 Benchmarking

The AOP benchmark suite provides:
- Performance metrics across multiple LLM providers
- Memory usage tracking
- Response time analysis
- Throughput measurements
- Visual performance reports

## 🐛 Debugging

### Verbose Output
```bash
pytest -v
```

### Debug Mode
```bash
pytest --pdb
```

### Logging
Tests use Loguru for comprehensive logging. Check console output for detailed test execution logs.

## 📝 Contributing

When adding new tests:
1. Follow the existing directory structure
2. Use descriptive test names
3. Include proper docstrings
4. Add appropriate fixtures and mocks
5. Update this README if adding new test categories

## 🔍 Test Coverage

The test suite aims for comprehensive coverage of:
- ✅ Agent creation and execution
- ✅ Workflow patterns and orchestration
- ✅ Tool integration and execution
- ✅ Utility functions and helpers
- ✅ Error handling and edge cases
- ✅ Performance and benchmarking
- ✅ Communication and conversation management
- ✅ Artifact management and versioning
