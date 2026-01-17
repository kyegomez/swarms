# Changelog 890 Examples - January 2026 Release

This folder contains comprehensive examples demonstrating all the new features introduced in the January 2026 Swarms release. Each example is a standalone script that showcases specific functionality.

## 📚 Quick Reference - All Examples

| # | Example File | Feature | Category |
|---|--------------|---------|----------|
| 01 | [01_dynamic_skills_loader.py](01_dynamic_skills_loader.py) | DynamicSkillsLoader Integration | Agent Enhancement |
| 02 | [02_autonomous_agent_loop.py](02_autonomous_agent_loop.py) | Autonomous Agent Loop Architecture | Agent Architecture |
| 03 | [03_agent_handoffs.py](03_agent_handoffs.py) | Enhanced Function Calling for Agent Handoffs | Multi-Agent |
| 04 | [04_api_key_validation.py](04_api_key_validation.py) | API Key Validation System | Infrastructure |
| 05 | [05_max_loops_parameter.py](05_max_loops_parameter.py) | Max Loops Parameter Refactoring | Agent Configuration |
| 06 | [06_multi_tool_agent_tutorial.py](06_multi_tool_agent_tutorial.py) | Multi-Tool Agent Tutorial with X402 | Tools Integration |
| 07 | [07_hierarchical_voice_agent.py](07_hierarchical_voice_agent.py) | Hierarchical Voice Agent | Voice Agents |
| 08 | [08_agent_rearrange_patterns.py](08_agent_rearrange_patterns.py) | Agent Rearrange Patterns | Workflow Orchestration |

## Prerequisites

Install the required packages:

```bash
# Core package
pip install -U swarms

# Optional dependencies for specific examples
pip install swarms-tools          # For multi-tool examples (06)
pip install voice-agents          # For voice agent examples (07)
```

## Environment Setup

Set your API keys as environment variables:

```bash
# Required for most examples
export OPENAI_API_KEY="your-openai-api-key"

# Required for marketplace examples (04)
export SWARMS_API_KEY="your-swarms-api-key"
```

## Quick Start

Each example is a standalone Python script. Run any example directly:

```bash
# From the project root
python examples/guides/changelog_890/01_dynamic_skills_loader.py

# Or from this directory
cd examples/guides/changelog_890
python 01_dynamic_skills_loader.py
```

---

## Examples by Category

### Agent Enhancement

#### [01_dynamic_skills_loader.py](01_dynamic_skills_loader.py)
**Feature:** DynamicSkillsLoader Integration
**Description:** Automatic skill loading based on task requirements using similarity matching.

#### [05_max_loops_parameter.py](05_max_loops_parameter.py)
**Feature:** Max Loops Parameter Refactoring
**Description:** Demonstrates the max_loops parameter (renamed from max_iterations) with backwards compatibility.

### Agent Architecture

#### [02_autonomous_agent_loop.py](02_autonomous_agent_loop.py)
**Feature:** Autonomous Agent Loop Architecture
**Description:** Plan-think-act-continue pattern for independent agent operation with max_loops="auto".

### Multi-Agent

#### [03_agent_handoffs.py](03_agent_handoffs.py)
**Feature:** Enhanced Function Calling for Agent Handoffs
**Description:** Improved inter-agent communication with type-safe handoffs and context preservation.

#### [08_agent_rearrange_patterns.py](08_agent_rearrange_patterns.py)
**Feature:** Agent Rearrange Patterns
**Description:** Dynamic agent reconfiguration with sequential and concurrent execution patterns.

### Infrastructure

#### [04_api_key_validation.py](04_api_key_validation.py)
**Feature:** API Key Validation System
**Description:** Proactive API key validation to prevent runtime authentication errors.

### Tools Integration

#### [06_multi_tool_agent_tutorial.py](06_multi_tool_agent_tutorial.py)
**Feature:** Multi-Tool Agent Tutorial with X402
**Description:** Comprehensive guide for building agents with multiple X402 tools for complex workflows.

### Voice Agents

#### [07_hierarchical_voice_agent.py](07_hierarchical_voice_agent.py)
**Feature:** Hierarchical Voice Agent
**Description:** Voice-enabled hierarchical swarms with distinct TTS voices for each agent.

---

## Running All Examples

To run all examples sequentially:

```bash
# From the examples directory
cd examples/guides/changelog_890

# Run all examples
for file in *.py; do
    echo "Running $file..."
    python "$file"
    echo "---"
done
```

## Example Output

Each example will execute immediately and produce output demonstrating the featured functionality.

---

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure `swarms` is installed: `pip install -U swarms`
   - Check that optional dependencies are installed for specific examples

2. **API Key Errors**
   - Verify environment variables are set correctly
   - Check that API keys are valid and have sufficient credits

3. **Voice Agent Issues**
   - Ensure `voice-agents` package is installed
   - Verify OpenAI API key has TTS access

4. **Tool Integration Errors**
   - Install `swarms-tools`: `pip install swarms-tools`
   - Check that external service APIs are accessible

## Additional Resources

- [Swarms Documentation](https://docs.swarms.world)
- [Swarms GitHub Repository](https://github.com/kyegomez/swarms)
- [Swarms Marketplace](https://swarms.world/marketplace)

---

**Note:** All examples are designed to be simple and educational. They demonstrate core functionality without complex error handling or production-ready patterns. Adapt them to your specific use cases as needed.