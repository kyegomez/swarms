# Changelog Examples - New Features Showcase

This folder contains comprehensive examples demonstrating all the new features introduced in the latest Swarms release. Each example is a standalone script that can be run directly to see the feature in action.

## 📚 Quick Reference - All Examples

| # | Example File | Feature | Category |
|---|--------------|---------|----------|
| 01 | [01_marketplace_prompt_fetching.py](01_marketplace_prompt_fetching.py) | Marketplace Prompt Fetching | Marketplace Integration |
| 02 | [02_round_robin_swarm_routing.py](02_round_robin_swarm_routing.py) | Round Robin Swarm Routing | Multi-Agent Structures |
| 03 | [03_graph_workflow_rustworkx.py](03_graph_workflow_rustworkx.py) | Graph Workflow with Rustworkx | Workflow Orchestration |
| 04 | [04_agent_rearrange.py](04_agent_rearrange.py) | Agent Rearrangement | Multi-Agent Structures |
| 05 | [05_swarm_rearrange.py](05_swarm_rearrange.py) | Swarm Rearrangement | Multi-Agent Structures |
| 06 | [06_spreadsheet_swarm.py](06_spreadsheet_swarm.py) | SpreadsheetSwarm | Workflow Orchestration |
| 07 | [07_model_router.py](07_model_router.py) | ModelRouter | Agent Management |
| 08 | [08_self_moa_seq.py](08_self_moa_seq.py) | SelfMoASeq | Agent Management |
| 09 | [09_single_voice_agent.py](09_single_voice_agent.py) | Single Voice Agent | Voice Agents |
| 10 | [10_multi_agent_voice_debate.py](10_multi_agent_voice_debate.py) | Multi-Agent Voice Debate | Voice Agents |
| 11 | [11_hierarchical_voice_swarm.py](11_hierarchical_voice_swarm.py) | Hierarchical Voice Swarm | Voice Agents |
| 12 | [12_debate_with_judge.py](12_debate_with_judge.py) | Debate with Judge | Evaluation & Debate |
| 13 | [13_council_as_judge.py](13_council_as_judge.py) | Council as a Judge | Evaluation & Debate |
| 14 | [14_swarm_router_round_robin.py](14_swarm_router_round_robin.py) | SwarmRouter Round Robin | Routing & Orchestration |
| 15 | [15_graph_workflow_batch_agents.py](15_graph_workflow_batch_agents.py) | Graph Workflow Batch Agents | Workflow Orchestration |
| - | [autosaving_example.py](autosaving_example.py) | Autosaving Feature | Agent Management |

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Examples by Category](#examples-by-category)
  - [Marketplace Integration](#marketplace-integration)
  - [Multi-Agent Structures](#multi-agent-structures)
  - [Workflow Orchestration](#workflow-orchestration)
  - [Agent Management](#agent-management)
  - [Voice Agents](#voice-agents)
  - [Evaluation & Debate](#evaluation--debate)
  - [Routing & Orchestration](#routing--orchestration)

## Prerequisites

Install the required packages:

```bash
# Core package
pip install -U swarms

# Optional dependencies for specific examples
pip install rustworkx          # For graph workflow examples (03, 15)
pip install voice-agents       # For voice agent examples (09, 10, 11)
```

## Environment Setup

Set your API keys as environment variables:

```bash
# Required for most examples
export OPENAI_API_KEY="your-openai-api-key"

# Required for marketplace examples (01)
export SWARMS_API_KEY="your-swarms-api-key"

# Optional: For other providers (ModelRouter, etc.)
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

Or create a `.env` file in your project root:

```env
OPENAI_API_KEY=your-openai-api-key
SWARMS_API_KEY=your-swarms-api-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
```

## Quick Start

Each example is a standalone Python script. Run any example directly:

```bash
# From the project root
python examples/guides/880_update_changelog_examples/01_marketplace_prompt_fetching.py

# Or from this directory
cd examples/guides/880_update_changelog_examples
python 01_marketplace_prompt_fetching.py
```

All examples execute immediately without requiring function definitions - they run and produce output directly.

---

## Examples by Category

### Marketplace Integration

#### [01_marketplace_prompt_fetching.py](01_marketplace_prompt_fetching.py)
**Feature:** Marketplace Prompt Fetching  
**Description:** Load prompts directly from the Swarms Marketplace using `marketplace_prompt_id`. The agent automatically fetches and uses the prompt as its system prompt.

**Key Features:**
- One-line prompt loading from marketplace
- Automatic system prompt configuration
- Versioned prompt orchestration

**Usage:**
```bash
python 01_marketplace_prompt_fetching.py
```

**Requirements:**
- `SWARMS_API_KEY` environment variable
- Valid marketplace prompt ID

---

### Multi-Agent Structures

#### [02_round_robin_swarm_routing.py](02_round_robin_swarm_routing.py)
**Feature:** Round Robin Swarm Routing  
**Description:** Demonstrates fair, cyclic agent execution with randomized turn order for varied interaction patterns.

**Key Features:**
- Fair distribution of tasks
- Randomized agent order each loop
- Full conversation context sharing
- AutoGen-style communication pattern

**Usage:**
```bash
python 02_round_robin_swarm_routing.py
```

#### [04_agent_rearrange.py](04_agent_rearrange.py)
**Feature:** Agent Rearrangement  
**Description:** Dynamic reordering and restructuring of agents at runtime using flow patterns.

**Key Features:**
- Sequential execution with `->`
- Concurrent execution with `,`
- Team awareness and flow information
- Runtime agent restructuring

**Usage:**
```bash
python 04_agent_rearrange.py
```

#### [05_swarm_rearrange.py](05_swarm_rearrange.py)
**Feature:** Swarm Rearrangement  
**Description:** Rearranging swarms of swarms with dynamic flow patterns for complex orchestration.

**Key Features:**
- Swarm-level rearrangement
- Flow pattern syntax
- Multi-level orchestration

**Usage:**
```bash
python 05_swarm_rearrange.py
```

#### [12_debate_with_judge.py](12_debate_with_judge.py)
**Feature:** Debate with Judge  
**Description:** Pro and Con agents debate a topic with iterative refinement through a judge agent.

**Key Features:**
- Preset agents for quick setup
- Iterative refinement loops
- Judge-based synthesis
- Argument evaluation

**Usage:**
```bash
python 12_debate_with_judge.py
```

#### [13_council_as_judge.py](13_council_as_judge.py)
**Feature:** Council as a Judge  
**Description:** Multi-dimensional evaluation of task responses across multiple specialized judge agents.

**Key Features:**
- Parallel evaluation across dimensions
- Specialized judge agents
- Aggregated comprehensive reports
- Multi-criteria assessment

**Usage:**
```bash
python 13_council_as_judge.py
```

---

### Workflow Orchestration

#### [03_graph_workflow_rustworkx.py](03_graph_workflow_rustworkx.py)
**Feature:** Graph Workflow with Rustworkx Backend  
**Description:** High-performance workflow orchestration using rustworkx backend for 5-10x faster execution.

**Key Features:**
- Rustworkx backend for performance
- Directed graph structure
- Parallel execution within layers
- Automatic compilation

**Usage:**
```bash
python 03_graph_workflow_rustworkx.py
```

**Requirements:**
- `rustworkx` package installed

#### [15_graph_workflow_batch_agents.py](15_graph_workflow_batch_agents.py)
**Feature:** Graph Workflow Batch Agent Addition  
**Description:** Demonstrates batch agent addition and parallel execution patterns with layer-based workflows.

**Key Features:**
- Batch agent addition
- Layer-based parallel execution
- Fan-out/fan-in patterns
- Complex workflow orchestration

**Usage:**
```bash
python 15_graph_workflow_batch_agents.py
```

**Requirements:**
- `rustworkx` package installed

#### [06_spreadsheet_swarm.py](06_spreadsheet_swarm.py)
**Feature:** SpreadsheetSwarm  
**Description:** Concurrent processing of tasks with automatic CSV tracking of results and metadata.

**Key Features:**
- Concurrent agent execution
- Automatic CSV output
- Metadata tracking
- Batch processing support

**Usage:**
```bash
python 06_spreadsheet_swarm.py
```

---

### Agent Management

#### [07_model_router.py](07_model_router.py)
**Feature:** ModelRouter  
**Description:** Intelligent model selection and execution based on task requirements.

**Key Features:**
- Automatic model selection
- Task-based routing
- Multi-provider support
- Cost optimization

**Usage:**
```bash
python 07_model_router.py
```

#### [08_self_moa_seq.py](08_self_moa_seq.py)
**Feature:** SelfMoASeq (Self-Mixture of Agents Sequential)  
**Description:** Generates multiple outputs from a single model and aggregates them sequentially using a sliding window approach.

**Key Features:**
- Ensemble method with single model
- Sliding window aggregation
- Context length management
- Sequential synthesis

**Usage:**
```bash
python 08_self_moa_seq.py
```

---

### Voice Agents

#### [09_single_voice_agent.py](09_single_voice_agent.py)
**Feature:** Single Voice Agent  
**Description:** Single agent with text-to-speech (TTS) capabilities for real-time speech output.

**Key Features:**
- Real-time TTS streaming
- Distinct voice profiles
- Streaming callback integration
- Hands-free interaction

**Usage:**
```bash
python 09_single_voice_agent.py
```

**Requirements:**
- `voice-agents` package installed
- `OPENAI_API_KEY` for TTS

#### [10_multi_agent_voice_debate.py](10_multi_agent_voice_debate.py)
**Feature:** Multi-Agent Voice Debate  
**Description:** Two agents debating a topic using different voices with text-to-speech capabilities.

**Key Features:**
- Multiple distinct voices
- Real-time voice debate
- Conversation history tracking
- Differentiated speaker identification

**Usage:**
```bash
python 10_multi_agent_voice_debate.py
```

**Requirements:**
- `voice-agents` package installed
- `OPENAI_API_KEY` for TTS

#### [11_hierarchical_voice_swarm.py](11_hierarchical_voice_swarm.py)
**Feature:** Hierarchical Voice Swarm  
**Description:** Hierarchical swarm where agents communicate through voice using distinct TTS voices.

**Key Features:**
- Hierarchical agent structure
- Multiple voice profiles
- Director-worker coordination
- Real-time voice collaboration

**Usage:**
```bash
python 11_hierarchical_voice_swarm.py
```

**Requirements:**
- `voice-agents` package installed
- `OPENAI_API_KEY` for TTS

---

### Routing & Orchestration

#### [14_swarm_router_round_robin.py](14_swarm_router_round_robin.py)
**Feature:** SwarmRouter with Round Robin  
**Description:** SwarmRouter using RoundRobin routing strategy for fair, cyclic agent execution.

**Key Features:**
- Round-robin routing in SwarmRouter
- Fair task distribution
- Improved communication flow
- Multiple swarm type support

**Usage:**
```bash
python 14_swarm_router_round_robin.py
```

---

## Running All Examples

To run all examples sequentially:

```bash
# From the examples directory
cd examples/guides/880_update_changelog_examples

# Run all examples
for file in *.py; do
    echo "Running $file..."
    python "$file"
    echo "---"
done
```

## Example Output

Each example will:
1. Initialize the required agents/swarms
2. Execute the task
3. Print the results to console

Example output format:
```
[Feature Name] Result:
[Output content here]
```

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

4. **Graph Workflow Errors**
   - Install `rustworkx`: `pip install rustworkx`
   - For NetworkX fallback, ensure `networkx` is installed

## Additional Resources

- [Swarms Documentation](https://docs.swarms.world)
- [Swarms GitHub Repository](https://github.com/kyegomez/swarms)
- [Swarms Marketplace](https://swarms.world/marketplace)

## Contributing

If you find issues or have suggestions for these examples, please:
1. Check existing issues on GitHub
2. Create a new issue with details
3. Or submit a pull request with improvements

---

**Note:** All examples are designed to be simple and educational. They demonstrate core functionality without complex error handling or production-ready patterns. Adapt them to your specific use cases as needed.
