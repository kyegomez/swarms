# Reasoning Agents Examples

This directory contains examples demonstrating advanced reasoning capabilities and agent evaluation systems in Swarms.

## Reasoning Agent Router Examples

The `reasoning_agent_router_examples/` folder contains simple examples for each agent type supported by the `ReasoningAgentRouter`:

- [reasoning_duo_example.py](reasoning_agent_router_examples/reasoning_duo_example.py) - Reasoning Duo agent for collaborative reasoning
- [self_consistency_example.py](reasoning_agent_router_examples/self_consistency_example.py) - Self-Consistency agent with multiple samples
- [ire_example.py](reasoning_agent_router_examples/ire_example.py) - Iterative Reflective Expansion (IRE) agent
- [agent_judge_example.py](reasoning_agent_router_examples/agent_judge_example.py) - Agent Judge for evaluation and judgment
- [reflexion_agent_example.py](reasoning_agent_router_examples/reflexion_agent_example.py) - Reflexion agent with memory capabilities
- [gkp_agent_example.py](reasoning_agent_router_examples/gkp_agent_example.py) - Generated Knowledge Prompting (GKP) agent

## Agent Judge Examples

The `agent_judge_examples/` folder contains detailed examples of the AgentJudge system:

- [example1_basic_evaluation.py](agent_judge_examples/example1_basic_evaluation.py) - Basic agent evaluation
- [example2_technical_evaluation.py](agent_judge_examples/example2_technical_evaluation.py) - Technical evaluation criteria
- [example3_creative_evaluation.py](agent_judge_examples/example3_creative_evaluation.py) - Creative evaluation patterns

## Tree of Thoughts Examples

The `tree_of_thoughts_examples/` folder contains `TreeOfThoughts` examples in mathematics, physics and general reasoning, each with a checkable answer. See its [README](tree_of_thoughts_examples/README.md) for how to choose settings.

- [mathematics/](tree_of_thoughts_examples/mathematics/) - Number theory counting, an irrationality proof, and calculus optimization
- [physics/](tree_of_thoughts_examples/physics/) - Projectile motion, a loop-the-loop energy problem, and a Fermi estimate
- [reasoning/](tree_of_thoughts_examples/reasoning/) - Knights and knaves, constraint scheduling, and bridge-and-torch planning

## Self-MoA Sequential Examples

- [moa_seq_example.py](moa_seq_example.py) - Self-MoA Sequential reasoning example for complex problem-solving
