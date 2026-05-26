# Swarms Overview

Swarms is a framework for building production-oriented single-agent and multi-agent systems. It provides agent primitives, workflows, routing, tools, memory, structured outputs, and deployment-oriented utilities.

This page preserves the `swarms/overview` route used by existing documentation links and points users to the maintained overview pages.

## Core Areas

- **Agents**: configurable workers with prompts, tools, model settings, and memory.
- **Structs**: reusable orchestration patterns such as sequential, concurrent, graph, group chat, and router workflows.
- **Tools**: callable functions, schemas, and MCP utilities that extend agent capabilities.
- **Examples**: runnable patterns for single-agent, multi-agent, tool, RAG, and deployment workflows.
- **Contributors**: setup, docs, test, and contribution guides.

## Recommended Starting Points

- [Installation](install/install.md)
- [Agents Index](agents/index.md)
- [Structs Overview](structs/overview.md)
- [Tool System](tools/main.md)
- [Basic Agent Example](examples/basic_agent.md)

## Choosing a Pattern

Start with a single `Agent` for focused tasks. Move to `SequentialWorkflow` when the work has ordered steps, `ConcurrentWorkflow` when work can happen in parallel, and `GraphWorkflow` when you need explicit dependencies between nodes. Use `SwarmRouter` when the architecture should be selected at runtime.
