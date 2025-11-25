# Rustworkx Backend Examples

This directory contains comprehensive examples demonstrating the use of the **rustworkx backend** in GraphWorkflow. Rustworkx provides faster graph operations compared to NetworkX, especially for large graphs and complex operations.

## Installation

Before running these examples, ensure rustworkx is installed:

```bash
pip install rustworkx
```

If rustworkx is not installed, GraphWorkflow will automatically fallback to NetworkX backend.

## Examples Overview

### 01_basic_usage.py
Basic example showing how to use rustworkx backend with GraphWorkflow. Demonstrates simple linear workflow creation and execution.

**Key Concepts:**
- Initializing GraphWorkflow with rustworkx backend
- Adding agents and creating edges
- Running a workflow

### 02_backend_comparison.py
Compares NetworkX and Rustworkx backends side-by-side, showing performance differences and functional equivalence.

**Key Concepts:**
- Backend comparison
- Performance metrics
- Functional equivalence verification

### 03_fan_out_fan_in_patterns.py
Demonstrates parallel processing patterns: fan-out (one-to-many) and fan-in (many-to-one) connections.

**Key Concepts:**
- Fan-out pattern: `add_edges_from_source()`
- Fan-in pattern: `add_edges_to_target()`
- Parallel execution optimization

### 04_complex_workflow.py
Shows a complex multi-layer workflow with multiple parallel branches and convergence points.

**Key Concepts:**
- Multi-layer workflows
- Parallel chains: `add_parallel_chain()`
- Complex graph structures

### 05_performance_benchmark.py
Benchmarks performance differences between NetworkX and Rustworkx for various graph sizes and structures.

**Key Concepts:**
- Performance benchmarking
- Scalability testing
- Different graph topologies (chain, tree)

### 06_error_handling.py
Demonstrates error handling and graceful fallback behavior when rustworkx is unavailable.

**Key Concepts:**
- Error handling
- Automatic fallback to NetworkX
- Backend availability checking

### 07_large_scale_workflow.py
Demonstrates rustworkx's efficiency with large-scale workflows containing many agents.

**Key Concepts:**
- Large-scale workflows
- Performance with many nodes/edges
- Complex interconnections

### 08_parallel_chain_example.py
Detailed example of the parallel chain pattern creating a full mesh connection.

**Key Concepts:**
- Parallel chain pattern
- Full mesh connections
- Maximum parallelization

### 09_workflow_validation.py
Shows workflow validation features including cycle detection, isolated nodes, and auto-fixing.

**Key Concepts:**
- Workflow validation
- Cycle detection
- Auto-fixing capabilities

### 10_real_world_scenario.py
A realistic market research workflow demonstrating real-world agent coordination scenarios.

**Key Concepts:**
- Real-world use case
- Complex multi-phase workflow
- Practical application

## Quick Start

Run any example:

```bash
python 01_basic_usage.py
```

## Backend Selection

To use rustworkx backend:

```python
workflow = GraphWorkflow(
    backend="rustworkx",  # Use rustworkx
    # ... other parameters
)
```

To use NetworkX backend (default):

```python
workflow = GraphWorkflow(
    backend="networkx",  # Or omit for default
    # ... other parameters
)
```

## Performance Benefits

Rustworkx provides performance benefits especially for:
- **Large graphs** (100+ nodes)
- **Complex operations** (topological sorting, cycle detection)
- **Frequent graph modifications** (adding/removing nodes/edges)

## Key Differences

While both backends are functionally equivalent, rustworkx:
- Uses integer indices internally (abstracted away)
- Provides faster graph operations
- Better memory efficiency for large graphs
- Maintains full compatibility with GraphWorkflow API

## Notes

- Both backends produce identical results
- Rustworkx automatically falls back to NetworkX if not installed
- All GraphWorkflow features work with both backends
- Performance gains become more significant with larger graphs

## Requirements

- `swarms` package
- `rustworkx` (optional, for rustworkx backend)
- `networkx` (always available, default backend)

## Contributing

Feel free to add more examples demonstrating rustworkx capabilities or specific use cases!

