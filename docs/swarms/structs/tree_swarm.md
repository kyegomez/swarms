# Tree Swarm

Tree Swarm organizes agents into tree structures so work can flow from a root agent through branches of specialized agents. The implementation lives in `swarms.structs.tree_swarm` and exposes `TreeAgent`, `Tree`, and `ForestSwarm` patterns.

Use this route when architecture docs link to Tree Swarm directly. For broader forest-style orchestration, see [ForestSwarm](forest_swarm.md).

## When to Use

- Hierarchical decision trees
- Routing a task through specialized branches
- Multi-stage analysis where each branch handles a focused area
- Forest-style workflows that combine several trees

## Basic Shape

```python
from swarms.structs.tree_swarm import ForestSwarm, Tree, TreeAgent


root = TreeAgent(agent_name="Root")
research = TreeAgent(agent_name="Research")
review = TreeAgent(agent_name="Review")

tree = Tree(root=root)
tree.add_child(root, research)
tree.add_child(root, review)

forest = ForestSwarm(trees=[tree])
```

Adapt the tree structure to the agents and routing logic your workflow needs.

## Related Docs

- [ForestSwarm](forest_swarm.md)
- [Swarm Architectures](../concept/swarm_architectures.md)
- [GraphWorkflow](graph_workflow.md)

## Source Examples

- `examples/multi_agent/tree_swarm_new_updates.py`
- `examples/multi_agent/forest_swarm_examples/`
