"""Persistence and topology conversion logic for GraphWorkflow."""
import json
from typing import Any, Dict
from swarms.structs.graph_workflow.models import Node, Edge, NodeType
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="graph_workflow")

def to_topology_spec(workflow: Any) -> Dict[str, Any]:
    """
    Generate an executable specification for this workflow.
    """
    if workflow.verbose:
        logger.debug("Generating executable topology specification")

    if not workflow._compiled:
        logger.warning(
            "Generating specification for uncompiled workflow. Sorting may not be optimal."
        )

    # Collect nodes spec
    nodes_spec = {}
    for node_id, node in workflow.nodes.items():
        node_spec = {
            "type": node.node_type.value,
            "agent_name": getattr(node.agent, "agent_name", "Unknown"),
            "agent_class": node.agent.__class__.__name__
            if hasattr(node.agent, "__class__")
            else "Unknown",
        }

        # Add optional attributes
        if node.metadata:
            node_spec["metadata"] = node.metadata

        if hasattr(node.agent, "system_prompt"):
            node_spec["system_prompt"] = node.agent.system_prompt
        if hasattr(node.agent, "model_name"):
            node_spec["model_name"] = node.agent.model_name
        if hasattr(node.agent, "description"):
            node_spec["description"] = node.agent.description

        nodes_spec[node_id] = node_spec

    # Collect edges spec
    edges_spec = []
    
    # Sort edges for deterministic output
    def get_edge_sort_key(edge):
        # Find which layer source is in
        layer_idx = 999
        if hasattr(workflow, "_sorted_layers"):
            for i, layer in enumerate(workflow._sorted_layers):
                if edge.source in layer:
                    layer_idx = i
                    break
        return (layer_idx, edge.source, edge.target)
        
    sorted_edges = sorted(workflow.edges, key=get_edge_sort_key)

    for edge in sorted_edges:
        edge_spec = {
            "source": edge.source,
            "target": edge.target,
        }
        if edge.metadata:
            edge_spec["metadata"] = edge.metadata
        edges_spec.append(edge_spec)

    # Build full spec
    spec = {
        "name": workflow.name or "Swarm_Workflow",
        "description": workflow.description or "Automated multi-agent workflow",
        "version": "1.0",
        "nodes": nodes_spec,
        "edges": edges_spec,
        "entry_points": workflow.entry_points,
        "end_points": workflow.end_points,
    }

    return spec


def from_topology_spec(cls: Any, spec: Dict[str, Any], agent_registry: Dict[str, Any] = None) -> Any:
    """
    Create a new GraphWorkflow from a topology specification.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"Topology specification must be a dictionary, got {type(spec)}")

    logger.info(f"Creating GraphWorkflow from spec: {spec.get('name', 'Unnamed')}")

    # Default empty registry if none provided
    registry = agent_registry or {}

    # Initialize workflow
    workflow = cls(
        name=spec.get("name", "Generated_Workflow"),
        description=spec.get("description"),
    )

    # Reconstruct nodes
    nodes_added = 0
    spec_nodes = spec.get("nodes", {})
    for node_id, node_data in spec_nodes.items():
        if not isinstance(node_data, dict):
            logger.warning(f"Skipping malformed node {node_id}: {node_data}")
            continue

        agent_name = node_data.get("agent_name")
        node_type_str = node_data.get("type", NodeType.ROUTER.value)
        metadata = node_data.get("metadata", {})

        # Convert string to enum
        try:
            node_type = NodeType(node_type_str)
        except ValueError:
            logger.warning(
                f"Invalid node type {node_type_str} for {node_id}, defaulting to ROUTER"
            )
            node_type = NodeType.ROUTER

        # Find agent instance
        agent = None
        if agent_name in registry:
            agent = registry[agent_name]
        else:
            logger.warning(f"Agent '{agent_name}' for node {node_id} not found in registry. Using placeholder.")
            # Create a minimal mock agent structure if needed (depends on Swarms internals)
            # We'll just define a dynamic class for it
            class MockAgent:
                def __init__(self, name):
                    self.agent_name = name
                def run(self, task):
                    return f"Mock execution for {self.agent_name}: {task}"
            agent = MockAgent(agent_name)

        workflow.add_node(agent, node_type, name=node_id, metadata=metadata)
        nodes_added += 1

    # Reconstruct edges
    edges_added = 0
    spec_edges = spec.get("edges", [])
    for edge_data in spec_edges:
        if not isinstance(edge_data, dict):
            logger.warning(f"Skipping malformed edge: {edge_data}")
            continue

        source = edge_data.get("source")
        target = edge_data.get("target")
        metadata = edge_data.get("metadata", {})

        if source and target:
            if source in spec_nodes and target in spec_nodes:
                workflow.add_edge(source, target, metadata=metadata)
                edges_added += 1
            else:
                logger.warning(f"Skipping edge {source}->{target}: One or both nodes don't exist")

    # Manually configure entry/end points if specified
    if "entry_points" in spec:
        for ep in spec["entry_points"]:
            workflow.set_entry_point(ep)

    if "end_points" in spec:
        for ep in spec["end_points"]:
            workflow.set_end_point(ep)

    logger.success(
        f"GraphWorkflow reconstruction complete: {nodes_added} nodes, {edges_added} edges"
    )

    return workflow
