"""Visualization utilities for GraphWorkflow using Graphviz."""
import uuid
import time
from typing import Any, Dict
from pathlib import Path

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    graphviz = None

from swarms.utils.loguru_logger import initialize_logger
logger = initialize_logger(log_folder="graph_workflow")

def visualize_workflow(
    workflow: Any,
    format: str = "png",
    view: bool = True,
    engine: str = "dot",
    show_summary: bool = False,
) -> str:
    """
    Visualize the workflow graph using Graphviz with enhanced parallel pattern detection.
    """
    output_path = f"{workflow.name}_visualization_{str(uuid.uuid4())}"

    if not GRAPHVIZ_AVAILABLE:
        error_msg = "Graphviz is not installed. Install it with: pip install graphviz"
        logger.error(error_msg)
        raise ImportError(error_msg)

    if workflow.verbose:
        logger.debug(
            f"Visualizing GraphWorkflow with Graphviz (format={format}, engine={engine})"
        )

    try:
        dot = graphviz.Digraph(
            name=f"GraphWorkflow_{workflow.name or 'Unnamed'}",
            comment=f"GraphWorkflow: {workflow.description or 'No description'}",
            engine=engine,
            format=format,
        )

        dot.attr(rankdir="TB")
        dot.attr(bgcolor="white")
        dot.attr(fontname="Arial")
        dot.attr(fontsize="12")
        dot.attr(labelloc="t")
        dot.attr(
            label=f'GraphWorkflow: {workflow.name or "Unnamed"}\n{len(workflow.nodes)} Agents, {len(workflow.edges)} Connections'
        )

        dot.attr("node", shape="box", style="rounded,filled", fontname="Arial", fontsize="10", margin="0.1,0.05")
        dot.attr("edge", fontname="Arial", fontsize="8", arrowsize="0.8")

        fan_out_nodes = {}
        fan_in_nodes = {}

        for edge in workflow.edges:
            if edge.source not in fan_out_nodes:
                fan_out_nodes[edge.source] = []
            fan_out_nodes[edge.source].append(edge.target)

            if edge.target not in fan_in_nodes:
                fan_in_nodes[edge.target] = []
            fan_in_nodes[edge.target].append(edge.source)

        for node_id, node in workflow.nodes.items():
            agent_name = getattr(node.agent, "agent_name", node_id)
            sys_prompt = getattr(node.agent, "system_prompt", "")
            
            prompt_preview = (
                sys_prompt[:50] + "..." if len(sys_prompt) > 50 else sys_prompt
            )

            fillcolor = "#e1f5fe"
            color = "#0288d1"
            penwidth = "1"

            if node_id in workflow.entry_points:
                fillcolor = "#e8f5e9"
                color = "#2e7d32"
                penwidth = "2"
            elif node_id in workflow.end_points:
                fillcolor = "#fff3e0"
                color = "#ef6c00"
                penwidth = "2"

            label = f"""<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                <TR><TD><B>{agent_name}</B></TD></TR>
                <TR><TD><FONT POINT-SIZE="8" COLOR="gray30">ID: {node_id}</FONT></TD></TR>
            """

            if prompt_preview:
                label += f"""<TR><TD><FONT POINT-SIZE="8" COLOR="gray50">({prompt_preview})</FONT></TD></TR>"""

            label += "</TABLE>>"

            dot.node(node_id, label=label, fillcolor=fillcolor, color=color, penwidth=penwidth)

        for edge in workflow.edges:
            color = "gray50"
            penwidth = "1"
            style = "solid"
            label = ""

            is_fan_out = len(fan_out_nodes.get(edge.source, [])) > 1
            is_fan_in = len(fan_in_nodes.get(edge.target, [])) > 1

            if is_fan_out and is_fan_in:
                color = "#9c27b0"
                style = "dashed"
            elif is_fan_out:
                color = "#1976d2"
            elif is_fan_in:
                color = "#388e3c"
                
            if edge.metadata:
                metadata_str = ", ".join(f"{k}: {v}" for k, v in edge.metadata.items())
                label = f'<<FONT POINT-SIZE="6">{metadata_str}</FONT>>'

            dot.edge(edge.source, edge.target, color=color, penwidth=penwidth, style=style, label=label)

        if getattr(workflow, "_compiled", False) and len(getattr(workflow, "_sorted_layers", [])) > 1:
            for i, layer in enumerate(getattr(workflow, "_sorted_layers", [])):
                with dot.subgraph(name=f"cluster_layer_{i}") as c:
                    c.attr(label=f"Layer {i}", style="dashed", color="gray80", fontcolor="gray50", fontsize="10")
                    for node_id in layer:
                        c.node(node_id)

        if output_path is None:
            output_path = f"workflow_{workflow.id[:8]}_{int(time.time())}"

        output_file = dot.render(output_path, view=view, cleanup=True)

        if show_summary:
            print("\n" + "=" * 50)
            print(f"Workflow Pattern Summary: {workflow.name or 'Unnamed'}")
            print("=" * 50)

            multi_targets = {s: t for s, t in fan_out_nodes.items() if len(t) > 1}
            if multi_targets:
                print("\n📡 Fan-Out Patterns (Parallel Distribution):")
                for source, targets in multi_targets.items():
                    print(f"  • {source} → {len(targets)} agents")

            multi_sources = {t: s for t, s in fan_in_nodes.items() if len(s) > 1}
            if multi_sources:
                print("\n🎯 Fan-In Patterns (Parallel Convergence):")
                for target, sources in multi_sources.items():
                    print(f"  • {len(sources)} agents → {target}")
            print("=" * 50 + "\n")

        if workflow.verbose:
            logger.success(f"Visualization saved to {output_file}")

        return output_file

    except Exception as e:
        logger.exception(f"Error in visualize_workflow: {e}")
        raise e

def visualize_workflow_simple(workflow: Any) -> str:
    """
    Simple text-based visualization for environments without Graphviz.
    """
    if workflow.verbose:
        logger.debug("Generating simple text visualization")

    try:
        lines = []
        lines.append(f"GraphWorkflow: {workflow.name or 'Unnamed'}")
        lines.append(f"Description: {workflow.description or 'No description'}")
        lines.append(f"Nodes: {len(workflow.nodes)}, Edges: {len(workflow.edges)}")
        lines.append("")

        lines.append("🤖 Agents:")
        for node_id, node in workflow.nodes.items():
            agent_name = getattr(node.agent, "agent_name", "Unknown")
            role = " (Entry)" if node_id in workflow.entry_points else ""
            role += " (End)" if node_id in workflow.end_points else ""
            lines.append(f"  • [{node_id}] {agent_name}{role}")

        lines.append("")

        lines.append("🔗 Connections:")
        for edge in workflow.edges:
            meta = f" ({edge.metadata})" if edge.metadata else ""
            lines.append(f"  • {edge.source} → {edge.target}{meta}")

        result = "\n".join(lines)
        print(result)
        return result

    except Exception as e:
        logger.exception(f"Error in visualize_workflow_simple: {e}")
        raise e
