import subprocess
import sys
import uuid
import threading
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from graphviz import Digraph
from loguru import logger

# Airflow imports
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:
    logger.error(
        "Airflow is not installed. Please install it using 'pip install apache-airflow'."
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "apache-airflow"]
    )
    from airflow import DAG
    from airflow.operators.python import PythonOperator

# Import the real Agent from swarms.
from swarms.structs.conversation import Conversation


class NodeType(Enum):
    AGENT = "agent"
    CALLABLE = "callable"
    TOOL = "tool"


def dag_id():
    return uuid.uuid4().hex


@dataclass
class Node:
    """Represents a node in the DAG"""

    id: str
    type: NodeType
    component: Any  # Agent, Callable, or Tool
    query: Optional[str] = None
    args: Optional[List[Any]] = None
    kwargs: Optional[Dict[str, Any]] = None
    concurrent: bool = False


# ======= Airflow DAG Swarm Class =========
class AirflowDAGSwarm:
    """
    A simplified and more intuitive DAG-based swarm for orchestrating agents, callables, and tools.
    Provides an easy-to-use API for building agent pipelines with support for concurrent execution.
    """

    def __init__(
        self,
        dag_id: str = dag_id(),
        description: str = "A DAG Swarm for Airflow",
        name: str = "Airflow DAG Swarm",
        schedule_interval: Union[timedelta, str] = timedelta(days=1),
        start_date: datetime = datetime(2025, 2, 14),
        default_args: Optional[Dict[str, Any]] = None,
        initial_message: Optional[str] = None,
        max_workers: int = 5,
    ):
        """Initialize the AirflowDAGSwarm with improved configuration."""
        self.dag_id = dag_id
        self.name = name
        self.description = description
        self.max_workers = max_workers

        self.default_args = default_args or {
            "owner": "airflow",
            "depends_on_past": False,
            "email_on_failure": False,
            "email_on_retry": False,
            "retries": 1,
            "retry_delay": timedelta(minutes=5),
        }

        self.dag = DAG(
            dag_id=dag_id,
            default_args=self.default_args,
            schedule_interval=schedule_interval,
            start_date=start_date,
            catchup=False,
        )

        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Set[str]] = (
            {}
        )  # node_id -> set of child node_ids

        # Initialize conversation
        self.conversation = Conversation()
        if initial_message:
            self.conversation.add("user", initial_message)

        self.lock = threading.Lock()

    def add_user_message(self, message: str) -> None:
        """Add a user message to the conversation."""
        with self.lock:
            self.conversation.add("user", message)
            logger.info(f"Added user message: {message}")

    def get_conversation_history(self) -> str:
        """Get the conversation history as JSON."""
        return self.conversation.to_json()

    def add_node(
        self,
        node_id: str,
        component: Any,
        node_type: NodeType,
        query: Optional[str] = None,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        concurrent: bool = False,
    ) -> str:
        """
        Add a node to the DAG with improved type checking and validation.

        Args:
            node_id: Unique identifier for the node
            component: Agent, callable, or tool to execute
            node_type: Type of the node (AGENT, CALLABLE, or TOOL)
            query: Query string for agents
            args: Positional arguments for callables/tools
            kwargs: Keyword arguments for callables/tools
            concurrent: Whether to execute this node concurrently

        Returns:
            node_id: The ID of the created node
        """
        if node_id in self.nodes:
            raise ValueError(f"Node with ID {node_id} already exists")

        if node_type == NodeType.AGENT and not hasattr(
            component, "run"
        ):
            raise ValueError("Agent must have a 'run' method")
        elif node_type in (
            NodeType.CALLABLE,
            NodeType.TOOL,
        ) and not callable(component):
            raise ValueError(f"{node_type.value} must be callable")

        node = Node(
            id=node_id,
            type=node_type,
            component=component,
            query=query,
            args=args or [],
            kwargs=kwargs or {},
            concurrent=concurrent,
        )

        self.nodes[node_id] = node
        self.edges[node_id] = set()
        logger.info(f"Added {node_type.value} node: {node_id}")
        return node_id

    def add_edge(self, from_node: str, to_node: str) -> None:
        """
        Add a directed edge between two nodes in the DAG.

        Args:
            from_node: ID of the source node
            to_node: ID of the target node
        """
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError("Both nodes must exist in the DAG")

        self.edges[from_node].add(to_node)
        logger.info(f"Added edge: {from_node} -> {to_node}")

    def _execute_node(self, node: Node) -> str:
        """Execute a single node and return its output."""
        try:
            if node.type == NodeType.AGENT:
                query = (
                    node.query
                    or self.conversation.get_last_message_as_string()
                    or "Default query"
                )
                logger.info(
                    f"Executing agent node {node.id} with query: {query}"
                )
                return node.component.run(query)

            elif node.type in (NodeType.CALLABLE, NodeType.TOOL):
                logger.info(
                    f"Executing {node.type.value} node {node.id}"
                )
                return node.component(
                    *node.args,
                    conversation=self.conversation,
                    **node.kwargs,
                )
        except Exception as e:
            logger.exception(f"Error executing node {node.id}: {e}")
            return f"Error in node {node.id}: {str(e)}"

    def _get_root_nodes(self) -> List[str]:
        """Get nodes with no incoming edges."""
        all_nodes = set(self.nodes.keys())
        nodes_with_incoming = {
            node for edges in self.edges.values() for node in edges
        }
        return list(all_nodes - nodes_with_incoming)

    def run(self, **context: Any) -> str:
        """
        Execute the DAG with improved concurrency handling and error recovery.

        Returns:
            The final conversation state as a JSON string
        """
        logger.info("Starting swarm execution")

        # Track completed nodes and their results
        completed: Dict[str, str] = {}

        def can_execute_node(node_id: str) -> bool:
            """Check if all dependencies of a node are completed."""
            return all(
                dep in completed
                for dep_set in self.edges.values()
                for dep in dep_set
                if node_id in dep_set
            )

        with ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Initialize futures dict for concurrent root nodes
            futures_dict = {
                executor.submit(
                    self._execute_node, self.nodes[node_id]
                ): node_id
                for node_id in self._get_root_nodes()
                if self.nodes[node_id].concurrent
            }

            # Execute nodes that shouldn't run concurrently
            for node_id in self._get_root_nodes():
                if not self.nodes[node_id].concurrent:
                    output = self._execute_node(self.nodes[node_id])
                    with self.lock:
                        completed[node_id] = output
                        self.conversation.add("assistant", output)

            # Process remaining nodes
            while futures_dict:
                done, _ = wait(
                    futures_dict.keys(), return_when=FIRST_COMPLETED
                )

                for future in done:
                    node_id = futures_dict.pop(future)
                    try:
                        output = future.result()
                        with self.lock:
                            completed[node_id] = output
                            self.conversation.add("assistant", output)
                    except Exception as e:
                        logger.exception(
                            f"Error in node {node_id}: {e}"
                        )
                        completed[node_id] = f"Error: {str(e)}"

                    # Add new nodes that are ready to execute
                    new_nodes = [
                        node_id
                        for node_id in self.nodes
                        if node_id not in completed
                        and can_execute_node(node_id)
                    ]

                    for node_id in new_nodes:
                        if self.nodes[node_id].concurrent:
                            future = executor.submit(
                                self._execute_node,
                                self.nodes[node_id],
                            )
                            futures_dict[future] = node_id
                        else:
                            output = self._execute_node(
                                self.nodes[node_id]
                            )
                            with self.lock:
                                completed[node_id] = output
                                self.conversation.add(
                                    "assistant", output
                                )

        return self.conversation.to_json()

    def visualize(
        self, filename: str = "dag_visualization", view: bool = True
    ) -> Digraph:
        """
        Generate a visualization of the DAG structure.

        Args:
            filename: Output filename for the visualization
            view: Whether to open the visualization

        Returns:
            Graphviz Digraph object
        """
        dot = Digraph(comment=f"DAG Visualization: {self.name}")

        # Add nodes
        for node_id, node in self.nodes.items():
            label = f"{node_id}\n({node.type.value})"
            shape = "box" if node.concurrent else "ellipse"
            dot.node(node_id, label, shape=shape)

        # Add edges
        for from_node, to_nodes in self.edges.items():
            for to_node in to_nodes:
                dot.edge(from_node, to_node)

        dot.render(filename, view=view, format="pdf")
        return dot

    def create_dag(self) -> DAG:
        """
        Create an Airflow DAG with a single PythonOperator that executes the entire swarm.
        In a production environment, you might break the components into multiple tasks.

        :return: The configured Airflow DAG.
        """
        logger.info("Creating Airflow DAG for swarm execution.")
        PythonOperator(
            task_id="run",
            python_callable=self.run,
            op_kwargs={
                "concurrent": False
            },  # Change to True for concurrent execution.
            dag=self.dag,
        )
        return self.dag


# # ======= Example Usage =========
# if __name__ == "__main__":
#     # Configure logger to output to console.
#     logger.remove()
#     logger.add(lambda msg: print(msg, end=""), level="DEBUG")

#     # Create the DAG swarm with an initial message
#     swarm = AirflowDAGSwarm(
#         dag_id="swarm_conversation_dag",
#         initial_message="Hello, how can I help you with financial planning?",
#     )

#     # Create a real financial agent using the swarms package.
#     financial_agent = Agent(
#         agent_name="Financial-Analysis-Agent",
#         system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
#         model_name="gpt-4o-mini",
#         max_loops=1,
#     )

#     # Add the real agent with a specific query.
#     swarm.add_node(
#         "financial_advisor",
#         financial_agent,
#         NodeType.AGENT,
#         query="How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria",
#         concurrent=True,
#     )

#     # Add a callable component.
#     def extra_processing(x: int, conversation: Conversation) -> str:
#         return f"Extra processing output with data {x} and conversation length {len(conversation.messages)}"

#     swarm.add_node(
#         "extra_processing",
#         extra_processing,
#         NodeType.CALLABLE,
#         args=[42],
#         concurrent=True,
#     )

#     # Add a tool component (for example, a tool to create a conversation graph).
#     def create_conversation_graph(conversation: Conversation) -> str:
#         # In this tool, we generate the graph and return a confirmation message.
#         swarm.visualize(
#             filename="swarm_conversation_tool_graph", view=False
#         )
#         return "Graph created."

#     swarm.add_node(
#         "conversation_graph",
#         create_conversation_graph,
#         NodeType.TOOL,
#         concurrent=False,
#     )

#     # Add edges to create the pipeline
#     swarm.add_edge("financial_advisor", "extra_processing")
#     swarm.add_edge("extra_processing", "conversation_graph")

#     # Execute the swarm
#     final_state = swarm.run()
#     logger.info(f"Final conversation: {final_state}")

#     # Visualize the DAG
#     print(
#         swarm.visualize(
#             filename="swarm_conversation_final", view=False
#         )
#     )

#     # Create the Airflow DAG.
#     dag = swarm.create_dag()
