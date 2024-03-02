from dataclasses import asdict
from typing import List

import networkx as nx
import redis
from redis.commands.graph import Graph, Node

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import AbstractSwarm


class SwarmRelationship:
    JOINED = "joined"


class RedisSwarmRegistry(AbstractSwarm):
    """
    Initialize the SwarmRedisRegistry object.

    Args:
        host (str): The hostname or IP address of the Redis server. Default is "localhost".
        port (int): The port number of the Redis server. Default is 6379.
        db: The Redis database number. Default is 0.
        graph_name (str): The name of the RedisGraph graph. Default is "swarm_registry".
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db=0,
        graph_name: str = "swarm_registry",
    ):
        self.redis = redis.StrictRedis(
            host=host, port=port, db=db, decode_responses=True
        )
        self.redis_graph = Graph(self.redis, graph_name)
        self.graph = nx.DiGraph()

    def _entity_to_node(self, entity: Agent | Agent) -> Node:
        """
        Converts an Agent or Swarm object to a Node object.

        Args:
            entity (Agent | Agent): The Agent or Swarm object to convert.

        Returns:
            Node: The converted Node object.
        """
        return Node(
            node_id=entity.id,
            alias=entity.agent_name,
            label=entity.agent_description,
            properties=asdict(entity),
        )

    def _add_node(self, node: Agent | Agent):
        """
        Adds a node to the graph.

        Args:
            node (Agent | Agent): The Agent or Swarm node to add.
        """
        self.graph.add_node(node.id)
        if isinstance(node, Agent):
            self.add_swarm_entry(node)
        elif isinstance(node, Agent):
            self.add_agent_entry(node)

    def _add_edge(self, from_node: Node, to_node: Node, relationship):
        """
        Adds an edge between two nodes in the graph.

        Args:
            from_node (Node): The source node of the edge.
            to_node (Node): The target node of the edge.
            relationship: The relationship type between the nodes.
        """
        match_query = (
            f"MATCH (a:{from_node.label}),(b:{to_node.label}) WHERE"
            f" a.id = {from_node.id} AND b.id = {to_node.id}"
        )

        query = f"""
        {match_query}
        CREATE (a)-[r:joined]->(b) RETURN r
        """.replace(
            "\n", ""
        )

        self.redis_graph.query(query)

    def add_swarm_entry(self, swarm: Agent):
        """
        Adds a swarm entry to the graph.

        Args:
            swarm (Agent): The swarm object to add.
        """
        node = self._entity_to_node(swarm)
        self._persist_node(node)

    def add_agent_entry(self, agent: Agent):
        """
        Adds an agent entry to the graph.

        Args:
            agent (Agent): The agent object to add.
        """
        node = self._entity_to_node(agent)
        self._persist_node(node)

    def join_swarm(
        self,
        from_entity: Agent | Agent,
        to_entity: Agent,
    ):
        """
        Adds an edge between two nodes in the graph.

        Args:
            from_entity (Agent | Agent): The source entity of the edge.
            to_entity (Agent): The target entity of the edge.

        Returns:
            Any: The result of adding the edge.
        """
        from_node = self._entity_to_node(from_entity)
        to_node = self._entity_to_node(to_entity)

        return self._add_edge(
            from_node, to_node, SwarmRelationship.JOINED
        )

    def _persist_node(self, node: Node):
        """
        Persists a node in the graph.

        Args:
            node (Node): The node to persist.
        """
        query = f"CREATE {node}"
        self.redis_graph.query(query)

    def retrieve_swarm_information(self, swarm_id: int) -> Agent:
        """
        Retrieves swarm information from the registry.

        Args:
            swarm_id (int): The ID of the swarm to retrieve.

        Returns:
            Agent: The retrieved swarm information as an Agent object.
        """
        swarm_key = f"swarm:{swarm_id}"
        swarm_data = self.redis.hgetall(swarm_key)
        if swarm_data:
            # Parse the swarm_data and return an instance of AgentBase
            # You can use the retrieved data to populate the AgentBase attributes

            return Agent(**swarm_data)
        return None

    def retrieve_joined_agents(self) -> List[Agent]:
        """
        Retrieves a list of joined agents from the registry.

        Returns:
            List[Agent]: The retrieved joined agents as a list of Agent objects.
        """
        agent_data = self.redis_graph.query(
            "MATCH (a:agent)-[:joined]->(b:manager) RETURN a"
        )
        if agent_data:
            # Parse the agent_data and return an instance of AgentBase
            # You can use the retrieved data to populate the AgentBase attributes
            return [Agent(**agent_data) for agent_data in agent_data]
        return None
