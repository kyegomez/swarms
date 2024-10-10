import heapq
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional, Callable
from swarms import Agent
from swarms.utils.loguru_logger import logger

class AStarSwarm:
    def __init__(
        self,
        agents: List[Agent],
        communication_costs: Optional[Dict[Tuple[str, str], float]] = None,
        heuristic: Optional[Callable[[Agent, str], float]] = None,
    ):
        self.agents = agents
        self.communication_costs = communication_costs or {}  # Default to no cost
        self.heuristic = heuristic or self.default_heuristic
        self.graph = self._build_communication_graph()

    def _build_communication_graph(self) -> nx.Graph:
        graph = nx.Graph()
        for agent in self.agents:
            graph.add_node(agent.agent_name)

        # Add edges with communication costs (if provided)
        for (agent1_name, agent2_name), cost in self.communication_costs.items():
            if agent1_name in graph.nodes and agent2_name in graph.nodes:
                graph.add_edge(agent1_name, agent2_name, weight=cost)

        return graph



    def a_star_search(self, start_agent: Agent, task: str, goal_agent: Optional[Agent]=None) -> Optional[List[Agent]]:
        """
        Performs A* search to find a path to the goal agent or all agents.

        """

        open_set = [(0, start_agent.agent_name)]
        came_from = {}
        g_score = {agent.agent_name: float('inf') for agent in self.agents}
        g_score[start_agent.agent_name] = 0
        f_score = {agent.agent_name: float('inf') for agent in self.agents}
        f_score[start_agent.agent_name] = self.heuristic(start_agent, task)

        while open_set:
            _, current_agent_name = heapq.heappop(open_set)


            if goal_agent and current_agent_name == goal_agent.agent_name: # Stop if specific goal agent is reached
                return self._reconstruct_path(came_from, current_agent_name)
            elif not goal_agent and len(came_from) == len(self.agents) -1: # Stop if all agents (except the starting one) are reached
                return self._reconstruct_path(came_from, current_agent_name)


            for neighbor_name in self.graph.neighbors(current_agent_name):
                weight = self.graph[current_agent_name][neighbor_name].get('weight', 1) # Default weight is 1
                tentative_g_score = g_score[current_agent_name] + weight

                if tentative_g_score < g_score[neighbor_name]:
                    came_from[neighbor_name] = current_agent_name
                    g_score[neighbor_name] = tentative_g_score

                    neighbor_agent = self.get_agent_by_name(neighbor_name)
                    if neighbor_agent:
                        f_score[neighbor_name] = tentative_g_score + self.heuristic(neighbor_agent, task)
                        if (f_score[neighbor_name], neighbor_name) not in open_set:
                            heapq.heappush(open_set, (f_score[neighbor_name], neighbor_name))

        return None  # No path found

    def _reconstruct_path(self, came_from: Dict[str, str], current: str) -> List[Agent]:
        path = [self.get_agent_by_name(current)]
        while current in came_from:
            current = came_from[current]
            path.insert(0, self.get_agent_by_name(current))  # Insert at beginning
        return path

    def get_agent_by_name(self, name:str) -> Optional[Agent]:
        for agent in self.agents:
            if agent.agent_name == name:
                return agent
        return None

    def default_heuristic(self, agent: Agent, task: str) -> float:
        return 0 # Default heuristic (equivalent to Dijkstra's algorithm)



    def run(self, task: str, start_agent_name: str, goal_agent_name:Optional[str] = None) -> List[Dict[str, Any]]:
        start_agent = self.get_agent_by_name(start_agent_name)
        goal_agent = self.get_agent_by_name(goal_agent_name) if goal_agent_name else None

        if not start_agent:
            logger.error(f"Start agent '{start_agent_name}' not found.")
            return []

        if goal_agent_name and not goal_agent:
            logger.error(f"Goal agent '{goal_agent_name}' not found.")
            return []

        agent_path = self.a_star_search(start_agent, task, goal_agent)

        results = []
        if agent_path:
            current_input = task
            for agent in agent_path:
                logger.info(f"Agent {agent.agent_name} processing task: {current_input}")
                try:
                    result = agent.run(current_input)
                    results.append({"agent": agent.agent_name, "task": current_input, "result": result})
                    current_input = str(result) # Pass output to the next agent
                except Exception as e:
                    logger.error(f"Agent {agent.agent_name} encountered an error: {e}")
                    results.append({"agent": agent.agent_name, "task": current_input, "result": f"Error: {e}"})
                    break # Stop processing if an agent fails
        else:
            logger.warning("No path found between agents.")


        return results


    def visualize(self):
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(10, 8))
        nx.draw(self.graph, pos, with_labels=True, node_color="lightblue", node_size=3000, font_weight="bold")
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels) # Display edge weights
        plt.title("Agent Communication Graph")
        plt.show()
