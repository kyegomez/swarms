import logging
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple
from swarms import Agent

# Setup basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class AgentDFS:
    """
    A DFS search class that uses a single Agent to generate and manually evaluate text states.
    """

    def __init__(
        self,
        agent: Agent,
        evaluator: Agent,
        initial_prompt: str,
        num_thoughts: int,
        max_steps: int,
        max_states: int,
        pruning_threshold: float,
    ):
        self.agent = agent
        self.initial_prompt = initial_prompt
        self.num_thoughts = num_thoughts
        self.max_steps = max_steps
        self.max_states = max_states
        self.pruning_threshold = pruning_threshold
        self.visited = {}
        self.graph = nx.DiGraph()

    def search(self) -> List[Tuple[str, float]]:
        stack = [(self.initial_prompt, 0.0)]
        self.graph.add_node(self.initial_prompt, score=0.0)
        results = []

        while stack and len(results) < self.max_steps:
            current_prompt, _ = stack.pop()
            logging.info(f"Generating from: {current_prompt}")

            # Use agent to generate a response
            out = self.agent.run(current_prompt)

            # Retrieve and split generated text into segments (assuming `agent.response` holds the text)
            generated_texts = self.split_into_thoughts(
                out, self.num_thoughts
            )

            for text, score in generated_texts:
                if score >= self.pruning_threshold:
                    stack.append((text, score))
                    results.append((text, score))
                    self.graph.add_node(text, score=score)
                    self.graph.add_edge(current_prompt, text)
                    logging.info(f"Added node: {text} with score: {score}")

            results.sort(key=lambda x: x[1], reverse=True)
            results = results[: self.max_states]

        logging.info("Search completed")
        return results

    def split_into_thoughts(
        self, text: str, num_thoughts: int
    ) -> List[Tuple[str, float]]:
        """Simulate the split of text into thoughts and assign random scores."""
        import random

        # Simple split based on punctuation or predefined length
        thoughts = text.split(".")[:num_thoughts]
        return [
            (thought.strip(), random.random())
            for thought in thoughts
            if thought.strip()
        ]

    def visualize(self):
        pos = nx.spring_layout(self.graph, seed=42)
        labels = {
            node: f"{node[:15]}...: {self.graph.nodes[node]['score']:.2f}"
            for node in self.graph.nodes()
        }
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            labels=labels,
            node_size=7000,
            node_color="skyblue",
            font_size=8,
            font_weight="bold",
            edge_color="gray",
        )
        plt.show()


# Example usage setup remains the same as before


# Example usage setup remains the same as before, simply instantiate two agents: one for generation and one for evaluation
# # Example usage
# if __name__ == "__main__":
#     load_dotenv()
#     api_key = os.environ.get("OPENAI_API_KEY")
#     llm = llama3Hosted(max_tokens=400)
#     agent = Agent(llm=llm, max_loops=1, autosave=True, dashboard=True)
#     dfs_agent = AgentDFS(
#         agent=agent,
#         initial_prompt="Explore the benefits of regular exercise.",
#         num_thoughts=5,
#         max_steps=20,
#         max_states=10,
#         pruning_threshold=0.3,
#     )
#     results = dfs_agent.search()
#     dfs_agent.visualize()
