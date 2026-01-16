"""Simple benchmark for HierarchicalSwarm order execution.

Creates fake agents whose `run` method sleeps for a short duration to
simulate work. Compares sequential vs parallel execution using
`HierarchicalSwarm.execute_orders`.

Run:
    python examples/hierarchical_benchmark.py

"""
import time
import random

from concurrent.futures import ThreadPoolExecutor, as_completed


class FakeAgent:
    def __init__(self, name: str, latency: float = 0.5):
        self.agent_name = name
        self.latency = latency

    def run(self, task: str, *args, **kwargs):
        # Simulate work
        time.sleep(self.latency)
        return f"{self.agent_name} done ({task})"


def build_agents_and_orders(num_agents: int, latency_mean: float = 0.5):
    agents = [FakeAgent(f"agent_{i}", latency=random.uniform(latency_mean * 0.8, latency_mean * 1.2)) for i in range(num_agents)]
    orders = [(a.agent_name, f"task_{i}") for i, a in enumerate(agents)]
    agents_map = {a.agent_name: a for a in agents}
    return agents_map, orders


def sequential_execute_orders(orders, agents_map):
    outputs = []
    for agent_name, task in orders:
        agent = agents_map[agent_name]
        out = agent.run(task)
        outputs.append(out)
    return outputs


def parallel_execute_orders(orders, agents_map, max_workers=None):
    outputs = [None] * len(orders)
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        future_to_idx = {}
        for idx, (agent_name, task) in enumerate(orders):
            agent = agents_map[agent_name]
            fut = exc.submit(agent.run, task)
            future_to_idx[fut] = idx

        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                outputs[idx] = fut.result()
            except Exception as e:
                outputs[idx] = f"[ERROR] {e}"

    return outputs


def run_benchmark(num_agents=8, latency_mean=0.5, max_workers=None):
    agents_map, orders = build_agents_and_orders(num_agents, latency_mean=latency_mean)

    # Sequential
    t0 = time.time()
    out_seq = sequential_execute_orders(orders, agents_map)
    seq_time = time.time() - t0

    # Parallel
    t0 = time.time()
    out_par = parallel_execute_orders(orders, agents_map, max_workers=max_workers)
    par_time = time.time() - t0

    print(f"Agents: {num_agents}, mean_latency: {latency_mean:.2f}s")
    print(f"Sequential time: {seq_time:.2f}s")
    print(f"Parallel   time: {par_time:.2f}s")
    print("Sequential outputs sample:", out_seq[:2])
    print("Parallel outputs sample:  ", out_par[:2])


if __name__ == "__main__":
    run_benchmark(num_agents=12, latency_mean=0.4)
