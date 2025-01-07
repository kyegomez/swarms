from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.graph_swarm import GraphSwarm

if __name__ == "__main__":
    try:
        # Create agents
        data_collector = Agent(
            agent_name="Market-Data-Collector",
            model_name="openai/gpt-4o",
            max_loops=1,
            streaming_on=True,
        )

        trend_analyzer = Agent(
            agent_name="Market-Trend-Analyzer",
            model_name="openai/gpt-4o",
            max_loops=1,
            streaming_on=True,
        )

        report_generator = Agent(
            agent_name="Investment-Report-Generator",
            model_name="openai/gpt-4o",
            max_loops=1,
            streaming_on=True,
        )

        # Create swarm
        swarm = GraphSwarm(
            agents=[
                (data_collector, []),
                (trend_analyzer, ["Market-Data-Collector"]),
                (report_generator, ["Market-Trend-Analyzer"]),
            ],
            swarm_name="Market Analysis Intelligence Network",
        )

        # Run the swarm
        result = swarm.run(
            "Analyze current market trends for tech stocks and provide investment recommendations"
        )

        # Print results
        print(f"Execution success: {result.success}")
        print(f"Total time: {result.execution_time:.2f} seconds")

        for agent_name, output in result.outputs.items():
            print(f"\nAgent: {agent_name}")
            print(f"Output: {output.output}")
            if output.error:
                print(f"Error: {output.error}")
    except Exception as error:
        logger.error(error)
        raise error
