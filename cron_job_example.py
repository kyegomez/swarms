from swarms import Agent, CronJob
from loguru import logger


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = Agent(
        agent_name="Quantitative-Trading-Agent",
        agent_description="Advanced quantitative trading and algorithmic analysis agent",
        system_prompt="""You are an expert quantitative trading agent with deep expertise in:
        - Algorithmic trading strategies and implementation
        - Statistical arbitrage and market making
        - Risk management and portfolio optimization
        - High-frequency trading systems
        - Market microstructure analysis
        - Quantitative research methodologies
        - Financial mathematics and stochastic processes
        - Machine learning applications in trading

        Your core responsibilities include:
        1. Developing and backtesting trading strategies
        2. Analyzing market data and identifying alpha opportunities
        3. Implementing risk management frameworks
        4. Optimizing portfolio allocations
        5. Conducting quantitative research
        6. Monitoring market microstructure
        7. Evaluating trading system performance

        You maintain strict adherence to:
        - Mathematical rigor in all analyses
        - Statistical significance in strategy development
        - Risk-adjusted return optimization
        - Market impact minimization
        - Regulatory compliance
        - Transaction cost analysis
        - Performance attribution

        You communicate in precise, technical terms while maintaining clarity for stakeholders.""",
        max_loops=1,
        model_name="gpt-4.1",
        dynamic_temperature_enabled=True,
        output_type="str-all-except-first",
        streaming_on=True,
        print_on=True,
        telemetry_enable=False,
    )

    # Example 1: Basic usage with just a task
    logger.info("Starting example cron job")
    cron_job = CronJob(agent=agent, interval="10seconds")
    cron_job.run(
        task="What are the best top 3 etfs for gold coverage?"
    )
