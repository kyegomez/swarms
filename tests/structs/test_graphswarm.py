from loguru import logger
from swarms.structs.agent import Agent
from swarms.structs.graph_swarm import GraphSwarm

# Configure logger
logger.add(
    "market_analysis_tests.log",
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    level="DEBUG",
)


def create_market_agents():
    """Create and return the standard set of market analysis agents."""
    logger.info("Creating market analysis agents...")
    try:
        data_collector = Agent(
            agent_name="Market-Data-Collector",
            model_name="openai/gpt-4o",
            max_loops=1,
            streaming_on=True,
        )
        logger.debug("Created Data Collector agent")

        trend_analyzer = Agent(
            agent_name="Market-Trend-Analyzer",
            model_name="openai/gpt-4o",
            max_loops=1,
            streaming_on=True,
        )
        logger.debug("Created Trend Analyzer agent")

        report_generator = Agent(
            agent_name="Investment-Report-Generator",
            model_name="openai/gpt-4o",
            max_loops=1,
            streaming_on=True,
        )
        logger.debug("Created Report Generator agent")

        return data_collector, trend_analyzer, report_generator
    except Exception as e:
        logger.error(f"Failed to create market agents: {str(e)}")
        raise


def validate_agent_output(output, agent_name):
    """Validate the output of an agent and provide detailed error information."""
    if output.error:
        logger.error(f"{agent_name} error: {output.error}")
        return (
            False,
            f"{agent_name} failed with error: {output.error}",
        )

    if output.output is None:
        logger.error(f"{agent_name} returned None output")
        return False, f"{agent_name} output is None"

    if not isinstance(output.output, str):
        logger.error(
            f"{agent_name} returned non-string output: {type(output.output)}"
        )
        return (
            False,
            f"{agent_name} output type is {type(output.output)}, expected str",
        )

    if output.execution_time <= 0:
        logger.error(
            f"{agent_name} has invalid execution time: {output.execution_time}"
        )
        return False, f"{agent_name} has invalid execution time"

    return True, None


def test_swarm_initialization():
    """Test the initialization of the market analysis swarm."""
    logger.info("\nTesting market analysis swarm initialization...")

    try:
        data_collector, trend_analyzer, report_generator = (
            create_market_agents()
        )

        swarm = GraphSwarm(
            agents=[
                (data_collector, []),
                (trend_analyzer, ["Market-Data-Collector"]),
                (report_generator, ["Market-Trend-Analyzer"]),
            ],
            swarm_name="Market Analysis Intelligence Network",
        )

        # Detailed swarm validation
        logger.debug("Validating swarm configuration...")

        assert (
            swarm.swarm_name == "Market Analysis Intelligence Network"
        ), f"Expected swarm name 'Market Analysis Intelligence Network', got {swarm.swarm_name}"

        assert (
            len(swarm.agents) == 3
        ), f"Expected 3 agents, got {len(swarm.agents)}"

        for agent_name in [
            "Market-Data-Collector",
            "Market-Trend-Analyzer",
            "Investment-Report-Generator",
        ]:
            assert (
                agent_name in swarm.agents
            ), f"Missing agent: {agent_name}"

        # Validate dependencies
        assert (
            len(swarm.dependencies["Market-Data-Collector"]) == 0
        ), "Data collector should have no dependencies"

        assert (
            "Market-Data-Collector"
            in swarm.dependencies["Market-Trend-Analyzer"]
        ), "Trend analyzer should depend on data collector"

        assert (
            "Market-Trend-Analyzer"
            in swarm.dependencies["Investment-Report-Generator"]
        ), "Report generator should depend on trend analyzer"

        logger.info("✓ Swarm initialization test passed")
        return swarm
    except Exception as e:
        logger.error(f"Swarm initialization test failed: {str(e)}")
        raise


def test_market_data_collection():
    """Test the market data collection agent in isolation."""
    logger.info("\nTesting market data collection...")

    try:
        data_collector = Agent(
            agent_name="Market-Data-Collector",
            model_name="openai/gpt-4o",
            max_loops=1,
            streaming_on=True,
        )

        swarm = GraphSwarm(
            agents=[(data_collector, [])],
            swarm_name="Data Collection Test",
        )

        test_prompt = """
        Collect current market data for the following tech stocks:
        1. Apple (AAPL)
        2. Microsoft (MSFT)
        3. Google (GOOGL)
        
        Include:
        - Current price
        - Market cap
        - P/E ratio
        - 52-week range
        """

        result = swarm.run(test_prompt)
        logger.debug(f"Data collector result: {result}")

        assert (
            result.success
        ), f"Data collection failed: {result.error if result.error else 'Unknown error'}"

        collector_output = result.outputs["Market-Data-Collector"]
        is_valid, error_msg = validate_agent_output(
            collector_output, "Market-Data-Collector"
        )
        assert is_valid, error_msg

        logger.info("✓ Market data collection test passed")
        return result
    except Exception as e:
        logger.error(f"Market data collection test failed: {str(e)}")
        raise


def test_trend_analysis():
    """Test the trend analysis agent with proper error handling."""
    logger.info("\nTesting market trend analysis...")

    try:
        data_collector, trend_analyzer, _ = create_market_agents()

        swarm = GraphSwarm(
            agents=[
                (data_collector, []),
                (trend_analyzer, ["Market-Data-Collector"]),
            ],
            swarm_name="Trend Analysis Test",
        )

        test_prompt = """
        Analyze market trends for tech stocks with focus on:
        - Price movement patterns
        - Volume analysis
        - Technical indicators
        - Market sentiment
        """

        result = swarm.run(test_prompt)
        logger.debug(f"Trend analysis result: {result}")

        # Validate data collector output first
        collector_output = result.outputs.get("Market-Data-Collector")
        assert (
            collector_output is not None
        ), "Missing data collector output"
        is_valid, error_msg = validate_agent_output(
            collector_output, "Market-Data-Collector"
        )
        assert is_valid, error_msg

        # Then validate trend analyzer output
        analyzer_output = result.outputs.get("Market-Trend-Analyzer")
        assert (
            analyzer_output is not None
        ), "Missing trend analyzer output"
        is_valid, error_msg = validate_agent_output(
            analyzer_output, "Market-Trend-Analyzer"
        )
        assert is_valid, error_msg

        logger.info("✓ Market trend analysis test passed")
        return result
    except Exception as e:
        logger.error(f"Market trend analysis test failed: {str(e)}")
        logger.debug(f"Full error details: {str(e)}", exc_info=True)
        raise


def test_error_handling():
    """Test error handling with deliberately invalid configurations."""
    logger.info("\nTesting error handling...")

    try:
        # Test with invalid model name
        bad_agent = Agent(
            agent_name="Bad-Agent",
            model_name="invalid-model",
            max_loops=1,
            streaming_on=True,
        )

        swarm = GraphSwarm(
            agents=[(bad_agent, [])],
            swarm_name="Error Handling Test",
        )

        result = swarm.run("Test invalid model")

        assert (
            not result.success
        ), "Execution with invalid model should fail"
        assert (
            result.error is not None
        ), "Expected error message, got None"
        assert (
            "Bad-Agent" in result.outputs
        ), "Missing output for failed agent"
        assert (
            result.outputs["Bad-Agent"].error is not None
        ), "Missing agent error message"

        logger.info("✓ Error handling test passed")
        return result
    except Exception as e:
        logger.error(f"Error handling test failed: {str(e)}")
        raise


def run_all_tests():
    """Run all market analysis swarm tests with proper error handling."""
    logger.info("Starting Market Analysis GraphSwarm test suite...")

    test_results = {
        "initialization": False,
        "data_collection": False,
        "trend_analysis": False,
        "error_handling": False,
    }

    try:
        test_swarm_initialization()
        test_results["initialization"] = True
        logger.debug("Initialization test completed")

        test_market_data_collection()
        test_results["data_collection"] = True
        logger.debug("Data collection test completed")

        test_trend_analysis()
        test_results["trend_analysis"] = True
        logger.debug("Trend analysis test completed")

        test_error_handling()
        test_results["error_handling"] = True
        logger.debug("Error handling test completed")

        logger.info(
            "\nAll market analysis tests completed successfully! ✓"
        )

    except Exception as e:
        logger.error("Test suite failed!")
        logger.error(f"Error details: {str(e)}")
        logger.debug("Test results summary:")
        for test_name, passed in test_results.items():
            logger.debug(f"{test_name}: {'✓' if passed else '✗'}")
        raise
    finally:
        logger.debug("Test suite finished executing")


if __name__ == "__main__":
    run_all_tests()
