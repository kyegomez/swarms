import os
import json
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from loguru import logger
from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import FINANCIAL_AGENT_SYS_PROMPT

# Configure Loguru logger
logger.remove()  # Remove default handler
logger.add(
    "financial_agent_tests_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

class FinancialAgentTestSuite:
    """
    Production-grade test suite for Financial Analysis Agent.
    
    This test suite provides comprehensive testing of the Financial Analysis Agent's
    functionality, including initialization, configuration, and response validation.
    
    Attributes:
        test_data_path (str): Path to store test data and outputs
        agent_config (Dict[str, Any]): Default configuration for test agents
    """
    
    def __init__(self, test_data_path: str = "./test_data"):
        """
        Initialize the test suite with configuration and setup.
        
        Args:
            test_data_path (str): Directory to store test data and outputs
        """
        self.test_data_path = test_data_path
        self.agent_config = {
            "agent_name": "Test-Financial-Analysis-Agent",
            "system_prompt": FINANCIAL_AGENT_SYS_PROMPT,
            "model_name": "gpt-4o-mini",
            "max_loops": 1,
            "autosave": True,
            "dashboard": False,
            "verbose": True,
            "dynamic_temperature_enabled": True,
            "saved_state_path": "test_finance_agent.json",
            "user_name": "test_user",
            "retry_attempts": 1,
            "context_length": 200000,
            "return_step_meta": False,
            "output_type": "string",
            "streaming_on": False,
        }
        self._setup_test_environment()

    def _setup_test_environment(self) -> None:
        """Create necessary directories and files for testing."""
        try:
            os.makedirs(self.test_data_path, exist_ok=True)
            logger.info(f"Test environment setup completed at {self.test_data_path}")
        except Exception as e:
            logger.error(f"Failed to setup test environment: {str(e)}")
            raise

    async def _create_test_agent(self, config_override: Optional[Dict[str, Any]] = None) -> Agent:
        """
        Create a test agent with specified or default configuration.
        
        Args:
            config_override (Optional[Dict[str, Any]]): Override default config values
            
        Returns:
            Agent: Configured test agent instance
        """
        try:
            test_config = self.agent_config.copy()
            if config_override:
                test_config.update(config_override)
            
            agent = Agent(**test_config)
            logger.debug(f"Created test agent with config: {test_config}")
            return agent
        except Exception as e:
            logger.error(f"Failed to create test agent: {str(e)}")
            raise

    async def test_agent_initialization(self) -> Tuple[bool, str]:
        """
        Test agent initialization with various configurations.
        
        Returns:
            Tuple[bool, str]: Success status and result message
        """
        try:
            logger.info("Starting agent initialization test")
            
            # Test default initialization
            agent = await self._create_test_agent()
            assert isinstance(agent, Agent), "Agent initialization failed"
            
            # Test with modified configuration
            custom_config = {"max_loops": 2, "context_length": 150000}
            agent_custom = await self._create_test_agent(custom_config)
            assert agent_custom.max_loops == 2, "Custom configuration not applied"
            
            logger.info("Agent initialization test passed")
            return True, "Agent initialization successful"
        except Exception as e:
            logger.error(f"Agent initialization test failed: {str(e)}")
            return False, f"Agent initialization failed: {str(e)}"

    async def test_agent_response(self) -> Tuple[bool, str]:
        """
        Test agent's response functionality with various queries.
        
        Returns:
            Tuple[bool, str]: Success status and result message
        """
        try:
            logger.info("Starting agent response test")
            agent = await self._create_test_agent()
            
            test_queries = [
                "How can I establish a ROTH IRA?",
                "What are the tax implications of stock trading?",
                "Explain mutual fund investment strategies"
            ]
            
            for query in test_queries:
                response = agent.run(query)
                assert isinstance(response, str), "Response type mismatch"
                assert len(response) > 0, "Empty response received"
                logger.debug(f"Query: {query[:50]}... | Response length: {len(response)}")
            
            logger.info("Agent response test passed")
            return True, "Agent response test successful"
        except Exception as e:
            logger.error(f"Agent response test failed: {str(e)}")
            return False, f"Agent response test failed: {str(e)}"

    async def test_agent_persistence(self) -> Tuple[bool, str]:
        """
        Test agent's state persistence and recovery.
        
        Returns:
            Tuple[bool, str]: Success status and result message
        """
        try:
            logger.info("Starting agent persistence test")
            
            # Test state saving
            save_path = os.path.join(self.test_data_path, "test_state.json")
            agent = await self._create_test_agent({"saved_state_path": save_path})
            
            test_query = "What is a 401k plan?"
            agent.run(test_query)
            
            assert os.path.exists(save_path), "State file not created"
            
            # Verify state content
            with open(save_path, 'r') as f:
                saved_state = json.load(f)
            assert "agent_name" in saved_state, "Invalid state file content"
            
            logger.info("Agent persistence test passed")
            return True, "Agent persistence test successful"
        except Exception as e:
            logger.error(f"Agent persistence test failed: {str(e)}")
            return False, f"Agent persistence test failed: {str(e)}"

    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all test cases and generate a comprehensive report.
        
        Returns:
            Dict[str, Any]: Test results and statistics
        """
        start_time = datetime.now()
        results = []
        
        test_cases = [
            ("Agent Initialization", self.test_agent_initialization),
            ("Agent Response", self.test_agent_response),
            ("Agent Persistence", self.test_agent_persistence)
        ]
        
        for test_name, test_func in test_cases:
            try:
                success, message = await test_func()
                results.append({
                    "test_name": test_name,
                    "success": success,
                    "message": message,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Test {test_name} failed with unexpected error: {str(e)}")
                results.append({
                    "test_name": test_name,
                    "success": False,
                    "message": f"Unexpected error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Generate report
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["success"])
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": f"{(passed_tests/total_tests)*100:.2f}%",
                "duration_seconds": duration
            },
            "test_results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save report
        report_path = os.path.join(self.test_data_path, f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test suite completed. Report saved to {report_path}")
        return report

async def main():
    """Main entry point for running the test suite."""
    logger.info("Starting Financial Agent Test Suite")
    test_suite = FinancialAgentTestSuite()
    report = await test_suite.run_all_tests()
    
    # Print summary to console
    print("\n" + "="*50)
    print("Financial Agent Test Suite Results")
    print("="*50)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed Tests: {report['summary']['passed_tests']}")
    print(f"Failed Tests: {report['summary']['failed_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']}")
    print(f"Duration: {report['summary']['duration_seconds']:.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())