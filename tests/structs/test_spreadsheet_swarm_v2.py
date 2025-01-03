import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from loguru import logger
import aiofiles
from swarms.structs.agent import Agent
from swarms.structs.spreadsheet_swarm import (
    SpreadSheetSwarm,
    AgentOutput,
    SwarmRunMetadata
)

# Configure Loguru logger
logger.remove()  # Remove default handler
logger.add(
    "spreadsheet_swarm_{time}.log",
    rotation="1 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    backtrace=True,
    diagnose=True
)

class SpreadSheetSwarmTestSuite:
    """
    Enhanced test suite for SpreadSheetSwarm functionality.
    
    Provides comprehensive testing of swarm initialization, CSV operations,
    task execution, and data persistence with detailed logging and error tracking.
    """
    
    def __init__(self, test_data_path: str = "./test_data"):
        """
        Initialize test suite with configuration.
        
        Args:
            test_data_path (str): Directory for test data and outputs
        """
        self.test_data_path = test_data_path
        self._setup_test_environment()
        
    def _setup_test_environment(self) -> None:
        """Setup required test directories and resources."""
        try:
            os.makedirs(self.test_data_path, exist_ok=True)
            logger.info(f"Test environment initialized at {self.test_data_path}")
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            raise

    async def create_test_csv(self) -> str:
        """
        Create a test CSV file with agent configurations.
        
        Returns:
            str: Path to created CSV file
        """
        try:
            csv_content = """agent_name,description,system_prompt,task
test_agent_1,Test Agent 1,System prompt 1,Task 1
test_agent_2,Test Agent 2,System prompt 2,Task 2
test_agent_3,Test Agent 3,System prompt 3,Task 3"""
            
            file_path = os.path.join(self.test_data_path, "test_agents.csv")
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(csv_content)
            
            logger.debug(f"Created test CSV at {file_path} with content:\n{csv_content}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to create test CSV: {e}")
            raise

    def create_test_agent(self, name: str, **kwargs) -> Agent:
        """
        Create a test agent with specified configuration.
        
        Args:
            name (str): Agent name
            **kwargs: Additional agent configuration
            
        Returns:
            Agent: Configured test agent
        """
        try:
            config = {
                "agent_name": name,
                "system_prompt": f"Test prompt for {name}",
                "model_name": "gpt-4o-mini",
                "max_loops": 1,
                "autosave": True,
                "verbose": True,
                **kwargs
            }
            agent = Agent(**config)
            logger.debug(f"Created test agent: {name}")
            return agent
        except Exception as e:
            logger.error(f"Failed to create agent {name}: {e}")
            raise

    async def test_swarm_initialization(self) -> Tuple[bool, str]:
        """
        Test swarm initialization with various configurations.
        
        Returns:
            Tuple[bool, str]: Success status and message
        """
        try:
            logger.info("Starting swarm initialization test")
            
            # Test basic initialization
            agents = [
                self.create_test_agent("agent1"),
                self.create_test_agent("agent2", max_loops=2)
            ]
            
            swarm = SpreadSheetSwarm(
                name="Test Swarm",
                description="Test Description",
                agents=agents,
                max_loops=2
            )
            
            # Verify configuration
            assert swarm.name == "Test Swarm"
            assert swarm.description == "Test Description"
            assert len(swarm.agents) == 2
            assert swarm.max_loops == 2
            
            # Test empty initialization
            empty_swarm = SpreadSheetSwarm()
            assert len(empty_swarm.agents) == 0
            
            logger.info("Swarm initialization test passed")
            return True, "Initialization successful"
        except Exception as e:
            logger.error(f"Swarm initialization test failed: {e}")
            return False, str(e)

    async def test_csv_operations(self) -> Tuple[bool, str]:
        """
        Test CSV loading and saving operations.
        
        Returns:
            Tuple[bool, str]: Success status and message
        """
        try:
            logger.info("Starting CSV operations test")
            
            # Test CSV loading
            csv_path = await self.create_test_csv()
            swarm = SpreadSheetSwarm(load_path=csv_path)
            await swarm._load_from_csv()
            
            assert len(swarm.agents) == 3
            assert len(swarm.agent_configs) == 3
            
            # Test CSV saving
            output_path = os.path.join(self.test_data_path, "test_output.csv")
            swarm.save_file_path = output_path
            swarm._track_output("test_agent_1", "Test task", "Test result")
            await swarm._save_to_csv()
            
            assert os.path.exists(output_path)
            
            # Cleanup
            os.remove(csv_path)
            os.remove(output_path)
            
            logger.info("CSV operations test passed")
            return True, "CSV operations successful"
        except Exception as e:
            logger.error(f"CSV operations test failed: {e}")
            return False, str(e)

    async def test_task_execution(self) -> Tuple[bool, str]:
        """
        Test task execution and output tracking.
        
        Returns:
            Tuple[bool, str]: Success status and message
        """
        try:
            logger.info("Starting task execution test")
            
            agents = [
                self.create_test_agent("agent1"),
                self.create_test_agent("agent2")
            ]
            swarm = SpreadSheetSwarm(agents=agents, max_loops=1)
            
            # Run test tasks
            test_tasks = ["Task 1", "Task 2"]
            for task in test_tasks:
                await swarm._run_tasks(task)
            
            # Verify execution
            assert swarm.metadata.tasks_completed == 4  # 2 agents × 2 tasks
            assert len(swarm.metadata.outputs) == 4
            
            # Test output tracking
            assert all(output.agent_name in ["agent1", "agent2"] 
                      for output in swarm.metadata.outputs)
            
            logger.info("Task execution test passed")
            return True, "Task execution successful"
        except Exception as e:
            logger.error(f"Task execution test failed: {e}")
            return False, str(e)

    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Execute all test cases and generate report.
        
        Returns:
            Dict[str, Any]: Comprehensive test results and metrics
        """
        start_time = datetime.now()
        results = []
        
        test_cases = [
            ("Swarm Initialization", self.test_swarm_initialization),
            ("CSV Operations", self.test_csv_operations),
            ("Task Execution", self.test_task_execution)
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
                logger.error(f"Unexpected error in {test_name}: {e}")
                results.append({
                    "test_name": test_name,
                    "success": False,
                    "message": f"Unexpected error: {e}",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Generate report
        duration = (datetime.now() - start_time).total_seconds()
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
        report_path = os.path.join(
            self.test_data_path,
            f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        async with aiofiles.open(report_path, 'w') as f:
            await f.write(json.dumps(report, indent=2))
        
        logger.info(f"Test suite completed. Report saved to {report_path}")
        return report

async def main():
    """Entry point for test execution."""
    logger.info("Starting SpreadSheetSwarm Test Suite")
    
    test_suite = SpreadSheetSwarmTestSuite()
    report = await test_suite.run_all_tests()
    
    # Print summary
    print("\n" + "="*50)
    print("SpreadSheetSwarm Test Results")
    print("="*50)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed Tests: {report['summary']['passed_tests']}")
    print(f"Failed Tests: {report['summary']['failed_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']}")
    print(f"Duration: {report['summary']['duration_seconds']:.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())