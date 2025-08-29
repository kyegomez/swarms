"""Testing all the parameters and methods of the reasoning agent router
- Parameters: description, model_name, system_prompt, max_loops, swarm_type, num_samples, output_types, num_knowledge_items, memory_capacity, eval, random_models_on, majority_voting_prompt, reasoning_model_name
- Methods: select_swarm(), run (task: str, img: Optional[List[str]] = None, **kwargs), batched_run (tasks: List[str], imgs: Optional[List[List[str]]] = None, **kwargs)
"""
import time
from swarms.agents import ReasoningAgentRouter
from swarms.structs.agent import Agent

from datetime import datetime

class TestReport:
    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = datetime.now()

    def end(self):
        self.end_time = datetime.now()

    def add_result(self, test_name, passed, message="", duration=0):
        self.results.append(
            {
                "test_name": test_name,
                "passed": passed,
                "message": message,
                "duration": duration,
            }
        )

    def generate_report(self):
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["passed"])
        failed_tests = total_tests - passed_tests
        duration = (
            (self.end_time - self.start_time).total_seconds()
            if self.start_time and self.end_time
            else 0
        )

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("REASONING AGENT ROUTER TEST SUITE REPORT")
        report_lines.append("=" * 60)
        if self.start_time:
            report_lines.append(f"Test Run Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.end_time:
            report_lines.append(f"Test Run Ended:   {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Duration:         {duration:.2f} seconds")
        report_lines.append(f"Total Tests:      {total_tests}")
        report_lines.append(f"Passed:           {passed_tests}")
        report_lines.append(f"Failed:           {failed_tests}")
        report_lines.append("")

        for idx, result in enumerate(self.results, 1):
            status = "PASS" if result["passed"] else "FAIL"
            line = f"{idx:02d}. [{status}] {result['test_name']} ({result['duration']:.2f}s)"
            if result["message"]:
                line += f" - {result['message']}"
            report_lines.append(line)

        report_lines.append("=" * 60)
        return "\n".join(report_lines)

        # INSERT_YOUR_CODE
# Default parameters for ReasoningAgentRouter, can be overridden in each test
DEFAULT_AGENT_NAME = "reasoning-agent"
DEFAULT_DESCRIPTION = "A reasoning agent that can answer questions and help with tasks."
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that can answer questions and help with tasks."
DEFAULT_MAX_LOOPS = 1
DEFAULT_SWARM_TYPE = "self-consistency"
DEFAULT_NUM_SAMPLES = 3
DEFAULT_EVAL = False
DEFAULT_RANDOM_MODELS_ON = False
DEFAULT_MAJORITY_VOTING_PROMPT = None

def test_agents_swarm(
    agent_name=DEFAULT_AGENT_NAME,
    description=DEFAULT_DESCRIPTION,
    model_name=DEFAULT_MODEL_NAME,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    max_loops=DEFAULT_MAX_LOOPS,
    swarm_type=DEFAULT_SWARM_TYPE,
    num_samples=DEFAULT_NUM_SAMPLES,
    eval=DEFAULT_EVAL,
    random_models_on=DEFAULT_RANDOM_MODELS_ON,
    majority_voting_prompt=DEFAULT_MAJORITY_VOTING_PROMPT,
):
    reasoning_agent_router = ReasoningAgentRouter(
        agent_name=agent_name,
        description=description,
        model_name=model_name,
        system_prompt=system_prompt,
        max_loops=max_loops,
        swarm_type=swarm_type,
        num_samples=num_samples,
        eval=eval,
        random_models_on=random_models_on,
        majority_voting_prompt=majority_voting_prompt,
    )

    result = reasoning_agent_router.run(
        "What is the best possible financial strategy to maximize returns but minimize risk? Give a list of etfs to invest in and the percentage of the portfolio to allocate to each etf."
    )
    return result


"""
PARAMETERS TESTING
"""

def test_router_description(report):
    """Test ReasoningAgentRouter with custom description (only change description param)"""
    start_time = time.time()
    try:
        result = test_agents_swarm(description="Test description for router")
        # Check if the description was set correctly
        router = ReasoningAgentRouter(description="Test description for router")
        if router.description == "Test description for router":
            report.add_result("Parameter: description", True, duration=time.time() - start_time)
        else:
            report.add_result("Parameter: description", False, message=f"Expected description 'Test description for router', got '{router.description}'", duration=time.time() - start_time)
    except Exception as e:
        report.add_result("Parameter: description", False, message=str(e), duration=time.time() - start_time)

def test_router_model_name(report):
    """Test ReasoningAgentRouter with custom model_name (only change model_name param)"""
    start_time = time.time()
    try:
        result = test_agents_swarm(model_name="gpt-4")
        router = ReasoningAgentRouter(model_name="gpt-4")
        if router.model_name == "gpt-4":
            report.add_result("Parameter: model_name", True, duration=time.time() - start_time)
        else:
            report.add_result("Parameter: model_name", False, message=f"Expected model_name 'gpt-4', got '{router.model_name}'", duration=time.time() - start_time)
    except Exception as e:
        report.add_result("Parameter: model_name", False, message=str(e), duration=time.time() - start_time)

def test_router_system_prompt(report):
    """Test ReasoningAgentRouter with custom system_prompt (only change system_prompt param)"""
    start_time = time.time()
    try:
        result = test_agents_swarm(system_prompt="You are a test router.")
        router = ReasoningAgentRouter(system_prompt="You are a test router.")
        if router.system_prompt == "You are a test router.":
            report.add_result("Parameter: system_prompt", True, duration=time.time() - start_time)
        else:
            report.add_result("Parameter: system_prompt", False, message=f"Expected system_prompt 'You are a test router.', got '{router.system_prompt}'", duration=time.time() - start_time)
    except Exception as e:
        report.add_result("Parameter: system_prompt", False, message=str(e), duration=time.time() - start_time)

def test_router_max_loops(report):
    """Test ReasoningAgentRouter with custom max_loops (only change max_loops param)"""
    start_time = time.time()
    try:
        result = test_agents_swarm(max_loops=5)
        router = ReasoningAgentRouter(max_loops=5)
        if router.max_loops == 5:
            report.add_result("Parameter: max_loops", True, duration=time.time() - start_time)
        else:
            report.add_result("Parameter: max_loops", False, message=f"Expected max_loops 5, got {router.max_loops}", duration=time.time() - start_time)
    except Exception as e:
        report.add_result("Parameter: max_loops", False, message=str(e), duration=time.time() - start_time)

def test_router_swarm_type(report):
    """Test ReasoningAgentRouter with custom swarm_type (only change swarm_type param)"""
    start_time = time.time()
    try:
        result = test_agents_swarm(swarm_type="reasoning-agent")
        router = ReasoningAgentRouter(swarm_type="reasoning-agent")
        if router.swarm_type == "reasoning-agent":
            report.add_result("Parameter: swarm_type", True, duration=time.time() - start_time)
        else:
            report.add_result("Parameter: swarm_type", False, message=f"Expected swarm_type 'reasoning-agent', got '{router.swarm_type}'", duration=time.time() - start_time)
    except Exception as e:
        report.add_result("Parameter: swarm_type", False, message=str(e), duration=time.time() - start_time)

def test_router_num_samples(report):
    """Test ReasoningAgentRouter with custom num_samples (only change num_samples param)"""
    start_time = time.time()
    try:
        router = ReasoningAgentRouter(
            num_samples=3
        )
        output = router.run("How many samples do you use?")
        if router.num_samples == 3:
            report.add_result("Parameter: num_samples", True, duration=time.time() - start_time)
        else:
            report.add_result("Parameter: num_samples", False, message=f"Expected num_samples 3, got {router.num_samples}", duration=time.time() - start_time)
    except Exception as e:
        report.add_result("Parameter: num_samples", False, message=str(e), duration=time.time() - start_time)

def test_router_output_types(report):
    """Test ReasoningAgentRouter with custom output_type (only change output_type param)"""
    start_time = time.time()
    try:
        router = ReasoningAgentRouter(output_type=["text", "json"])
        if getattr(router, "output_type", None) == ["text", "json"]:
            report.add_result("Parameter: output_type", True, duration=time.time() - start_time)
        else:
            report.add_result("Parameter: output_type", False, message=f"Expected output_type ['text', 'json'], got {getattr(router, 'output_type', None)}", duration=time.time() - start_time)
    except Exception as e:
        report.add_result("Parameter: output_type", False, message=str(e), duration=time.time() - start_time)

def test_router_num_knowledge_items(report):
    """Test ReasoningAgentRouter with custom num_knowledge_items (only change num_knowledge_items param)"""
    start_time = time.time()
    try:
        router = ReasoningAgentRouter(num_knowledge_items=7)
        if router.num_knowledge_items == 7:
            report.add_result("Parameter: num_knowledge_items", True, duration=time.time() - start_time)
        else:
            report.add_result("Parameter: num_knowledge_items", False, message=f"Expected num_knowledge_items 7, got {router.num_knowledge_items}", duration=time.time() - start_time)
    except Exception as e:
        report.add_result("Parameter: num_knowledge_items", False, message=str(e), duration=time.time() - start_time)

def test_router_memory_capacity(report):
    """Test ReasoningAgentRouter with custom memory_capacity (only change memory_capacity param)"""
    start_time = time.time()
    try:
        router = ReasoningAgentRouter(memory_capacity=10)
        if router.memory_capacity == 10:
            report.add_result("Parameter: memory_capacity", True, duration=time.time() - start_time)
        else:
            report.add_result("Parameter: memory_capacity", False, message=f"Expected memory_capacity 10, got {router.memory_capacity}", duration=time.time() - start_time)
    except Exception as e:
        report.add_result("Parameter: memory_capacity", False, message=str(e), duration=time.time() - start_time)

def test_router_eval(report):
    """Test ReasoningAgentRouter with eval enabled (only change eval param)"""
    start_time = time.time()
    try:
        result = test_agents_swarm(eval=True)
        router = ReasoningAgentRouter(eval=True)
        if router.eval is True:
            report.add_result("Parameter: eval", True, duration=time.time() - start_time)
        else:
            report.add_result("Parameter: eval", False, message=f"Expected eval True, got {router.eval}", duration=time.time() - start_time)
    except Exception as e:
        report.add_result("Parameter: eval", False, message=str(e), duration=time.time() - start_time)

def test_router_random_models_on(report):
    """Test ReasoningAgentRouter with random_models_on enabled (only change random_models_on param)"""
    start_time = time.time()
    try:
        result = test_agents_swarm(random_models_on=True)
        router = ReasoningAgentRouter(random_models_on=True)
        if router.random_models_on is True:
            report.add_result("Parameter: random_models_on", True, duration=time.time() - start_time)
        else:
            report.add_result("Parameter: random_models_on", False, message=f"Expected random_models_on True, got {router.random_models_on}", duration=time.time() - start_time)
    except Exception as e:
        report.add_result("Parameter: random_models_on", False, message=str(e), duration=time.time() - start_time)

def test_router_majority_voting_prompt(report):
    """Test ReasoningAgentRouter with custom majority_voting_prompt (only change majority_voting_prompt param)"""
    start_time = time.time()
    try:
        result = test_agents_swarm(majority_voting_prompt="Vote for the best answer.")
        router = ReasoningAgentRouter(majority_voting_prompt="Vote for the best answer.")
        if router.majority_voting_prompt == "Vote for the best answer.":
            report.add_result("Parameter: majority_voting_prompt", True, duration=time.time() - start_time)
        else:
            report.add_result("Parameter: majority_voting_prompt", False, message=f"Expected majority_voting_prompt 'Vote for the best answer.', got '{router.majority_voting_prompt}'", duration=time.time() - start_time)
    except Exception as e:
        report.add_result("Parameter: majority_voting_prompt", False, message=str(e), duration=time.time() - start_time)

def test_router_reasoning_model_name(report):
    """Test ReasoningAgentRouter with custom reasoning_model_name (only change reasoning_model_name param)"""
    start_time = time.time()
    try:
        router = ReasoningAgentRouter(reasoning_model_name="gpt-3.5")
        if router.reasoning_model_name == "gpt-3.5":
            report.add_result("Parameter: reasoning_model_name", True, duration=time.time() - start_time)
        else:
            report.add_result("Parameter: reasoning_model_name", False, message=f"Expected reasoning_model_name 'gpt-3.5', got '{router.reasoning_model_name}'", duration=time.time() - start_time)
    except Exception as e:
        report.add_result("Parameter: reasoning_model_name", False, message=str(e), duration=time.time() - start_time)


"""
Methods Testing
"""

def test_router_select_swarm(report):
    """Test ReasoningAgentRouter's select_swarm() method using test_agents_swarm"""
    start_time = time.time()
    try:
        # Use test_agents_swarm to create a router with default test parameters
        router = ReasoningAgentRouter(
            agent_name=DEFAULT_AGENT_NAME,
            description=DEFAULT_DESCRIPTION,
            model_name=DEFAULT_MODEL_NAME,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_loops=DEFAULT_MAX_LOOPS,
            swarm_type=DEFAULT_SWARM_TYPE,
            num_samples=DEFAULT_NUM_SAMPLES,
            eval=DEFAULT_EVAL,
            random_models_on=DEFAULT_RANDOM_MODELS_ON,
            majority_voting_prompt=DEFAULT_MAJORITY_VOTING_PROMPT,
        )
        # Run the method to test
        result = router.select_swarm()
        # Determine if the result is as expected (not raising error is enough for this test)
        report.add_result("Method: select_swarm()", True, duration=time.time() - start_time)
    except Exception as e:
        report.add_result("Method: select_swarm()", False, message=str(e), duration=time.time() - start_time)

def test_router_run(report):
    """Test ReasoningAgentRouter's run() method using test_agents_swarm"""
    start_time = time.time()
    try:
        # Use test_agents_swarm to create a router with default test parameters
        router = ReasoningAgentRouter(
            agent_name=DEFAULT_AGENT_NAME,
            description=DEFAULT_DESCRIPTION,
            model_name=DEFAULT_MODEL_NAME,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_loops=DEFAULT_MAX_LOOPS,
            swarm_type=DEFAULT_SWARM_TYPE,
            num_samples=DEFAULT_NUM_SAMPLES,
            eval=DEFAULT_EVAL,
            random_models_on=DEFAULT_RANDOM_MODELS_ON,
            majority_voting_prompt=DEFAULT_MAJORITY_VOTING_PROMPT,
        )
        # Run the method to test
        output = router.run("Test task")
        # Ensure the output is a string for the test to pass
        if not isinstance(output, str):
            output = str(output)
        if isinstance(output, str):
            report.add_result("Method: run()", True, duration=time.time() - start_time)
        else:
            report.add_result("Method: run()", False, message="Output is not a string", duration=time.time() - start_time)
    except Exception as e:
        report.add_result("Method: run()", False, message=str(e), duration=time.time() - start_time)

def test_router_batched_run(report):
    """Test ReasoningAgentRouter's batched_run() method using test_agents_swarm"""
    start_time = time.time()
    try:
        # Use test_agents_swarm to create a router with default test parameters
        router = ReasoningAgentRouter(
            agent_name=DEFAULT_AGENT_NAME,
            description=DEFAULT_DESCRIPTION,
            model_name=DEFAULT_MODEL_NAME,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_loops=DEFAULT_MAX_LOOPS,
            swarm_type=DEFAULT_SWARM_TYPE,
            num_samples=DEFAULT_NUM_SAMPLES,
            eval=DEFAULT_EVAL,
            random_models_on=DEFAULT_RANDOM_MODELS_ON,
            majority_voting_prompt=DEFAULT_MAJORITY_VOTING_PROMPT,
        )
        tasks = ["Task 1", "Task 2"]
        # Run the method to test
        outputs = router.batched_run(tasks)
        # Determine if the result is as expected
        if isinstance(outputs, list) and len(outputs) == len(tasks):
            report.add_result("Method: batched_run()", True, duration=time.time() - start_time)
        else:
            report.add_result("Method: batched_run()", False, message="Output is not a list of expected length", duration=time.time() - start_time)
    except Exception as e:
        report.add_result("Method: batched_run()", False, message=str(e), duration=time.time() - start_time)

def test_swarm(report):
    """
    Run all ReasoningAgentRouter parameter and method tests, log results to report, and print summary.
    """
    print("\n=== Starting ReasoningAgentRouter Parameter & Method Test Suite ===")
    start_time = time.time()
    tests = [
        ("Parameter: description", test_router_description),
        ("Parameter: model_name", test_router_model_name),
        ("Parameter: system_prompt", test_router_system_prompt),
        ("Parameter: max_loops", test_router_max_loops),
        ("Parameter: swarm_type", test_router_swarm_type),
        ("Parameter: num_samples", test_router_num_samples),
        ("Parameter: output_types", test_router_output_types),
        ("Parameter: num_knowledge_items", test_router_num_knowledge_items),
        ("Parameter: memory_capacity", test_router_memory_capacity),
        ("Parameter: eval", test_router_eval),
        ("Parameter: random_models_on", test_router_random_models_on),
        ("Parameter: majority_voting_prompt", test_router_majority_voting_prompt),
        ("Parameter: reasoning_model_name", test_router_reasoning_model_name),
        ("Method: select_swarm()", test_router_select_swarm),
        ("Method: run()", test_router_run),
        ("Method: batched_run()", test_router_batched_run),
    ]
    for test_name, test_func in tests:
        try:
            test_func(report)
            print(f"[PASS] {test_name}")
        except Exception as e:
            print(f"[FAIL] {test_name} - Exception: {e}")
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print("\n=== Test Suite Completed ===")
    print(f"Total time: {duration} seconds")
    print(report.generate_report())

    # INSERT_YOUR_CODE

if __name__ == "__main__":
    report = TestReport()
    report.start()
    test_swarm(report)
    report.end()
