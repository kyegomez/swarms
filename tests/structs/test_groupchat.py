from swarms.structs.agent import Agent
from swarms.structs.groupchat import GroupChat, RESPOND_TOOL
from datetime import datetime
import time


class TestReport:
    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None

    def add_result(self, test_name, passed, message="", duration=0):
        self.results.append(
            {
                "test_name": test_name,
                "passed": passed,
                "message": message,
                "duration": duration,
            }
        )

    def start(self):
        self.start_time = datetime.now()

    def end(self):
        self.end_time = datetime.now()

    def generate_report(self):
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["passed"])
        failed_tests = total_tests - passed_tests
        duration = (self.end_time - self.start_time).total_seconds()

        report = "\n" + "=" * 50 + "\n"
        report += "GROUP CHAT TEST SUITE REPORT\n"
        report += "=" * 50 + "\n\n"
        report += f"Test Run: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Duration: {duration:.2f} seconds\n"
        report += f"Total Tests: {total_tests}\n"
        report += f"Passed: {passed_tests}\n"
        report += f"Failed: {failed_tests}\n"
        report += (
            f"Success Rate: {(passed_tests/total_tests)*100:.1f}%\n\n"
        )

        report += "Detailed Test Results:\n"
        report += "-" * 50 + "\n"

        for result in self.results:
            status = "✓" if result["passed"] else "✗"
            report += f"{status} {result['test_name']} ({result['duration']:.2f}s)\n"
            if result["message"]:
                report += f"   {result['message']}\n"

        return report


def create_test_agents(num_agents, diverse_prompts=False):
    """Helper function to create test agents with diverse prompts.

    Each agent is configured for the new dynamic GroupChat:
    - ``tools_list_dictionary=[RESPOND_TOOL]`` so the agent emits a structured
      ``respond(score, message)`` decision per inbound message.
    - ``max_loops=1`` because GroupChat itself orchestrates iterations.
    - ``persistent_memory=False`` so test runs do not pollute on-disk memory.
    """
    agents = []
    specialties = [
        (
            "Finance",
            "You are a financial expert focusing on investment strategies and market analysis. Be concise and data-driven in your responses.",
        ),
        (
            "Tech",
            "You are a technology expert specializing in AI and cybersecurity. Use technical terms and provide practical examples.",
        ),
        (
            "Healthcare",
            "You are a healthcare professional with expertise in public health. Focus on evidence-based information and patient care.",
        ),
        (
            "Marketing",
            "You are a marketing strategist focusing on digital trends. Be creative and audience-focused in your responses.",
        ),
        (
            "Legal",
            "You are a legal expert specializing in corporate law. Be precise and reference relevant regulations.",
        ),
    ]

    for i in range(num_agents):
        specialty, base_prompt = specialties[i % len(specialties)]
        if diverse_prompts:
            traits = [
                "Be analytical and data-focused",
                "Use analogies and examples",
                "Be concise and direct",
                "Ask thought-provoking questions",
                "Provide practical applications",
            ]
            prompt = f"{base_prompt} {traits[i % len(traits)]}"
        else:
            prompt = base_prompt

        agents.append(
            Agent(
                agent_name=f"{specialty}-Agent-{i+1}",
                system_prompt=prompt,
                model_name="gpt-4",
                max_loops=1,
                temperature=0.7,
                persistent_memory=False,
                tools_list_dictionary=[RESPOND_TOOL],
            )
        )
    return agents


def test_basic_groupchat(report):
    """Test basic GroupChat initialization and a single conversation run."""
    start_time = time.time()

    try:
        agents = create_test_agents(2)
        chat = GroupChat(
            name="Test Chat",
            description="A test group chat",
            agents=agents,
            max_loops=2,
            idle_timeout=4.0,
        )

        result = chat.run("Say hello!")
        assert result is not None
        report.add_result(
            "Basic GroupChat Test",
            True,
            duration=time.time() - start_time,
        )

    except Exception as e:
        report.add_result(
            "Basic GroupChat Test",
            False,
            message=str(e),
            duration=time.time() - start_time,
        )


def test_varying_agent_counts(report):
    """Test GroupChat with different numbers of agents.

    GroupChat now requires at least 2 agents, so we test 2, 3, and 5.
    """
    agent_counts = [2, 3, 5]

    for count in agent_counts:
        start_time = time.time()
        try:
            agents = create_test_agents(count)
            chat = GroupChat(
                name=f"{count}-Agent Test",
                agents=agents,
                max_loops=2,
                idle_timeout=4.0,
            )

            chat.run("Introduce yourselves briefly.")
            report.add_result(
                f"Agent Count Test - {count} agents",
                True,
                duration=time.time() - start_time,
            )

        except Exception as e:
            report.add_result(
                f"Agent Count Test - {count} agents",
                False,
                message=str(e),
                duration=time.time() - start_time,
            )


def test_threshold_behavior(report):
    """Test GroupChat with high and low publish thresholds.

    A high ``threshold`` (0.9) makes agents very selective; a low threshold
    (0.1) makes them eager to publish. Both must run without error.
    """
    for label, threshold in (("high", 0.9), ("low", 0.1)):
        start_time = time.time()
        try:
            agents = create_test_agents(2)
            chat = GroupChat(
                name=f"Threshold-{label}",
                agents=agents,
                max_loops=2,
                threshold=threshold,
                idle_timeout=4.0,
            )
            chat.run("Briefly comment on the future of remote work.")
            report.add_result(
                f"Threshold Behavior - {label} ({threshold})",
                True,
                duration=time.time() - start_time,
            )
        except Exception as e:
            report.add_result(
                f"Threshold Behavior - {label} ({threshold})",
                False,
                message=str(e),
                duration=time.time() - start_time,
            )


def test_idle_timeout(report):
    """Test GroupChat early termination via a small ``idle_timeout``.

    With a 1-second idle timeout, the chat should stop quickly once no new
    messages are produced. We verify completion within a generous wall-clock
    budget that still confirms early termination.
    """
    start_time = time.time()
    try:
        agents = create_test_agents(2)
        chat = GroupChat(
            name="Idle Timeout Test",
            agents=agents,
            max_loops=50,  # high cap so termination must come from idle
            idle_timeout=1.0,
            threshold=0.99,  # extremely selective → silence is likely
        )
        chat.run("Hello there.")
        elapsed = time.time() - start_time

        # If idle_timeout works, total runtime should not blow past max_loops.
        # We assert it finished in a reasonable window (well under 60s).
        assert (
            elapsed < 60
        ), f"Idle timeout did not fire in time: {elapsed:.1f}s"

        report.add_result(
            "Idle Timeout Test",
            True,
            message=f"Completed in {elapsed:.2f}s",
            duration=elapsed,
        )
    except Exception as e:
        report.add_result(
            "Idle Timeout Test",
            False,
            message=str(e),
            duration=time.time() - start_time,
        )


def test_error_cases(report):
    """Test error handling for the new GroupChat constructor."""
    test_cases = [
        ("Empty Agents List", lambda: GroupChat(agents=[])),
        (
            "Single Agent",
            lambda: GroupChat(agents=create_test_agents(1)),
        ),
        ("None Agents", lambda: GroupChat(agents=None)),
    ]

    for name, test_func in test_cases:
        start_time = time.time()
        try:
            test_func()
            report.add_result(
                f"Error Case - {name}",
                False,
                message="Expected ValueError not raised",
                duration=time.time() - start_time,
            )
        except ValueError:
            report.add_result(
                f"Error Case - {name}",
                True,
                duration=time.time() - start_time,
            )
        except Exception as e:
            report.add_result(
                f"Error Case - {name}",
                False,
                message=f"Unexpected error: {type(e).__name__}: {e}",
                duration=time.time() - start_time,
            )


if __name__ == "__main__":
    report = TestReport()
    report.start()

    print("Starting GroupChat Test Suite...\n")

    test_basic_groupchat(report)
    test_varying_agent_counts(report)
    test_threshold_behavior(report)
    test_idle_timeout(report)
    test_error_cases(report)

    report.end()
    print(report.generate_report())
