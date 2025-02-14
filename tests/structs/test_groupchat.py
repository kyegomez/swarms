from swarms.structs.agent import Agent
from swarms.structs.groupchat import (
    GroupChat,
    round_robin,
    expertise_based,
    random_selection,
    sentiment_based,
    length_based,
    question_based,
    topic_based,
)
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
    """Helper function to create test agents with diverse prompts"""
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
            # Add personality traits and communication style to make responses more diverse
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
                temperature=0.7,  # Add temperature to increase response variety
            )
        )
    return agents


def test_basic_groupchat(report):
    """Test basic GroupChat initialization and conversation"""
    start_time = time.time()

    try:
        agents = create_test_agents(2)
        chat = GroupChat(
            name="Test Chat",
            description="A test group chat",
            agents=agents,
            max_loops=2,
        )

        result = chat.run("Say hello!")
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


def test_speaker_functions(report):
    """Test all available speaker functions with enhanced prompts"""
    speaker_functions = {
        "round_robin": (
            round_robin,
            "What are your thoughts on sustainable practices?",
        ),
        "expertise_based": (
            expertise_based,
            "Discuss the impact of AI on your field.",
        ),
        "random_selection": (
            random_selection,
            "How do you approach problem-solving?",
        ),
        "sentiment_based": (
            sentiment_based,
            "Share your positive outlook on future trends.",
        ),
        "length_based": (
            length_based,
            "Provide a detailed analysis of recent developments.",
        ),
        "question_based": (
            question_based,
            "What challenges do you foresee in your industry?",
        ),
        "topic_based": (
            topic_based,
            "How does digital transformation affect your sector?",
        ),
    }

    for name, (func, prompt) in speaker_functions.items():
        start_time = time.time()
        try:
            # Create agents with diverse prompts for this test
            agents = create_test_agents(3, diverse_prompts=True)
            chat = GroupChat(
                name=f"{name.title()} Test",
                description=f"Testing {name} speaker function with diverse responses",
                agents=agents,
                speaker_fn=func,
                max_loops=2,
                rules="1. Be unique in your responses\n2. Build on others' points\n3. Stay relevant to your expertise",
            )

            result = chat.run(prompt)
            report.add_result(
                f"Speaker Function - {name}",
                True,
                duration=time.time() - start_time,
            )

        except Exception as e:
            report.add_result(
                f"Speaker Function - {name}",
                False,
                message=str(e),
                duration=time.time() - start_time,
            )


def test_varying_agent_counts(report):
    """Test GroupChat with different numbers of agents"""
    agent_counts = [1, 3, 5, 7]

    for count in agent_counts:
        start_time = time.time()
        try:
            agents = create_test_agents(count)
            chat = GroupChat(
                name=f"{count}-Agent Test", agents=agents, max_loops=2
            )

            result = chat.run("Introduce yourselves briefly.")
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


def test_error_cases(report):
    """Test error handling with expanded cases"""
    test_cases = [
        ("Empty Agents List", lambda: GroupChat(agents=[])),
        (
            "Invalid Max Loops",
            lambda: GroupChat(
                agents=[create_test_agents(1)[0]], max_loops=0
            ),
        ),
        (
            "Empty Task",
            lambda: GroupChat(agents=[create_test_agents(1)[0]]).run(
                ""
            ),
        ),
        (
            "None Task",
            lambda: GroupChat(agents=[create_test_agents(1)[0]]).run(
                None
            ),
        ),
        (
            "Invalid Speaker Function",
            lambda: GroupChat(
                agents=[create_test_agents(1)[0]],
                speaker_fn=lambda x, y: "not a boolean",  # This should raise ValueError
            ),
        ),
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
        except (
            ValueError,
            TypeError,
        ):  # Include TypeError for invalid speaker function
            report.add_result(
                f"Error Case - {name}",
                True,
                duration=time.time() - start_time,
            )
        except Exception as e:
            report.add_result(
                f"Error Case - {name}",
                False,
                message=f"Unexpected error: {str(e)}",
                duration=time.time() - start_time,
            )


def test_concurrent_execution(report):
    """Test concurrent execution with various task counts"""
    start_time = time.time()

    try:
        agents = create_test_agents(3)
        chat = GroupChat(
            name="Concurrent Test", agents=agents, max_loops=1
        )

        tasks = [
            "Task 1: Introduce yourself",
            "Task 2: What's your specialty?",
            "Task 3: How can you help?",
            "Task 4: What are your limitations?",
            "Task 5: Give an example of your expertise",
        ]

        results = chat.concurrent_run(tasks)
        report.add_result(
            "Concurrent Execution Test",
            True,
            message=f"Successfully completed {len(results)} tasks",
            duration=time.time() - start_time,
        )

    except Exception as e:
        report.add_result(
            "Concurrent Execution Test",
            False,
            message=str(e),
            duration=time.time() - start_time,
        )


def test_conversation_rules(report):
    """Test GroupChat with different conversation rules"""
    start_time = time.time()

    try:
        agents = create_test_agents(3, diverse_prompts=True)
        chat = GroupChat(
            name="Rules Test",
            description="Testing conversation with specific rules",
            agents=agents,
            max_loops=2,
            rules="""
            1. Keep responses under 50 words
            2. Always be professional
            3. Stay on topic
            4. Provide unique perspectives
            5. Build on previous responses
            """,
        )

        result = chat.run(
            "How can we ensure ethical AI development across different sectors?"
        )
        report.add_result(
            "Conversation Rules Test",
            True,
            duration=time.time() - start_time,
        )

    except Exception as e:
        report.add_result(
            "Conversation Rules Test",
            False,
            message=str(e),
            duration=time.time() - start_time,
        )


if __name__ == "__main__":
    report = TestReport()
    report.start()

    print("Starting Enhanced GroupChat Test Suite...\n")

    # Run all tests
    test_basic_groupchat(report)
    test_speaker_functions(report)
    test_varying_agent_counts(report)
    test_error_cases(report)
    test_concurrent_execution(report)
    test_conversation_rules(report)

    report.end()
    print(report.generate_report())
