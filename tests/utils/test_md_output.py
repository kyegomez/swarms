import os

from dotenv import load_dotenv

from swarms import (
    Agent,
    ConcurrentWorkflow,
    GroupChat,
    SequentialWorkflow,
)

from swarms.utils.formatter import Formatter

# Load environment variables
load_dotenv()


class MarkdownTestSwarm:
    """A test swarm that demonstrates markdown output capabilities"""

    def __init__(self):
        self.formatter = Formatter(markdown=True)
        self.setup_agents()
        self.setup_swarm()

    def setup_agents(self):
        """Setup specialized agents for markdown testing"""

        # Research Agent - Generates structured markdown reports
        self.research_agent = Agent(
            agent_name="Research Agent",
            system_prompt="""You are a research specialist. When given a topic, create a comprehensive markdown report with:
            - Clear headers and subheaders
            - Code examples when relevant
            - Bullet points and numbered lists
            - Bold and italic text for emphasis
            - Tables for data comparison
            - Code blocks with syntax highlighting
            
            Always format your response as clean markdown with proper structure.""",
            model_name="gpt-4o-mini",  # Use a more capable model
            temperature=0.7,
            max_tokens=4000,
            max_loops=1,
            context_length=8000,  # Limit context to prevent overflow
            return_history=False,  # Don't return history to reduce context
        )

        # Code Analysis Agent - Generates code-heavy markdown
        self.code_agent = Agent(
            agent_name="Code Analysis Agent",
            system_prompt="""You are a code analysis specialist. When given code or programming concepts, create markdown documentation with:
            - Syntax-highlighted code blocks
            - Function documentation
            - Code examples
            - Performance analysis
            - Best practices
            
            Use proper markdown formatting with code blocks, inline code, and structured content.""",
            model_name="gpt-4o-mini",  # Use a more capable model
            temperature=0.5,
            max_tokens=4000,
            max_loops=1,
            context_length=8000,  # Limit context to prevent overflow
            return_history=False,  # Don't return history to reduce context
        )

        # Data Visualization Agent - Creates data-focused markdown
        self.data_agent = Agent(
            agent_name="Data Visualization Agent",
            system_prompt="""You are a data visualization specialist. When given data or analysis requests, create markdown reports with:
            - Data tables
            - Statistical analysis
            - Charts and graphs descriptions
            - Key insights with bold formatting
            - Recommendations in structured lists
            
            Format everything as clean, readable markdown.""",
            model_name="gpt-4o-mini",  # Use a more capable model
            temperature=0.6,
            max_tokens=4000,
            max_loops=1,
            context_length=8000,  # Limit context to prevent overflow
            return_history=False,  # Don't return history to reduce context
        )

    def setup_swarm(self):
        """Setup the swarm with the agents"""
        # Create different swarm types for testing
        self.sequential_swarm = SequentialWorkflow(
            name="Markdown Test Sequential",
            description="Sequential workflow for markdown testing",
            agents=[
                self.research_agent,
                self.code_agent,
                self.data_agent,
            ],
            max_loops=1,  # Reduce loops to prevent context overflow
        )

        self.concurrent_swarm = ConcurrentWorkflow(
            name="Markdown Test Concurrent",
            description="Concurrent workflow for markdown testing",
            agents=[
                self.research_agent,
                self.code_agent,
                self.data_agent,
            ],
            max_loops=1,  # Reduce loops to prevent context overflow
        )

        self.groupchat_swarm = GroupChat(
            name="Markdown Test Group Chat",
            description="A group chat for testing markdown output",
            agents=[
                self.research_agent,
                self.code_agent,
                self.data_agent,
            ],
            max_loops=1,  # Reduce loops to prevent context overflow
        )

        # Default swarm for main tests
        self.swarm = self.sequential_swarm

    def test_basic_markdown_output(self):
        """Test basic markdown output with a simple topic"""
        print("\n" + "=" * 60)
        print("TEST 1: Basic Markdown Output")
        print("=" * 60)

        topic = "Python Web Development with FastAPI"

        self.formatter.print_panel(
            f"Starting research on: {topic}",
            title="Research Topic",
            style="bold blue",
        )

        # Run the research agent
        result = self.research_agent.run(topic)

        self.formatter.print_markdown(
            result, title="Research Report", border_style="green"
        )

    def test_code_analysis_markdown(self):
        """Test markdown output with code analysis"""
        print("\n" + "=" * 60)
        print("TEST 2: Code Analysis Markdown")
        print("=" * 60)

        code_sample = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the function
for i in range(10):
    print(fibonacci(i))
        """

        self.formatter.print_panel(
            "Analyzing Python code sample",
            title="Code Analysis",
            style="bold cyan",
        )

        # Run the code analysis agent
        result = self.code_agent.run(
            f"Analyze this Python code and provide improvements:\n\n{code_sample}"
        )

        self.formatter.print_markdown(
            result,
            title="Code Analysis Report",
            border_style="yellow",
        )

    def test_data_analysis_markdown(self):
        """Test markdown output with data analysis"""
        print("\n" + "=" * 60)
        print("TEST 3: Data Analysis Markdown")
        print("=" * 60)

        data_request = """
        Analyze the following dataset:
        - Sales: $1.2M (Q1), $1.5M (Q2), $1.8M (Q3), $2.1M (Q4)
        - Growth Rate: 8%, 12%, 15%, 18%
        - Customer Count: 1000, 1200, 1400, 1600
        
        Provide insights and recommendations in markdown format.
        """

        self.formatter.print_panel(
            "Analyzing quarterly business data",
            title="Data Analysis",
            style="bold magenta",
        )

        # Run the data analysis agent
        result = self.data_agent.run(data_request)

        self.formatter.print_markdown(
            result, title="Data Analysis Report", border_style="red"
        )

    def test_swarm_collaboration_markdown(self):
        """Test markdown output with swarm collaboration"""
        print("\n" + "=" * 60)
        print("TEST 4: Swarm Collaboration Markdown")
        print("=" * 60)

        complex_topic = """
        Create a comprehensive guide on building a machine learning pipeline that includes:
        1. Data preprocessing techniques
        2. Model selection strategies
        3. Performance evaluation metrics
        4. Deployment considerations
        
        Each agent should contribute their expertise and the final output should be well-formatted markdown.
        """

        self.formatter.print_panel(
            "Swarm collaboration on ML pipeline guide",
            title="Swarm Task",
            style="bold green",
        )

        # Run the swarm
        results = self.swarm.run(complex_topic)

        # Display individual agent results
        # SequentialWorkflow returns a list of results, not a dict
        for i, result in enumerate(results, 1):
            agent_name = f"Agent {i}"

            # Handle different result types
            if isinstance(result, dict):
                # Extract the output from dict result
                result_content = result.get("output", str(result))
            else:
                result_content = str(result)
            self.formatter.print_markdown(
                result_content,
                title=f"Agent {i}: {agent_name}",
                border_style="blue",
            )

    def test_markdown_toggle_functionality(self):
        """Test the markdown enable/disable functionality"""
        print("\n" + "=" * 60)
        print("TEST 5: Markdown Toggle Functionality")
        print("=" * 60)

        test_content = """
# Test Content

This is a **bold** test with `inline code`.

## Code Block
```python
def test_function():
    return "Hello, World!"
```

## List
- Item 1
- Item 2
- Item 3
        """

        # Test with markdown enabled
        self.formatter.print_panel(
            "Testing with markdown ENABLED",
            title="Markdown Enabled",
            style="bold green",
        )
        self.formatter.print_markdown(test_content, "Markdown Output")

        # Disable markdown
        self.formatter.disable_markdown()
        self.formatter.print_panel(
            "Testing with markdown DISABLED",
            title="Markdown Disabled",
            style="bold red",
        )
        self.formatter.print_panel(test_content, "Plain Text Output")

        # Re-enable markdown
        self.formatter.enable_markdown()
        self.formatter.print_panel(
            "Testing with markdown RE-ENABLED",
            title="Markdown Re-enabled",
            style="bold blue",
        )
        self.formatter.print_markdown(
            test_content, "Markdown Output Again"
        )

    def test_different_swarm_types(self):
        """Test markdown output with different swarm types"""
        print("\n" + "=" * 60)
        print("TEST 6: Different Swarm Types")
        print("=" * 60)

        simple_topic = (
            "Explain the benefits of using Python for data science"
        )

        # Test Sequential Workflow
        print("\n--- Sequential Workflow ---")
        self.formatter.print_panel(
            "Testing Sequential Workflow (agents work in sequence)",
            title="Swarm Type Test",
            style="bold blue",
        )
        sequential_results = self.sequential_swarm.run(simple_topic)
        for i, result in enumerate(sequential_results, 1):
            # Handle different result types
            if isinstance(result, dict):
                result_content = result.get("output", str(result))
            else:
                result_content = str(result)

            self.formatter.print_markdown(
                result_content,
                title=f"Sequential Agent {i}",
                border_style="blue",
            )

        # Test Concurrent Workflow
        print("\n--- Concurrent Workflow ---")
        self.formatter.print_panel(
            "Testing Concurrent Workflow (agents work in parallel)",
            title="Swarm Type Test",
            style="bold green",
        )
        concurrent_results = self.concurrent_swarm.run(simple_topic)
        for i, result in enumerate(concurrent_results, 1):
            # Handle different result types
            if isinstance(result, dict):
                result_content = result.get("output", str(result))
            else:
                result_content = str(result)

            self.formatter.print_markdown(
                result_content,
                title=f"Concurrent Agent {i}",
                border_style="green",
            )

        # Test Group Chat
        print("\n--- Group Chat ---")
        self.formatter.print_panel(
            "Testing Group Chat (agents collaborate in conversation)",
            title="Swarm Type Test",
            style="bold magenta",
        )
        groupchat_results = self.groupchat_swarm.run(simple_topic)

        # Handle different result types for GroupChat
        if isinstance(groupchat_results, dict):
            result_content = groupchat_results.get(
                "output", str(groupchat_results)
            )
        else:
            result_content = str(groupchat_results)

        self.formatter.print_markdown(
            result_content,
            title="Group Chat Result",
            border_style="magenta",
        )

    def test_simple_formatter_only(self):
        """Test just the formatter functionality without agents"""
        print("\n" + "=" * 60)
        print("TEST 7: Simple Formatter Test (No Agents)")
        print("=" * 60)

        # Test basic markdown rendering
        simple_markdown = """
# Simple Test

This is a **bold** test with `inline code`.

## Code Block
```python
def hello_world():
    print("Hello, World!")
    return "Success"
```

## List
- Item 1
- Item 2
- Item 3
        """

        self.formatter.print_panel(
            "Testing formatter without agents",
            title="Formatter Test",
            style="bold cyan",
        )

        self.formatter.print_markdown(
            simple_markdown,
            title="Simple Markdown Test",
            border_style="green",
        )

        # Test toggle functionality
        self.formatter.disable_markdown()
        self.formatter.print_panel(
            "Markdown disabled - this should be plain text",
            title="Plain Text Test",
            style="bold red",
        )
        self.formatter.enable_markdown()

    def test_error_handling_markdown(self):
        """Test markdown output with error handling"""
        print("\n" + "=" * 60)
        print("TEST 8: Error Handling in Markdown")
        print("=" * 60)

        # Test with malformed markdown
        malformed_content = """
# Incomplete header
**Unclosed bold
```python
def incomplete_code():
    # Missing closing backticks
        """

        self.formatter.print_panel(
            "Testing error handling with malformed markdown",
            title="Error Handling Test",
            style="bold yellow",
        )

        # This should handle the error gracefully
        self.formatter.print_markdown(
            malformed_content,
            title="Malformed Markdown Test",
            border_style="yellow",
        )

        # Test with empty content
        self.formatter.print_markdown(
            "", title="Empty Content Test", border_style="cyan"
        )

        # Test with None content
        self.formatter.print_markdown(
            None, title="None Content Test", border_style="magenta"
        )

    def run_all_tests(self):
        """Run all markdown output tests"""
        print(" Starting Swarm Markdown Output Tests")
        print("=" * 60)

        try:
            # Test 1: Basic markdown output
            self.test_basic_markdown_output()

            # Test 2: Code analysis markdown
            self.test_code_analysis_markdown()

            # Test 3: Data analysis markdown
            self.test_data_analysis_markdown()

            # Test 4: Swarm collaboration
            self.test_swarm_collaboration_markdown()

            # Test 5: Markdown toggle functionality
            self.test_markdown_toggle_functionality()

            # Test 6: Different swarm types
            self.test_different_swarm_types()

            # Test 7: Simple formatter test (no agents)
            self.test_simple_formatter_only()

            # Test 8: Error handling
            self.test_error_handling_markdown()

            print("\n" + "=" * 60)
            print(" All tests completed successfully!")
            print("=" * 60)

        except Exception as e:
            print(f"\n Test failed with error: {str(e)}")
            import traceback

            traceback.print_exc()


def main():
    """Main function to run the markdown output tests"""
    print("Swarms Markdown Output Test Suite")
    print(
        "Testing the current state of formatter.py with real swarm agents"
    )
    print("=" * 60)

    # Check environment setup
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv(
        "SWARMS_API_KEY"
    )
    if not api_key:
        print(
            "⚠  Warning: No API key found. Please set OPENAI_API_KEY or SWARMS_API_KEY environment variable."
        )
        print(
            "   You can create a .env file with: OPENAI_API_KEY=your_api_key_here"
        )
        print(
            "   Or set it in your environment: export OPENAI_API_KEY=your_api_key_here"
        )
        print()

    try:
        # Create and run the test swarm
        test_swarm = MarkdownTestSwarm()
        test_swarm.run_all_tests()
    except Exception as e:
        print(f"\n Test failed with error: {str(e)}")
        print("\n Troubleshooting tips:")
        print(
            "1. Make sure you have set your API key (OPENAI_API_KEY or SWARMS_API_KEY)"
        )
        print("2. Check your internet connection")
        print("3. Verify you have sufficient API credits")
        print("4. Try running with a simpler test first")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
