import pytest
from swarms import Agent
from swarms.structs.batched_grid_workflow import BatchedGridWorkflow


def test_batched_grid_workflow_basic_execution():
    """Test basic BatchedGridWorkflow execution with multiple agents and tasks"""
    # Create specialized agents for different tasks
    research_agent = Agent(
        agent_name="Research-Analyst",
        agent_description="Agent specializing in research and data collection",
        model_name="gpt-4o",
        max_loops=1,
    )

    analysis_agent = Agent(
        agent_name="Data-Analyst",
        agent_description="Agent specializing in data analysis",
        model_name="gpt-4o",
        max_loops=1,
    )

    reporting_agent = Agent(
        agent_name="Report-Writer",
        agent_description="Agent specializing in report writing",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create workflow with multiple agents
    workflow = BatchedGridWorkflow(
        name="Basic-Batched-Grid-Workflow",
        description="Testing basic batched grid execution",
        agents=[research_agent, analysis_agent, reporting_agent],
        max_loops=1,
    )

    # Run workflow with different tasks for each agent
    tasks = [
        "Research the latest trends in artificial intelligence",
        "Analyze the impact of AI on job markets",
        "Write a summary report on AI developments",
    ]
    result = workflow.run(tasks)

    # Verify results
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1  # max_loops=1
    assert result[0] is not None


def test_batched_grid_workflow_multiple_loops():
    """Test BatchedGridWorkflow with multiple execution loops"""
    # Create agents for iterative processing
    agent1 = Agent(
        agent_name="Iteration-Agent-1",
        agent_description="Agent for iterative task processing",
        model_name="gpt-4o",
        max_loops=1,
    )

    agent2 = Agent(
        agent_name="Iteration-Agent-2",
        agent_description="Agent for iterative task refinement",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = BatchedGridWorkflow(
        name="Multi-Loop-Workflow",
        description="Testing multiple execution loops",
        agents=[agent1, agent2],
        max_loops=3,
    )

    tasks = [
        "Generate ideas for a new product",
        "Evaluate the feasibility of product ideas",
    ]
    results = workflow.run(tasks)

    # Verify multiple loop execution
    assert results is not None
    assert isinstance(results, list)
    assert len(results) == 3  # max_loops=3
    for result in results:
        assert result is not None


def test_batched_grid_workflow_single_agent_single_task():
    """Test BatchedGridWorkflow with a single agent and single task"""
    agent = Agent(
        agent_name="Solo-Agent",
        agent_description="Single agent for solo task execution",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = BatchedGridWorkflow(
        name="Single-Agent-Workflow",
        description="Testing single agent execution",
        agents=[agent],
        max_loops=1,
    )

    tasks = ["Analyze the current state of renewable energy"]
    result = workflow.run(tasks)

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1


def test_batched_grid_workflow_four_agents():
    """Test BatchedGridWorkflow with four agents in grid pattern"""
    agents = [
        Agent(
            agent_name=f"Grid-Agent-{i+1}",
            agent_description=f"Agent {i+1} for grid execution pattern",
            model_name="gpt-4o",
            max_loops=1,
        )
        for i in range(4)
    ]

    workflow = BatchedGridWorkflow(
        name="Four-Agent-Grid-Workflow",
        description="Testing grid execution with four agents",
        agents=agents,
        max_loops=2,
    )

    tasks = [
        "Evaluate market trends in technology",
        "Assess competitive landscape",
        "Analyze customer sentiment",
        "Review financial performance",
    ]
    results = workflow.run(tasks)

    assert results is not None
    assert isinstance(results, list)
    assert len(results) == 2  # max_loops=2
    for result in results:
        assert result is not None


def test_batched_grid_workflow_max_loops_validation():
    """Test that BatchedGridWorkflow validates max_loops parameter"""
    agent = Agent(
        agent_name="Validation-Agent",
        agent_description="Agent for validation testing",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Test with invalid max_loops (zero)
    with pytest.raises(ValueError) as exc_info:
        BatchedGridWorkflow(
            name="Invalid-Loops-Workflow",
            agents=[agent],
            max_loops=0,
        )
    assert "max_loops must be a positive integer" in str(
        exc_info.value
    )

    # Test with invalid max_loops (negative)
    with pytest.raises(ValueError) as exc_info:
        BatchedGridWorkflow(
            name="Invalid-Loops-Workflow",
            agents=[agent],
            max_loops=-5,
        )
    assert "max_loops must be a positive integer" in str(
        exc_info.value
    )


def test_batched_grid_workflow_step_method():
    """Test the step method of BatchedGridWorkflow"""
    agent1 = Agent(
        agent_name="Step-Agent-1",
        agent_description="First agent for step testing",
        model_name="gpt-4o",
        max_loops=1,
    )

    agent2 = Agent(
        agent_name="Step-Agent-2",
        agent_description="Second agent for step testing",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = BatchedGridWorkflow(
        name="Step-Test-Workflow",
        description="Testing step method",
        agents=[agent1, agent2],
        max_loops=1,
    )

    tasks = [
        "Generate a business plan outline",
        "Create a financial projection model",
    ]

    # Test single step execution
    step_result = workflow.step(tasks)

    assert step_result is not None
    assert isinstance(step_result, list)


def test_batched_grid_workflow_with_complex_tasks():
    """Test BatchedGridWorkflow with complex multi-step tasks"""
    # Create agents with specific expertise
    financial_agent = Agent(
        agent_name="Financial-Analyst",
        agent_description="Expert in financial analysis and forecasting",
        model_name="gpt-4o",
        max_loops=1,
    )

    technical_agent = Agent(
        agent_name="Technical-Expert",
        agent_description="Expert in technical analysis and evaluation",
        model_name="gpt-4o",
        max_loops=1,
    )

    strategic_agent = Agent(
        agent_name="Strategic-Planner",
        agent_description="Expert in strategic planning and execution",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = BatchedGridWorkflow(
        name="Complex-Analysis-Workflow",
        description="Workflow for complex multi-perspective analysis",
        agents=[financial_agent, technical_agent, strategic_agent],
        max_loops=2,
    )

    tasks = [
        "Perform a comprehensive financial analysis of a SaaS company with $10M ARR. "
        "Include revenue projections, burn rate analysis, and unit economics.",
        "Evaluate the technical architecture and scalability of a cloud-based platform. "
        "Assess security measures, infrastructure costs, and technical debt.",
        "Develop a strategic expansion plan for entering European markets. "
        "Consider regulatory requirements, competition, and go-to-market strategy.",
    ]

    results = workflow.run(tasks)

    assert results is not None
    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 3  # Three agents


def test_batched_grid_workflow_default_parameters():
    """Test BatchedGridWorkflow with default parameters"""
    agent = Agent(
        agent_name="Default-Test-Agent",
        agent_description="Agent for testing default parameters",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create workflow with mostly default parameters
    workflow = BatchedGridWorkflow(agents=[agent])

    # Verify default values
    assert workflow.name == "BatchedGridWorkflow"
    assert (
        workflow.description
        == "For every agent, run the task on a different task"
    )
    assert workflow.max_loops == 1
    assert workflow.id is not None


def test_batched_grid_workflow_custom_id_and_name():
    """Test BatchedGridWorkflow with custom ID and name"""
    agent = Agent(
        agent_name="Custom-Agent",
        agent_description="Agent for custom parameter testing",
        model_name="gpt-4o",
        max_loops=1,
    )

    custom_id = "custom-workflow-id-12345"
    custom_name = "My-Custom-Workflow"
    custom_description = "This is a custom workflow for testing"

    workflow = BatchedGridWorkflow(
        id=custom_id,
        name=custom_name,
        description=custom_description,
        agents=[agent],
        max_loops=1,
    )

    assert workflow.id == custom_id
    assert workflow.name == custom_name
    assert workflow.description == custom_description


def test_batched_grid_workflow_large_scale():
    """Test BatchedGridWorkflow with multiple agents for scalability"""
    # Create 6 agents to test larger scale operations
    agents = [
        Agent(
            agent_name=f"Scale-Agent-{i+1}",
            agent_description=f"Agent {i+1} for scalability testing",
            model_name="gpt-4o",
            max_loops=1,
        )
        for i in range(6)
    ]

    workflow = BatchedGridWorkflow(
        name="Large-Scale-Workflow",
        description="Testing workflow scalability",
        agents=agents,
        max_loops=1,
    )

    tasks = [
        "Task 1: Market research for technology sector",
        "Task 2: Competitor analysis in cloud computing",
        "Task 3: Customer needs assessment",
        "Task 4: Pricing strategy development",
        "Task 5: Risk assessment and mitigation",
        "Task 6: Implementation roadmap planning",
    ]

    results = workflow.run(tasks)

    assert results is not None
    assert isinstance(results, list)
    assert len(results) == 1


def test_batched_grid_workflow_sequential_runs():
    """Test running BatchedGridWorkflow multiple times sequentially"""
    agent1 = Agent(
        agent_name="Sequential-Agent-1",
        agent_description="Agent for sequential execution testing",
        model_name="gpt-4o",
        max_loops=1,
    )

    agent2 = Agent(
        agent_name="Sequential-Agent-2",
        agent_description="Agent for sequential execution testing",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = BatchedGridWorkflow(
        name="Sequential-Runs-Workflow",
        description="Testing multiple sequential runs",
        agents=[agent1, agent2],
        max_loops=1,
    )

    tasks_batch1 = [
        "Analyze Q1 business performance",
        "Review Q1 customer feedback",
    ]

    tasks_batch2 = [
        "Analyze Q2 business performance",
        "Review Q2 customer feedback",
    ]

    # Run workflow twice with different tasks
    results1 = workflow.run(tasks_batch1)
    results2 = workflow.run(tasks_batch2)

    assert results1 is not None
    assert results2 is not None
    assert isinstance(results1, list)
    assert isinstance(results2, list)


def test_batched_grid_workflow_specialized_agents():
    """Test BatchedGridWorkflow with highly specialized agents"""
    # Create agents with very specific roles
    seo_agent = Agent(
        agent_name="SEO-Specialist",
        agent_description="Expert in search engine optimization and content strategy",
        model_name="gpt-4o",
        max_loops=1,
    )

    content_agent = Agent(
        agent_name="Content-Creator",
        agent_description="Expert in content creation and copywriting",
        model_name="gpt-4o",
        max_loops=1,
    )

    social_media_agent = Agent(
        agent_name="Social-Media-Manager",
        agent_description="Expert in social media marketing and engagement",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = BatchedGridWorkflow(
        name="Marketing-Campaign-Workflow",
        description="Specialized workflow for marketing campaign execution",
        agents=[seo_agent, content_agent, social_media_agent],
        max_loops=1,
    )

    tasks = [
        "Develop an SEO strategy for a new e-commerce website selling sustainable products",
        "Create compelling product descriptions and blog content for eco-friendly products",
        "Design a social media campaign to promote sustainable living and green products",
    ]

    results = workflow.run(tasks)

    assert results is not None
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0] is not None


def test_batched_grid_workflow_max_loops_ten():
    """Test BatchedGridWorkflow with max_loops set to 10"""
    agent = Agent(
        agent_name="Ten-Loop-Agent",
        agent_description="Agent for testing 10 execution loops",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = BatchedGridWorkflow(
        name="Ten-Loop-Workflow",
        description="Testing workflow with 10 loops",
        agents=[agent],
        max_loops=10,
    )

    tasks = ["Iteratively refine a machine learning model concept"]

    results = workflow.run(tasks)

    assert results is not None
    assert isinstance(results, list)
    assert len(results) == 10  # Verify all 10 loops executed


def test_batched_grid_workflow_output_consistency():
    """Test that BatchedGridWorkflow produces consistent output structure"""
    agents = [
        Agent(
            agent_name=f"Consistency-Agent-{i+1}",
            agent_description=f"Agent {i+1} for output consistency testing",
            model_name="gpt-4o",
            max_loops=1,
        )
        for i in range(3)
    ]

    workflow = BatchedGridWorkflow(
        name="Output-Consistency-Workflow",
        description="Testing output structure consistency",
        agents=agents,
        max_loops=2,
    )

    tasks = [
        "Generate idea A",
        "Generate idea B",
        "Generate idea C",
    ]

    results = workflow.run(tasks)

    # Check output structure consistency
    assert isinstance(results, list)
    assert len(results) == 2  # max_loops=2

    for loop_result in results:
        assert loop_result is not None
        assert isinstance(loop_result, list)
        assert len(loop_result) == 3  # Three agents
