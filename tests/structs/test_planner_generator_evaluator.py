import os
import tempfile


from swarms import Agent
from swarms.structs.planner_generator_evaluator import (
    EvaluationReport,
    HarnessResult,
    PlannerGeneratorEvaluator,
    StepContract,
)


def test_pge_basic_initialization():
    """Test basic PlannerGeneratorEvaluator initialization"""
    harness = PlannerGeneratorEvaluator(
        name="Test-PGE-Harness",
        description="Test harness for PGE",
        model_name="gpt-4.1",
        max_steps=3,
        max_retries_per_step=2,
    )

    assert harness.name == "Test-PGE-Harness"
    assert harness.max_steps == 3
    assert harness.max_retries_per_step == 2
    assert harness.planner_agent is not None
    assert harness.generator_agent is not None
    assert harness.evaluator_agent is not None
    assert harness.planner_agent.agent_name == "PGE-Planner"
    assert harness.generator_agent.agent_name == "PGE-Generator"
    assert harness.evaluator_agent.agent_name == "PGE-Evaluator"
    assert harness.last_result is None


def test_pge_custom_model_names():
    """Test PGE with different models per agent"""
    harness = PlannerGeneratorEvaluator(
        model_name="gpt-4.1",
        planner_model_name="gpt-4.1",
        generator_model_name="gpt-4.1",
        evaluator_model_name="gpt-4.1",
    )

    assert harness.planner_model_name == "gpt-4.1"
    assert harness.generator_model_name == "gpt-4.1"
    assert harness.evaluator_model_name == "gpt-4.1"


def test_pge_with_custom_agents():
    """Test PGE accepts pre-configured custom agents with tools"""
    custom_generator = Agent(
        agent_name="Custom-Generator",
        agent_description="Generator with custom config",
        model_name="gpt-4.1",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    custom_evaluator = Agent(
        agent_name="Custom-Evaluator",
        agent_description="Evaluator with custom config",
        model_name="gpt-4.1",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    harness = PlannerGeneratorEvaluator(
        model_name="gpt-4.1",
        generator_agent=custom_generator,
        evaluator_agent=custom_evaluator,
    )

    assert harness.generator_agent.agent_name == "Custom-Generator"
    assert harness.evaluator_agent.agent_name == "Custom-Evaluator"
    assert harness.planner_agent.agent_name == "PGE-Planner"


def test_pge_error_handling():
    """Test PGE error handling for invalid config"""
    try:
        PlannerGeneratorEvaluator(max_steps=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "max_steps" in str(e)

    try:
        PlannerGeneratorEvaluator(max_retries_per_step=-1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "max_retries" in str(e)

    try:
        PlannerGeneratorEvaluator(model_name="")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "model_name" in str(e)


def test_pge_run_validates_task():
    """Test that run() rejects empty tasks"""
    harness = PlannerGeneratorEvaluator(
        model_name="gpt-4.1",
    )

    try:
        harness.run("")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "non-empty" in str(e)


def test_pge_shared_state_file(tmp_path):
    """Test shared state file initialization and appending"""
    shared_state_path = str(tmp_path / "state.md")

    harness = PlannerGeneratorEvaluator(
        model_name="gpt-4.1",
        shared_state_path=shared_state_path,
    )

    harness._initialize_shared_state("Test prompt")
    assert os.path.exists(shared_state_path)

    content = harness._read_shared_state()
    assert "Test prompt" in content
    assert "PGE Harness Shared State" in content

    harness._append_to_shared_state(
        "TEST SECTION", "Test content here"
    )
    content = harness._read_shared_state()
    assert "TEST SECTION" in content
    assert "Test content here" in content


def test_pge_extract_step_count():
    """Test step count extraction from plan text"""
    harness = PlannerGeneratorEvaluator(
        model_name="gpt-4.1",
        max_steps=10,
    )

    assert (
        harness._extract_step_count("Step 1: A\nStep 2: B\nStep 3: C")
        == 3
    )

    assert (
        harness._extract_step_count(
            "1. First\n2. Second\n3. Third\n4. Fourth"
        )
        == 4
    )

    harness.max_steps = 2
    assert (
        harness._extract_step_count(
            "Step 1: A\nStep 2: B\nStep 3: C\nStep 4: D"
        )
        == 2
    )


def test_pge_extract_thresholds():
    """Test threshold extraction from plan's evaluation criteria table"""
    harness = PlannerGeneratorEvaluator(
        model_name="gpt-4.1",
    )

    plan = """
| Criterion | Weight | Description | Threshold |
|-----------|--------|-------------|-----------|
| accuracy | high | correctness | 7 |
| clarity | standard | readability | 6 |
| depth | low | thoroughness | 5 |
"""
    thresholds = harness._extract_thresholds(plan)
    assert thresholds["accuracy"] == 7.0
    assert thresholds["clarity"] == 6.0
    assert thresholds["depth"] == 5.0


def test_pge_parse_evaluation_pass():
    """Test evaluation parsing when all criteria pass"""
    harness = PlannerGeneratorEvaluator(
        model_name="gpt-4.1",
    )

    raw = """
## EVALUATION: Step 1

### Per-Criterion Scores:
| Criterion | Score (1-10) | Threshold | Status |
|-----------|-------------|-----------|--------|
| accuracy | 8 | 7 | PASS |
| clarity | 7 | 6 | PASS |

### Overall Status: PASS
"""
    report = harness._parse_evaluation(
        1, raw, {"accuracy": 7, "clarity": 6}
    )
    assert report.passed is True
    assert report.criterion_scores["accuracy"] == 8.0
    assert report.criterion_scores["clarity"] == 7.0


def test_pge_parse_evaluation_fail():
    """Test evaluation parsing when a criterion fails"""
    harness = PlannerGeneratorEvaluator(
        model_name="gpt-4.1",
    )

    raw = """
| Criterion | Score (1-10) | Threshold | Status |
|-----------|-------------|-----------|--------|
| accuracy | 5 | 7 | FAIL |
| clarity | 8 | 6 | PASS |

### Overall Status: FAIL
"""
    report = harness._parse_evaluation(
        1, raw, {"accuracy": 7, "clarity": 6}
    )
    assert report.passed is False
    assert report.criterion_scores["accuracy"] == 5.0


def test_pge_supporting_types():
    """Test StepContract, EvaluationReport, HarnessResult serialization"""
    sc = StepContract(step_number=1, title="Research", approved=True)
    d = sc.to_dict()
    assert d["step_number"] == 1
    assert d["approved"] is True

    er = EvaluationReport(
        step_number=2,
        criterion_scores={"accuracy": 9.0},
        passed=True,
    )
    d = er.to_dict()
    assert d["passed"] is True
    assert d["criterion_scores"]["accuracy"] == 9.0

    hr = HarnessResult(
        output_path="/tmp/test.md",
        total_steps_completed=3,
        total_retries=1,
    )
    d = hr.to_dict()
    assert d["total_steps_completed"] == 3


def test_pge_execution():
    """Test full PGE harness execution with real API calls"""
    with tempfile.TemporaryDirectory() as tmpdir:
        shared_state_path = os.path.join(tmpdir, "shared_state.md")

        harness = PlannerGeneratorEvaluator(
            name="Test-Execution-Harness",
            model_name="gpt-4.1",
            max_steps=2,
            max_retries_per_step=1,
            shared_state_path=shared_state_path,
            output_type="final",
            verbose=True,
        )

        result = harness.run(
            "Write a 3-paragraph explanation of why the sky is blue"
        )

        # Shared state file was created with content
        assert os.path.exists(shared_state_path)
        with open(shared_state_path) as f:
            state_content = f.read()
        assert "User Prompt" in state_content
        assert "PLANNER OUTPUT" in state_content
        assert len(state_content) > 500

        # Result is non-empty
        assert result is not None
        assert len(str(result)) > 0

        # HarnessResult was stored
        assert harness.last_result is not None
        assert harness.last_result.output_path == shared_state_path
        assert harness.last_result.plan != ""
        assert len(harness.last_result.step_logs) > 0
        assert harness.last_result.total_duration > 0

        # Conversation has entries from all three agents
        history = harness.conversation.to_dict()
        roles = {msg["role"] for msg in history}
        assert "User" in roles
        assert "PGE-Planner" in roles
        assert "PGE-Generator" in roles
        assert "PGE-Evaluator" in roles
