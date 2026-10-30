import tempfile
import pytest
from swarms import PlannerGeneratorEvaluator


class DummyAgent:
    """Stub agent for offline unit testing without API keys."""

    def __init__(self, name: str, reply: str):
        self.agent_name = name
        self.reply = reply

    def run(self, task=None, **kwargs):
        return self.reply


def test_extract_step_count_standard_format():
    plan = """## PLAN

### Deliverables
1. A backend
2. A frontend

### Steps
Step 1: Part 1
Step 2: Part 2
Step 3: Part 3
Step 4: Part 4
Step 5: Part 5

### Evaluation Criteria
| Criterion | Weight | Description | Threshold |
|-----------|--------|-------------|-----------|
| correctness | high | works | 7 |
"""
    pge = PlannerGeneratorEvaluator(
        working_directory=tempfile.mkdtemp(),
        planner_agent=DummyAgent("Planner", plan),
        generator_agent=DummyAgent("Generator", "done"),
        evaluator_agent=DummyAgent("Evaluator", "PASS"),
    )
    assert pge._extract_step_count(plan) == 5


def test_extract_step_count_markdown_h4_headings():
    plan = """## PLAN

### Deliverables
1. A backend
2. A frontend

### Steps
#### Step 1: Part 1
Build part 1.
#### Step 2: Part 2
Build part 2.
#### Step 3: Part 3
Build part 3.
#### Step 4: Part 4
Build part 4.
#### Step 5: Part 5
Build part 5.

### Evaluation Criteria
| Criterion | Weight | Description | Threshold |
|-----------|--------|-------------|-----------|
| correctness | high | works | 7 |
"""
    pge = PlannerGeneratorEvaluator(
        working_directory=tempfile.mkdtemp(),
        planner_agent=DummyAgent("Planner", plan),
        generator_agent=DummyAgent("Generator", "done"),
        evaluator_agent=DummyAgent("Evaluator", "PASS"),
    )
    assert pge._extract_step_count(plan) == 5


def test_extract_step_count_markdown_h3_headings():
    plan = """## PLAN

### Deliverables
1. A backend
2. A frontend

### Steps
### Step 1: Part 1
### Step 2: Part 2
### Step 3: Part 3
### Step 4: Part 4
### Step 5: Part 5

### Evaluation Criteria
| Criterion | Weight | Description | Threshold |
|-----------|--------|-------------|-----------|
| correctness | high | works | 7 |
"""
    pge = PlannerGeneratorEvaluator(
        working_directory=tempfile.mkdtemp(),
        planner_agent=DummyAgent("Planner", plan),
        generator_agent=DummyAgent("Generator", "done"),
        evaluator_agent=DummyAgent("Evaluator", "PASS"),
    )
    assert pge._extract_step_count(plan) == 5


def test_extract_step_count_bold_markers():
    plan = """## PLAN

### Deliverables
1. A backend
2. A frontend

### Steps
**Step 1:** Part 1
**Step 2:** Part 2
**Step 3:** Part 3
**Step 4:** Part 4
**Step 5:** Part 5

### Evaluation Criteria
| Criterion | Weight | Description | Threshold |
|-----------|--------|-------------|-----------|
| correctness | high | works | 7 |
"""
    pge = PlannerGeneratorEvaluator(
        working_directory=tempfile.mkdtemp(),
        planner_agent=DummyAgent("Planner", plan),
        generator_agent=DummyAgent("Generator", "done"),
        evaluator_agent=DummyAgent("Evaluator", "PASS"),
    )
    assert pge._extract_step_count(plan) == 5


def test_extract_step_count_bulleted_bold_markers():
    plan = """## PLAN

### Deliverables
1. A backend
2. A frontend

### Steps
- **Step 1:** Part 1
- **Step 2:** Part 2
- **Step 3:** Part 3
- **Step 4:** Part 4

### Evaluation Criteria
| Criterion | Weight | Description | Threshold |
|-----------|--------|-------------|-----------|
| correctness | high | works | 7 |
"""
    pge = PlannerGeneratorEvaluator(
        working_directory=tempfile.mkdtemp(),
        planner_agent=DummyAgent("Planner", plan),
        generator_agent=DummyAgent("Generator", "done"),
        evaluator_agent=DummyAgent("Evaluator", "PASS"),
    )
    assert pge._extract_step_count(plan) == 4


def test_extract_step_count_numbered_list_with_more_deliverables():
    plan = """## PLAN

### Deliverables
1. A backend
2. A frontend
3. Database
4. Cache
5. Queue
6. Monitoring
7. Documentation

### Steps
1. **Part 1**: Setup project
2. **Part 2**: Build backend
3. **Part 3**: Build frontend
4. **Part 4**: Write tests
5. **Part 5**: Deploy

### Evaluation Criteria
| Criterion | Weight | Description | Threshold |
|-----------|--------|-------------|-----------|
| correctness | high | works | 7 |
"""
    pge = PlannerGeneratorEvaluator(
        working_directory=tempfile.mkdtemp(),
        planner_agent=DummyAgent("Planner", plan),
        generator_agent=DummyAgent("Generator", "done"),
        evaluator_agent=DummyAgent("Evaluator", "PASS"),
    )
    assert pge._extract_step_count(plan) == 5


def test_extract_step_count_capped_at_max_steps():
    plan = """## PLAN

### Steps
Step 1: Part 1
Step 2: Part 2
Step 3: Part 3
Step 4: Part 4
Step 5: Part 5
Step 6: Part 6
Step 7: Part 7
"""
    pge = PlannerGeneratorEvaluator(
        working_directory=tempfile.mkdtemp(),
        planner_agent=DummyAgent("Planner", plan),
        generator_agent=DummyAgent("Generator", "done"),
        evaluator_agent=DummyAgent("Evaluator", "PASS"),
        max_steps=4,
    )
    assert pge._extract_step_count(plan) == 4


def test_extract_step_count_unparseable_defaults_to_3():
    plan = "A plan with no step markers or numbered lists whatsoever."
    pge = PlannerGeneratorEvaluator(
        working_directory=tempfile.mkdtemp(),
        planner_agent=DummyAgent("Planner", plan),
        generator_agent=DummyAgent("Generator", "done"),
        evaluator_agent=DummyAgent("Evaluator", "PASS"),
    )
    assert pge._extract_step_count(plan) == 3


def test_run_executes_all_steps_with_markdown_headings():
    steps = "\n".join(
        f"#### Step {n}: Part {n}\nBuild part {n}." for n in range(1, 6)
    )
    plan = f"""## PLAN

### Deliverables
1. A backend
2. A frontend

### Steps
{steps}

### Evaluation Criteria
| Criterion | Weight | Description | Threshold |
|-----------|--------|-------------|-----------|
| correctness | high | works | 7 |
"""
    evaluation = """### Per-Criterion Scores:
| Criterion | Score (1-10) | Threshold | Status |
|-----------|-------------|-----------|--------|
| correctness | 9 | 7 | PASS |
"""
    pge = PlannerGeneratorEvaluator(
        working_directory=tempfile.mkdtemp(),
        planner_agent=DummyAgent("Planner", plan),
        generator_agent=DummyAgent("Generator", "done"),
        evaluator_agent=DummyAgent("Evaluator", evaluation),
    )
    pge.run("Build application")
    assert len(pge.last_result.step_logs) == 5
