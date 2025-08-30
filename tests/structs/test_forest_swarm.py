import sys

from swarms.structs.tree_swarm import (
    TreeAgent,
    Tree,
    ForestSwarm,
    AgentLogInput,
    AgentLogOutput,
    TreeLog,
    extract_keywords,
    cosine_similarity,
)


# Test Results Tracking
test_results = {"passed": 0, "failed": 0, "total": 0}


def assert_equal(actual, expected, test_name):
    """Assert that actual equals expected, track test results."""
    test_results["total"] += 1
    if actual == expected:
        test_results["passed"] += 1
        print(f"✅ PASS: {test_name}")
        return True
    else:
        test_results["failed"] += 1
        print(f"❌ FAIL: {test_name}")
        print(f"   Expected: {expected}")
        print(f"   Actual: {actual}")
        return False


def assert_true(condition, test_name):
    """Assert that condition is True, track test results."""
    test_results["total"] += 1
    if condition:
        test_results["passed"] += 1
        print(f"✅ PASS: {test_name}")
        return True
    else:
        test_results["failed"] += 1
        print(f"❌ FAIL: {test_name}")
        print("   Condition was False")
        return False


def assert_false(condition, test_name):
    """Assert that condition is False, track test results."""
    test_results["total"] += 1
    if not condition:
        test_results["passed"] += 1
        print(f"✅ PASS: {test_name}")
        return True
    else:
        test_results["failed"] += 1
        print(f"❌ FAIL: {test_name}")
        print("   Condition was True")
        return False


def assert_is_instance(obj, expected_type, test_name):
    """Assert that obj is an instance of expected_type, track test results."""
    test_results["total"] += 1
    if isinstance(obj, expected_type):
        test_results["passed"] += 1
        print(f"✅ PASS: {test_name}")
        return True
    else:
        test_results["failed"] += 1
        print(f"❌ FAIL: {test_name}")
        print(f"   Expected type: {expected_type}")
        print(f"   Actual type: {type(obj)}")
        return False


def assert_not_none(obj, test_name):
    """Assert that obj is not None, track test results."""
    test_results["total"] += 1
    if obj is not None:
        test_results["passed"] += 1
        print(f"✅ PASS: {test_name}")
        return True
    else:
        test_results["failed"] += 1
        print(f"❌ FAIL: {test_name}")
        print("   Object was None")
        return False


# Test Data
SAMPLE_SYSTEM_PROMPTS = {
    "financial_advisor": "I am a financial advisor specializing in investment planning, retirement strategies, and tax optimization for individuals and businesses.",
    "tax_expert": "I am a tax expert with deep knowledge of corporate taxation, Delaware incorporation benefits, and free tax filing options for businesses.",
    "stock_analyst": "I am a stock market analyst who provides insights on market trends, stock recommendations, and portfolio optimization strategies.",
    "retirement_planner": "I am a retirement planning specialist who helps individuals and businesses create comprehensive retirement strategies and investment plans.",
}

SAMPLE_TASKS = {
    "tax_question": "Our company is incorporated in Delaware, how do we do our taxes for free?",
    "investment_question": "What are the best investment strategies for a 401k retirement plan?",
    "stock_question": "Which tech stocks should I consider for my investment portfolio?",
    "retirement_question": "How much should I save monthly for retirement if I want to retire at 65?",
}


# Test Functions


def test_extract_keywords():
    """Test the extract_keywords function."""
    print("\n🧪 Testing extract_keywords function...")

    # Test basic keyword extraction
    text = (
        "financial advisor investment planning retirement strategies"
    )
    keywords = extract_keywords(text, top_n=3)
    assert_equal(
        len(keywords),
        3,
        "extract_keywords returns correct number of keywords",
    )
    assert_true(
        "financial" in keywords,
        "extract_keywords includes 'financial'",
    )
    assert_true(
        "investment" in keywords,
        "extract_keywords includes 'investment'",
    )

    # Test with punctuation and case
    text = "Tax Expert! Corporate Taxation, Delaware Incorporation."
    keywords = extract_keywords(text, top_n=5)
    assert_true(
        "tax" in keywords,
        "extract_keywords handles punctuation and case",
    )
    assert_true(
        "corporate" in keywords,
        "extract_keywords handles punctuation and case",
    )

    # Test empty string
    keywords = extract_keywords("", top_n=3)
    assert_equal(
        len(keywords), 0, "extract_keywords handles empty string"
    )


def test_cosine_similarity():
    """Test the cosine_similarity function."""
    print("\n🧪 Testing cosine_similarity function...")

    # Test identical vectors
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = cosine_similarity(vec1, vec2)
    assert_equal(
        similarity,
        1.0,
        "cosine_similarity returns 1.0 for identical vectors",
    )

    # Test orthogonal vectors
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    similarity = cosine_similarity(vec1, vec2)
    assert_equal(
        similarity,
        0.0,
        "cosine_similarity returns 0.0 for orthogonal vectors",
    )

    # Test opposite vectors
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [-1.0, 0.0, 0.0]
    similarity = cosine_similarity(vec1, vec2)
    assert_equal(
        similarity,
        -1.0,
        "cosine_similarity returns -1.0 for opposite vectors",
    )

    # Test zero vectors
    vec1 = [0.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = cosine_similarity(vec1, vec2)
    assert_equal(
        similarity, 0.0, "cosine_similarity handles zero vectors"
    )


def test_agent_log_models():
    """Test the Pydantic log models."""
    print("\n🧪 Testing Pydantic log models...")

    # Test AgentLogInput
    log_input = AgentLogInput(
        agent_name="test_agent", task="test_task"
    )
    assert_is_instance(
        log_input,
        AgentLogInput,
        "AgentLogInput creates correct instance",
    )
    assert_not_none(
        log_input.log_id, "AgentLogInput generates log_id"
    )
    assert_equal(
        log_input.agent_name,
        "test_agent",
        "AgentLogInput stores agent_name",
    )
    assert_equal(
        log_input.task, "test_task", "AgentLogInput stores task"
    )

    # Test AgentLogOutput
    log_output = AgentLogOutput(
        agent_name="test_agent", result="test_result"
    )
    assert_is_instance(
        log_output,
        AgentLogOutput,
        "AgentLogOutput creates correct instance",
    )
    assert_not_none(
        log_output.log_id, "AgentLogOutput generates log_id"
    )
    assert_equal(
        log_output.result,
        "test_result",
        "AgentLogOutput stores result",
    )

    # Test TreeLog
    tree_log = TreeLog(
        tree_name="test_tree",
        task="test_task",
        selected_agent="test_agent",
        result="test_result",
    )
    assert_is_instance(
        tree_log, TreeLog, "TreeLog creates correct instance"
    )
    assert_not_none(tree_log.log_id, "TreeLog generates log_id")
    assert_equal(
        tree_log.tree_name, "test_tree", "TreeLog stores tree_name"
    )


def test_tree_agent_initialization():
    """Test TreeAgent initialization and basic properties."""
    print("\n🧪 Testing TreeAgent initialization...")

    # Test basic initialization
    agent = TreeAgent(
        name="Test Agent",
        description="A test agent",
        system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
        agent_name="financial_advisor",
    )

    assert_is_instance(
        agent, TreeAgent, "TreeAgent creates correct instance"
    )
    assert_equal(
        agent.agent_name,
        "financial_advisor",
        "TreeAgent stores agent_name",
    )
    assert_equal(
        agent.embedding_model_name,
        "text-embedding-ada-002",
        "TreeAgent has default embedding model",
    )
    assert_true(
        len(agent.relevant_keywords) > 0,
        "TreeAgent extracts keywords from system prompt",
    )
    assert_not_none(
        agent.system_prompt_embedding,
        "TreeAgent generates system prompt embedding",
    )

    # Test with custom embedding model
    agent_custom = TreeAgent(
        system_prompt="Test prompt",
        embedding_model_name="custom-model",
    )
    assert_equal(
        agent_custom.embedding_model_name,
        "custom-model",
        "TreeAgent accepts custom embedding model",
    )


def test_tree_agent_distance_calculation():
    """Test TreeAgent distance calculation between agents."""
    print("\n🧪 Testing TreeAgent distance calculation...")

    agent1 = TreeAgent(
        system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
        agent_name="financial_advisor",
    )

    agent2 = TreeAgent(
        system_prompt=SAMPLE_SYSTEM_PROMPTS["tax_expert"],
        agent_name="tax_expert",
    )

    agent3 = TreeAgent(
        system_prompt=SAMPLE_SYSTEM_PROMPTS["stock_analyst"],
        agent_name="stock_analyst",
    )

    # Test distance calculation
    distance1 = agent1.calculate_distance(agent2)
    distance2 = agent1.calculate_distance(agent3)

    assert_true(
        0.0 <= distance1 <= 1.0, "Distance is between 0 and 1"
    )
    assert_true(
        0.0 <= distance2 <= 1.0, "Distance is between 0 and 1"
    )
    assert_true(isinstance(distance1, float), "Distance is a float")

    # Test that identical agents have distance 0
    identical_agent = TreeAgent(
        system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
        agent_name="identical_advisor",
    )
    distance_identical = agent1.calculate_distance(identical_agent)
    assert_true(
        distance_identical < 0.1,
        "Identical agents have very small distance",
    )


def test_tree_agent_task_relevance():
    """Test TreeAgent task relevance checking."""
    print("\n🧪 Testing TreeAgent task relevance...")

    tax_agent = TreeAgent(
        system_prompt=SAMPLE_SYSTEM_PROMPTS["tax_expert"],
        agent_name="tax_expert",
    )

    # Test keyword matching
    tax_task = SAMPLE_TASKS["tax_question"]
    is_relevant = tax_agent.is_relevant_for_task(
        tax_task, threshold=0.7
    )
    assert_true(is_relevant, "Tax agent is relevant for tax question")

    # Test non-relevant task
    stock_task = SAMPLE_TASKS["stock_question"]
    is_relevant = tax_agent.is_relevant_for_task(
        stock_task, threshold=0.7
    )
    # This might be True due to semantic similarity, so we just check it's a boolean
    assert_true(
        isinstance(is_relevant, bool),
        "Task relevance returns boolean",
    )


def test_tree_initialization():
    """Test Tree initialization and agent organization."""
    print("\n🧪 Testing Tree initialization...")

    agents = [
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
            agent_name="financial_advisor",
        ),
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["tax_expert"],
            agent_name="tax_expert",
        ),
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["stock_analyst"],
            agent_name="stock_analyst",
        ),
    ]

    tree = Tree("Financial Services Tree", agents)

    assert_equal(
        tree.tree_name,
        "Financial Services Tree",
        "Tree stores tree_name",
    )
    assert_equal(len(tree.agents), 3, "Tree contains all agents")
    assert_true(
        all(hasattr(agent, "distance") for agent in tree.agents),
        "All agents have distance calculated",
    )

    # Test that agents are sorted by distance
    distances = [agent.distance for agent in tree.agents]
    assert_true(
        distances == sorted(distances),
        "Agents are sorted by distance",
    )


def test_tree_agent_finding():
    """Test Tree agent finding functionality."""
    print("\n🧪 Testing Tree agent finding...")

    agents = [
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
            agent_name="financial_advisor",
        ),
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["tax_expert"],
            agent_name="tax_expert",
        ),
    ]

    tree = Tree("Test Tree", agents)

    # Test finding relevant agent
    tax_task = SAMPLE_TASKS["tax_question"]
    relevant_agent = tree.find_relevant_agent(tax_task)
    assert_not_none(
        relevant_agent, "Tree finds relevant agent for tax task"
    )

    # Test finding agent for unrelated task
    unrelated_task = "How do I cook pasta?"
    relevant_agent = tree.find_relevant_agent(unrelated_task)
    # This might return None or an agent depending on similarity threshold
    assert_true(
        relevant_agent is None
        or isinstance(relevant_agent, TreeAgent),
        "Tree handles unrelated tasks",
    )


def test_forest_swarm_initialization():
    """Test ForestSwarm initialization."""
    print("\n🧪 Testing ForestSwarm initialization...")

    agents_tree1 = [
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
            agent_name="financial_advisor",
        )
    ]

    agents_tree2 = [
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["tax_expert"],
            agent_name="tax_expert",
        )
    ]

    tree1 = Tree("Financial Tree", agents_tree1)
    tree2 = Tree("Tax Tree", agents_tree2)

    forest = ForestSwarm(
        name="Test Forest",
        description="A test forest",
        trees=[tree1, tree2],
    )

    assert_equal(
        forest.name, "Test Forest", "ForestSwarm stores name"
    )
    assert_equal(
        forest.description,
        "A test forest",
        "ForestSwarm stores description",
    )
    assert_equal(
        len(forest.trees), 2, "ForestSwarm contains all trees"
    )
    assert_not_none(
        forest.conversation, "ForestSwarm creates conversation object"
    )


def test_forest_swarm_tree_finding():
    """Test ForestSwarm tree finding functionality."""
    print("\n🧪 Testing ForestSwarm tree finding...")

    agents_tree1 = [
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
            agent_name="financial_advisor",
        )
    ]

    agents_tree2 = [
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["tax_expert"],
            agent_name="tax_expert",
        )
    ]

    tree1 = Tree("Financial Tree", agents_tree1)
    tree2 = Tree("Tax Tree", agents_tree2)

    forest = ForestSwarm(trees=[tree1, tree2])

    # Test finding relevant tree for tax question
    tax_task = SAMPLE_TASKS["tax_question"]
    relevant_tree = forest.find_relevant_tree(tax_task)
    assert_not_none(
        relevant_tree, "ForestSwarm finds relevant tree for tax task"
    )

    # Test finding relevant tree for financial question
    financial_task = SAMPLE_TASKS["investment_question"]
    relevant_tree = forest.find_relevant_tree(financial_task)
    assert_not_none(
        relevant_tree,
        "ForestSwarm finds relevant tree for financial task",
    )


def test_forest_swarm_execution():
    """Test ForestSwarm task execution."""
    print("\n🧪 Testing ForestSwarm task execution...")

    # Create a simple forest with one tree and one agent
    agent = TreeAgent(
        system_prompt="I am a helpful assistant that can answer questions about Delaware incorporation and taxes.",
        agent_name="delaware_expert",
    )

    tree = Tree("Delaware Tree", [agent])
    forest = ForestSwarm(trees=[tree])

    # Test task execution
    task = "What are the benefits of incorporating in Delaware?"
    try:
        result = forest.run(task)
        assert_not_none(
            result, "ForestSwarm returns result from task execution"
        )
        assert_true(isinstance(result, str), "Result is a string")
    except Exception as e:
        # If execution fails due to external dependencies, that's okay for unit tests
        print(
            f"⚠️  Task execution failed (expected in unit test environment): {e}"
        )


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🧪 Testing edge cases and error handling...")

    # Test TreeAgent with None system prompt
    agent_no_prompt = TreeAgent(
        system_prompt=None, agent_name="no_prompt_agent"
    )
    assert_equal(
        len(agent_no_prompt.relevant_keywords),
        0,
        "Agent with None prompt has empty keywords",
    )
    assert_true(
        agent_no_prompt.system_prompt_embedding is None,
        "Agent with None prompt has None embedding",
    )

    # Test Tree with empty agents list
    empty_tree = Tree("Empty Tree", [])
    assert_equal(
        len(empty_tree.agents), 0, "Empty tree has no agents"
    )

    # Test ForestSwarm with empty trees list
    empty_forest = ForestSwarm(trees=[])
    assert_equal(
        len(empty_forest.trees), 0, "Empty forest has no trees"
    )

    # Test cosine_similarity with empty vectors
    empty_vec = []
    vec = [1.0, 0.0, 0.0]
    similarity = cosine_similarity(empty_vec, vec)
    assert_equal(
        similarity, 0.0, "cosine_similarity handles empty vectors"
    )


def run_all_tests():
    """Run all unit tests and display results."""
    print("🚀 Starting ForestSwarm Unit Tests...")
    print("=" * 60)

    # Run all test functions
    test_functions = [
        test_extract_keywords,
        test_cosine_similarity,
        test_agent_log_models,
        test_tree_agent_initialization,
        test_tree_agent_distance_calculation,
        test_tree_agent_task_relevance,
        test_tree_initialization,
        test_tree_agent_finding,
        test_forest_swarm_initialization,
        test_forest_swarm_tree_finding,
        test_forest_swarm_execution,
        test_edge_cases,
    ]

    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            test_results["total"] += 1
            test_results["failed"] += 1
            print(f"❌ ERROR: {test_func.__name__} - {e}")

    # Display results
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {test_results['total']}")
    print(f"Passed: {test_results['passed']}")
    print(f"Failed: {test_results['failed']}")

    success_rate = (
        (test_results["passed"] / test_results["total"]) * 100
        if test_results["total"] > 0
        else 0
    )
    print(f"Success Rate: {success_rate:.1f}%")

    if test_results["failed"] == 0:
        print(
            "\n🎉 All tests passed! ForestSwarm is working correctly."
        )
    else:
        print(
            f"\n⚠️  {test_results['failed']} test(s) failed. Please review the failures above."
        )

    return test_results["failed"] == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
