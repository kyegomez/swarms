from swarms.structs.matrix_swarm import MatrixSwarm, AgentOutput
from swarms import Agent


def create_test_matrix(rows: int, cols: int) -> MatrixSwarm:
    """Helper function to create a test agent matrix"""
    agents = [
        [
            Agent(
                agent_name=f"TestAgent-{i}-{j}",
                model_name="gpt-4o",
                system_prompt="Test prompt",
            )
            for j in range(cols)
        ]
        for i in range(rows)
    ]
    return MatrixSwarm(agents)


def test_init():
    """Test MatrixSwarm initialization"""
    # Test valid initialization
    matrix = create_test_matrix(2, 2)
    assert isinstance(matrix, MatrixSwarm)
    assert len(matrix.agents) == 2
    assert len(matrix.agents[0]) == 2

    # Test invalid initialization
    try:
        MatrixSwarm([[1, 2], [3, 4]])  # Non-agent elements
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    try:
        MatrixSwarm([])  # Empty matrix
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_transpose():
    """Test matrix transpose operation"""
    matrix = create_test_matrix(2, 3)
    transposed = matrix.transpose()

    assert len(transposed.agents) == 3  # Original cols become rows
    assert len(transposed.agents[0]) == 2  # Original rows become cols

    # Verify agent positions
    for i in range(2):
        for j in range(3):
            assert (
                matrix.agents[i][j].agent_name
                == transposed.agents[j][i].agent_name
            )


def test_add():
    """Test matrix addition"""
    matrix1 = create_test_matrix(2, 2)
    matrix2 = create_test_matrix(2, 2)

    result = matrix1.add(matrix2)
    assert len(result.agents) == 2
    assert len(result.agents[0]) == 2

    # Test incompatible dimensions
    matrix3 = create_test_matrix(2, 3)
    try:
        matrix1.add(matrix3)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_scalar_multiply():
    """Test scalar multiplication"""
    matrix = create_test_matrix(2, 2)
    scalar = 3
    result = matrix.scalar_multiply(scalar)

    assert len(result.agents) == 2
    assert len(result.agents[0]) == 2 * scalar

    # Verify agent duplication
    for i in range(len(result.agents)):
        for j in range(0, len(result.agents[0]), scalar):
            original_agent = matrix.agents[i][j // scalar]
            for k in range(scalar):
                assert (
                    result.agents[i][j + k].agent_name
                    == original_agent.agent_name
                )


def test_multiply():
    """Test matrix multiplication"""
    matrix1 = create_test_matrix(2, 3)
    matrix2 = create_test_matrix(3, 2)
    inputs = ["test query 1", "test query 2"]

    result = matrix1.multiply(matrix2, inputs)
    assert len(result) == 2  # Number of rows in first matrix
    assert len(result[0]) == 2  # Number of columns in second matrix

    # Verify output structure
    for row in result:
        for output in row:
            assert isinstance(output, AgentOutput)
            assert isinstance(output.input_query, str)
            assert isinstance(output.metadata, dict)


def test_subtract():
    """Test matrix subtraction"""
    matrix1 = create_test_matrix(2, 2)
    matrix2 = create_test_matrix(2, 2)

    result = matrix1.subtract(matrix2)
    assert len(result.agents) == 2
    assert len(result.agents[0]) == 2


def test_identity():
    """Test identity matrix creation"""
    matrix = create_test_matrix(3, 3)
    identity = matrix.identity(3)

    assert len(identity.agents) == 3
    assert len(identity.agents[0]) == 3

    # Verify diagonal elements are from original matrix
    for i in range(3):
        assert (
            identity.agents[i][i].agent_name
            == matrix.agents[i][i].agent_name
        )

        # Verify non-diagonal elements are zero agents
        for j in range(3):
            if i != j:
                assert identity.agents[i][j].agent_name.startswith(
                    "Zero-Agent"
                )


def test_determinant():
    """Test determinant calculation"""
    # Test 1x1 matrix
    matrix1 = create_test_matrix(1, 1)
    det1 = matrix1.determinant()
    assert det1 is not None

    # Test 2x2 matrix
    matrix2 = create_test_matrix(2, 2)
    det2 = matrix2.determinant()
    assert det2 is not None

    # Test non-square matrix
    matrix3 = create_test_matrix(2, 3)
    try:
        matrix3.determinant()
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_save_to_file(tmp_path):
    """Test saving matrix to file"""
    import os

    matrix = create_test_matrix(2, 2)
    file_path = os.path.join(tmp_path, "test_matrix.json")

    matrix.save_to_file(file_path)
    assert os.path.exists(file_path)

    # Verify file contents
    import json

    with open(file_path, "r") as f:
        data = json.load(f)
        assert "agents" in data
        assert "outputs" in data
        assert len(data["agents"]) == 2
        assert len(data["agents"][0]) == 2


def run_all_tests():
    """Run all test functions"""
    test_functions = [
        test_init,
        test_transpose,
        test_add,
        test_scalar_multiply,
        test_multiply,
        test_subtract,
        test_identity,
        test_determinant,
    ]

    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__} passed")
        except AssertionError as e:
            print(f"❌ {test_func.__name__} failed: {str(e)}")
        except Exception as e:
            print(
                f"❌ {test_func.__name__} failed with exception: {str(e)}"
            )


if __name__ == "__main__":
    run_all_tests()
