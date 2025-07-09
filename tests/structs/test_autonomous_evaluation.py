"""
Tests for the autonomous evaluation feature in AutoSwarmBuilder.

This test suite validates the iterative improvement functionality and evaluation system.
"""

import pytest
from unittest.mock import patch, MagicMock

from swarms.structs.auto_swarm_builder import (
    AutoSwarmBuilder,
    IterativeImprovementConfig,
    EvaluationResult,
)


class TestAutonomousEvaluation:
    """Test suite for autonomous evaluation features"""

    def test_iterative_improvement_config_defaults(self):
        """Test default configuration values"""
        config = IterativeImprovementConfig()
        
        assert config.max_iterations == 3
        assert config.improvement_threshold == 0.1
        assert "accuracy" in config.evaluation_dimensions
        assert "helpfulness" in config.evaluation_dimensions
        assert config.use_judge_agent is True
        assert config.store_all_iterations is True

    def test_iterative_improvement_config_custom(self):
        """Test custom configuration values"""
        config = IterativeImprovementConfig(
            max_iterations=5,
            improvement_threshold=0.2,
            evaluation_dimensions=["accuracy", "coherence"],
            use_judge_agent=False,
            store_all_iterations=False,
        )
        
        assert config.max_iterations == 5
        assert config.improvement_threshold == 0.2
        assert len(config.evaluation_dimensions) == 2
        assert config.use_judge_agent is False
        assert config.store_all_iterations is False

    def test_evaluation_result_model(self):
        """Test EvaluationResult model creation and validation"""
        result = EvaluationResult(
            iteration=1,
            task="Test task",
            output="Test output",
            evaluation_scores={"accuracy": 0.8, "helpfulness": 0.7},
            feedback="Good performance",
            strengths=["Clear response"],
            weaknesses=["Could be more detailed"],
            suggestions=["Add more examples"],
        )
        
        assert result.iteration == 1
        assert result.task == "Test task"
        assert result.evaluation_scores["accuracy"] == 0.8
        assert len(result.strengths) == 1
        assert len(result.weaknesses) == 1
        assert len(result.suggestions) == 1

    def test_auto_swarm_builder_init_with_evaluation(self):
        """Test AutoSwarmBuilder initialization with evaluation enabled"""
        config = IterativeImprovementConfig(max_iterations=2)
        
        with patch('swarms.structs.auto_swarm_builder.CouncilAsAJudge'):
            with patch('swarms.structs.auto_swarm_builder.Agent'):
                swarm = AutoSwarmBuilder(
                    name="TestSwarm",
                    description="Test swarm with evaluation",
                    enable_evaluation=True,
                    evaluation_config=config,
                )
                
                assert swarm.enable_evaluation is True
                assert swarm.evaluation_config.max_iterations == 2
                assert swarm.current_iteration == 0
                assert len(swarm.evaluation_history) == 0

    def test_auto_swarm_builder_init_without_evaluation(self):
        """Test AutoSwarmBuilder initialization with evaluation disabled"""
        swarm = AutoSwarmBuilder(
            name="TestSwarm",
            description="Test swarm without evaluation",
            enable_evaluation=False,
        )
        
        assert swarm.enable_evaluation is False
        assert swarm.current_iteration == 0
        assert len(swarm.evaluation_history) == 0

    @patch('swarms.structs.auto_swarm_builder.CouncilAsAJudge')
    @patch('swarms.structs.auto_swarm_builder.Agent')
    def test_evaluation_system_initialization(self, mock_agent, mock_council):
        """Test evaluation system initialization"""
        config = IterativeImprovementConfig()
        
        swarm = AutoSwarmBuilder(
            name="TestSwarm",
            enable_evaluation=True,
            evaluation_config=config,
        )
        
        # Verify CouncilAsAJudge was initialized
        mock_council.assert_called_once()
        
        # Verify improvement agent was created
        mock_agent.assert_called_once()
        assert mock_agent.call_args[1]['agent_name'] == 'ImprovementStrategist'

    def test_get_improvement_agent_prompt(self):
        """Test improvement agent prompt generation"""
        swarm = AutoSwarmBuilder(enable_evaluation=False)
        prompt = swarm._get_improvement_agent_prompt()
        
        assert "improvement strategist" in prompt.lower()
        assert "evaluation feedback" in prompt.lower()
        assert "recommendations" in prompt.lower()

    def test_extract_dimension_score(self):
        """Test dimension score extraction from feedback"""
        swarm = AutoSwarmBuilder(enable_evaluation=False)
        
        # Test positive feedback
        positive_feedback = "The response is accurate and helpful"
        accuracy_score = swarm._extract_dimension_score(positive_feedback, "accuracy")
        helpfulness_score = swarm._extract_dimension_score(positive_feedback, "helpfulness")
        
        assert accuracy_score > 0.5
        assert helpfulness_score > 0.5
        
        # Test negative feedback
        negative_feedback = "The response is inaccurate and unhelpful"
        accuracy_score_neg = swarm._extract_dimension_score(negative_feedback, "accuracy")
        helpfulness_score_neg = swarm._extract_dimension_score(negative_feedback, "helpfulness")
        
        assert accuracy_score_neg < 0.5
        assert helpfulness_score_neg < 0.5
        
        # Test neutral feedback
        neutral_feedback = "The response exists"
        neutral_score = swarm._extract_dimension_score(neutral_feedback, "accuracy")
        assert neutral_score == 0.5

    def test_parse_feedback(self):
        """Test feedback parsing into strengths, weaknesses, and suggestions"""
        swarm = AutoSwarmBuilder(enable_evaluation=False)
        
        feedback = """
        The response shows good understanding of the topic.
        However, there are some issues with clarity.
        I suggest adding more examples to improve comprehension.
        The strength is in the factual accuracy.
        The weakness is the lack of structure.
        Recommend reorganizing the content.
        """
        
        strengths, weaknesses, suggestions = swarm._parse_feedback(feedback)
        
        assert len(strengths) > 0
        assert len(weaknesses) > 0
        assert len(suggestions) > 0

    def test_get_evaluation_results(self):
        """Test getting evaluation results"""
        swarm = AutoSwarmBuilder(enable_evaluation=False)
        
        # Initially empty
        assert len(swarm.get_evaluation_results()) == 0
        
        # Add mock evaluation result
        mock_result = EvaluationResult(
            iteration=1,
            task="test",
            output="test output",
            evaluation_scores={"accuracy": 0.8},
            feedback="good",
            strengths=["clear"],
            weaknesses=["brief"],
            suggestions=["expand"],
        )
        swarm.evaluation_history.append(mock_result)
        
        results = swarm.get_evaluation_results()
        assert len(results) == 1
        assert results[0].iteration == 1

    def test_get_best_iteration(self):
        """Test getting the best performing iteration"""
        swarm = AutoSwarmBuilder(enable_evaluation=False)
        
        # No iterations initially
        assert swarm.get_best_iteration() is None
        
        # Add mock evaluation results
        result1 = EvaluationResult(
            iteration=1,
            task="test",
            output="output1",
            evaluation_scores={"accuracy": 0.6, "helpfulness": 0.5},
            feedback="ok",
            strengths=[],
            weaknesses=[],
            suggestions=[],
        )
        
        result2 = EvaluationResult(
            iteration=2,
            task="test",
            output="output2", 
            evaluation_scores={"accuracy": 0.8, "helpfulness": 0.7},
            feedback="better",
            strengths=[],
            weaknesses=[],
            suggestions=[],
        )
        
        swarm.evaluation_history.extend([result1, result2])
        
        best = swarm.get_best_iteration()
        assert best.iteration == 2  # Second iteration has higher scores

    @patch('swarms.structs.auto_swarm_builder.OpenAIFunctionCaller')
    def test_create_agents_with_feedback_first_iteration(self, mock_function_caller):
        """Test agent creation for first iteration (no feedback)"""
        swarm = AutoSwarmBuilder(enable_evaluation=False)
        
        # Mock the function caller
        mock_instance = MagicMock()
        mock_function_caller.return_value = mock_instance
        mock_instance.run.return_value.model_dump.return_value = {
            "agents": [
                {
                    "name": "TestAgent",
                    "description": "A test agent",
                    "system_prompt": "You are a test agent"
                }
            ]
        }
        
        # Mock build_agent method
        with patch.object(swarm, 'build_agent') as mock_build_agent:
            mock_agent = MagicMock()
            mock_build_agent.return_value = mock_agent
            
            agents = swarm.create_agents_with_feedback("test task")
            
            assert len(agents) == 1
            mock_build_agent.assert_called_once()

    def test_run_single_iteration_mode(self):
        """Test running in single iteration mode (evaluation disabled)"""
        swarm = AutoSwarmBuilder(enable_evaluation=False)
        
        with patch.object(swarm, 'create_agents') as mock_create:
            with patch.object(swarm, 'initialize_swarm_router') as mock_router:
                mock_create.return_value = []
                mock_router.return_value = "test result"
                
                result = swarm.run("test task")
                
                assert result == "test result"
                mock_create.assert_called_once_with("test task")
                mock_router.assert_called_once()


class TestEvaluationIntegration:
    """Integration tests for the evaluation system"""

    @patch('swarms.structs.auto_swarm_builder.CouncilAsAJudge')
    @patch('swarms.structs.auto_swarm_builder.Agent')
    @patch('swarms.structs.auto_swarm_builder.OpenAIFunctionCaller')
    def test_evaluation_workflow(self, mock_function_caller, mock_agent, mock_council):
        """Test the complete evaluation workflow"""
        # Setup mocks
        mock_council_instance = MagicMock()
        mock_council.return_value = mock_council_instance
        mock_council_instance.run.return_value = "Evaluation feedback"
        
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.run.return_value = "Improvement suggestions"
        
        mock_function_caller_instance = MagicMock()
        mock_function_caller.return_value = mock_function_caller_instance
        mock_function_caller_instance.run.return_value.model_dump.return_value = {
            "agents": [
                {
                    "name": "TestAgent",
                    "description": "Test",
                    "system_prompt": "Test prompt"
                }
            ]
        }
        
        # Configure swarm
        config = IterativeImprovementConfig(max_iterations=1)
        swarm = AutoSwarmBuilder(
            name="TestSwarm",
            enable_evaluation=True,
            evaluation_config=config,
        )
        
        # Mock additional methods
        with patch.object(swarm, 'build_agent') as mock_build:
            with patch.object(swarm, 'initialize_swarm_router') as mock_router:
                mock_build.return_value = mock_agent_instance
                mock_router.return_value = "Task output"
                
                # Run the swarm
                result = swarm.run("test task")
                
                # Verify evaluation was performed
                assert len(swarm.evaluation_history) == 1
                assert result == "Task output"


if __name__ == "__main__":
    pytest.main([__file__])