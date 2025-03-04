"""
AIScientistFramework: A multi-agent system for AI co-scientist based on
"Towards an AI co-scientist" research paper.
Implements hypothesis generation, review, ranking, and evolution using a tournament approach.
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from swarms import Agent
from swarms.structs.conversation import Conversation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Define the possible roles for agents in the AI co-scientist system."""
    GENERATION = "generation"
    REFLECTION = "reflection"
    RANKING = "ranking"
    EVOLUTION = "evolution"
    META_REVIEW = "meta_review"
    PROXIMITY = "proximity"
    SUPERVISOR = "supervisor"
    TOURNAMENT = "tournament"


@dataclass
class Hypothesis:
    """
    Represents a research hypothesis.

    Attributes:
        text (str): The text of the hypothesis.
        elo_rating (int): Elo rating for ranking (initially 1200).
        reviews (List[Dict]): List of review feedback for the hypothesis.
        score (float): Overall score based on reviews (0.0-1.0).
        similarity_cluster_id (Optional[str]): ID of the similarity cluster.
        evolution_history (List[str]): History of evolutions for this hypothesis.
        generation_timestamp (float): When the hypothesis was generated.
        win_count (int): Number of tournament wins.
        loss_count (int): Number of tournament losses.
    """
    text: str
    elo_rating: int = 1200
    reviews: List[Dict] = field(default_factory=list)
    score: float = 0.0
    similarity_cluster_id: Optional[str] = None
    evolution_history: List[str] = field(default_factory=list)
    generation_timestamp: float = field(default_factory=time.time)
    win_count: int = 0
    loss_count: int = 0

    def update_elo(self, opponent_elo: int, win: bool, k_factor: int = 32) -> None:
        """
        Update the Elo rating based on a tournament match outcome.

        Args:
            opponent_elo (int): The Elo rating of the opponent.
            win (bool): Whether this hypothesis won the match.
            k_factor (int): K-factor for Elo calculation, controlling update magnitude.
        """
        expected_score = 1 / (1 + 10 ** ((opponent_elo - self.elo_rating) / 400))
        actual_score = 1.0 if win else 0.0
        self.elo_rating += int(k_factor * (actual_score - expected_score))

        # Update win/loss count
        if win:
            self.win_count += 1
        else:
            self.loss_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert the hypothesis to a dictionary representation."""
        return {
            "text": self.text,
            "elo_rating": self.elo_rating,
            "score": self.score,
            "reviews": self.reviews,
            "similarity_cluster_id": self.similarity_cluster_id,
            "evolution_history": self.evolution_history,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "total_matches": self.win_count + self.loss_count,
            "win_rate": round(self.win_count / max(1, (self.win_count + self.loss_count)) * 100, 2)
        }


class AIScientistFramework:
    """
    A multi-agent system framework for AI co-scientist, designed to generate
    and refine research hypotheses using tournament-based evolution.

    Attributes:
        model_name (str): Name of the LLM model to use for agents.
        max_iterations (int): Maximum number of iterations for the research workflow.
        base_path (Path): Base path for saving agent states.
        verbose (bool): Enable verbose logging.
        conversation (Conversation): Tracks the conversation history.
        hypotheses (List[Hypothesis]): List to store generated hypotheses.
        tournament_size (int): Number of hypotheses to include in each tournament round.
        hypotheses_per_generation (int): Number of hypotheses to generate initially.
        evolution_top_k (int): Number of top hypotheses to evolve in each iteration.
    """

    def __init__(
        self,
        model_name: str = "gemini/gemini-2.0-flash",
        max_iterations: int = 3,
        base_path: Optional[str] = None,
        verbose: bool = False,
        tournament_size: int = 8,
        hypotheses_per_generation: int = 10,
        evolution_top_k: int = 3,
    ):
        """Initialize the AIScientistFramework system with configuration parameters."""
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.base_path = Path(base_path) if base_path else Path("./ai_coscientist_states")
        self.base_path.mkdir(exist_ok=True, parents=True)
        self.verbose = verbose
        self.conversation = Conversation()
        self.hypotheses: List[Hypothesis] = []

        # Tournament and evolution parameters
        self.tournament_size = tournament_size
        self.hypotheses_per_generation = hypotheses_per_generation
        self.evolution_top_k = evolution_top_k

        # Execution metrics
        self.start_time = None
        self.execution_metrics = {
            "total_time": 0,
            "hypothesis_count": 0,
            "reviews_count": 0,
            "tournaments_count": 0,
            "evolutions_count": 0,
            "agent_execution_times": {}
        }

        # Initialize agents
        self._init_agents()

    def _init_agents(self) -> None:
        """Initialize all specialized agents with their roles and prompts."""
        self.generation_agent = Agent(
            agent_name="HypothesisGenerator",
            system_prompt=self._get_generation_agent_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "generation_agent_state.json"),
            verbose=self.verbose,
        )
        self.reflection_agent = Agent(
            agent_name="HypothesisReflector",
            system_prompt=self._get_reflection_agent_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "reflection_agent_state.json"),
            verbose=self.verbose,
        )
        self.ranking_agent = Agent(
            agent_name="HypothesisRanker",
            system_prompt=self._get_ranking_agent_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "ranking_agent_state.json"),
            verbose=self.verbose,
        )
        self.evolution_agent = Agent(
            agent_name="HypothesisEvolver",
            system_prompt=self._get_evolution_agent_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "evolution_agent_state.json"),
            verbose=self.verbose,
        )
        self.meta_review_agent = Agent(
            agent_name="MetaReviewer",
            system_prompt=self._get_meta_review_agent_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "meta_review_agent_state.json"),
            verbose=self.verbose,
        )
        self.proximity_agent = Agent(
            agent_name="ProximityAnalyzer",
            system_prompt=self._get_proximity_agent_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "proximity_agent_state.json"),
            verbose=self.verbose,
        )
        self.tournament_agent = Agent(
            agent_name="TournamentJudge",
            system_prompt=self._get_tournament_agent_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "tournament_agent_state.json"),
            verbose=self.verbose,
        )
        self.supervisor_agent = Agent(
            agent_name="Supervisor",
            system_prompt=self._get_supervisor_agent_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "supervisor_agent_state.json"),
            verbose=self.verbose,
        )

    def _get_generation_agent_prompt(self) -> str:
        """Prompt for the Hypothesis Generation Agent."""
        return """You are a Hypothesis Generation Agent in an AI Co-scientist framework.
Your role is to generate novel and relevant research hypotheses based on a given research goal.

Consider current scientific literature and knowledge in the domain.
Focus on generating hypotheses that are:
- Novel and original
- Relevant to the research goal
- Potentially testable and falsifiable
- Scientifically sound
- Specific and well-defined

Each hypothesis should:
1. Challenge existing assumptions or extend current knowledge in the field
2. Be formulated as a clear statement that can be tested
3. Identify potential variables and relationships
4. Consider practical implications and significance
5. Balance ambition with feasibility

Output your hypotheses in JSON format. Provide a list of hypotheses, each with a clear and concise text description,
and brief justification explaining why it's novel and significant.

Example JSON Output:
{
  "hypotheses": [
    {
      "text": "Hypothesis text 1",
      "justification": "Brief explanation of novelty, significance, and scientific rationale"
    },
    {
      "text": "Hypothesis text 2",
      "justification": "Brief explanation of novelty, significance, and scientific rationale"
    },
    ...
  ]
}
"""

    def _get_reflection_agent_prompt(self) -> str:
        """Prompt for the Hypothesis Reflection Agent (Reviewer)."""
        return """You are a Hypothesis Reflection Agent, acting as a scientific peer reviewer.
Your task is to review and critique research hypotheses for correctness, novelty, quality, and potential safety/ethical concerns.

For each hypothesis, evaluate it based on the following criteria:
- Scientific Soundness (1-5): Is the hypothesis scientifically plausible and consistent with existing knowledge?
- Novelty (1-5): Does the hypothesis propose something new or original?
- Relevance (1-5): Is the hypothesis relevant to the stated research goal?
- Testability (1-5): Can the hypothesis be tested or investigated using scientific methods?
- Clarity (1-5): Is the hypothesis clearly and concisely stated?
- Potential Impact (1-5): If validated, what is the potential scientific or practical impact?
- Safety/Ethical Concerns: Are there any potential safety or ethical issues associated with investigating this hypothesis?

Provide a detailed review for each criterion, with specific feedback on strengths and weaknesses.
For the overall score, use a scale from 0.0 to 1.0, where:
- 0.0-0.2: Poor (multiple serious flaws)
- 0.2-0.4: Fair (notable deficiencies requiring substantial revision)
- 0.4-0.6: Good (promising but needs revisions)
- 0.6-0.8: Very Good (minor revisions needed)
- 0.8-1.0: Excellent (minimal or no revisions needed)

Output your review in JSON format:

Example JSON Output (for a single hypothesis):
{
  "hypothesis_text": "The hypothesis being reviewed",
  "review_summary": "Overall summary of the review",
  "scores": {
    "scientific_soundness": 4,
    "novelty": 3,
    "relevance": 5,
    "testability": 4,
    "clarity": 5,
    "potential_impact": 4
  },
  "safety_ethical_concerns": "Specific concerns or 'None identified'",
  "detailed_feedback": {
    "scientific_soundness": "Specific feedback on scientific soundness",
    "novelty": "Specific feedback on novelty",
    "relevance": "Specific feedback on relevance",
    "testability": "Specific feedback on testability",
    "clarity": "Specific feedback on clarity",
    "potential_impact": "Specific feedback on potential impact"
  },
  "constructive_feedback": "Specific suggestions for improvement",
  "overall_score": 0.8
}
"""

    def _get_ranking_agent_prompt(self) -> str:
        """Prompt for the Hypothesis Ranking Agent."""
        return """You are a Hypothesis Ranking Agent. Your role is to rank a set of research hypotheses based on their review scores and other relevant criteria.

Rank the hypotheses from highest to lowest quality based on:
1. The overall scores provided by the Reflection Agents
2. The detailed feedback for each criterion
3. Scientific merit and potential impact
4. Novelty and originality
5. Feasibility of testing and verification

For each hypothesis, calculate a composite ranking score that synthesizes these factors.
Consider not just the average scores, but also the distribution across criteria - a hypothesis with consistently good scores
might be preferable to one with extremely high scores in some areas but poor scores in others.

Output the ranked hypotheses in JSON format, ordered from highest to lowest rank. Include the hypothesis text,
overall score, and a brief explanation for each ranking decision.

Example JSON Output:
{
  "ranked_hypotheses": [
    {
      "text": "Hypothesis text 1",
      "overall_score": 0.9,
      "ranking_explanation": "Ranked highest due to exceptional novelty, strong scientific soundness, and high testability"
    },
    {
      "text": "Hypothesis text 2",
      "overall_score": 0.85,
      "ranking_explanation": "Strong overall but ranked below hypothesis 1 due to slightly lower novelty"
    },
    ...
  ]
}
"""

    def _get_evolution_agent_prompt(self) -> str:
        """Prompt for the Hypothesis Evolution Agent (Refiner)."""
        return """You are a Hypothesis Evolution Agent. Your task is to refine and improve the top-ranked research hypotheses based on the reviews and meta-review insights.

For each hypothesis, carefully analyze the review feedback, meta-review insights, and then apply the following approaches to refine the hypothesis:

1. Enhance clarity and precision:
   - Eliminate ambiguous language
   - Ensure clear definition of variables and relationships
   - Improve the logical structure

2. Strengthen scientific soundness:
   - Address any identified theoretical weaknesses
   - Ensure alignment with established scientific principles
   - Incorporate relevant background knowledge

3. Increase novelty and originality:
   - Identify opportunities to introduce more innovative elements
   - Consider unconventional perspectives or approaches

4. Improve testability:
   - Make the hypothesis more amenable to empirical investigation
   - Consider specific experimental designs or methodologies
   - Ensure falsifiability

5. Address safety/ethical concerns:
   - Integrate ethical considerations
   - Propose safeguards or limitations when necessary

6. Consider hybridization:
   - Identify complementary hypotheses that could be combined
   - Merge strengths from multiple hypotheses when beneficial

7. Simplify when appropriate:
   - Remove unnecessary complexity
   - Focus on the most promising and impactful aspects

Output the refined hypotheses in JSON format, including the original text, the refined text, a summary of changes made, and justifications for each significant modification:

Example JSON Output (for a single hypothesis):
{
  "original_hypothesis_text": "Original hypothesis text",
  "refined_hypothesis_text": "Refined hypothesis text",
  "refinement_summary": "Summary of overall changes and improvements",
  "specific_refinements": [
    {
      "aspect": "clarity",
      "change": "Specific change made",
      "justification": "Reason for this modification"
    },
    {
      "aspect": "scientific_soundness",
      "change": "Specific change made",
      "justification": "Reason for this modification"
    },
    ...
  ]
}
"""

    def _get_meta_review_agent_prompt(self) -> str:
        """Prompt for the Meta-Review Agent."""
        return """You are a Meta-Review Agent. Your role is to synthesize insights from all the reviews of the research hypotheses.

Analyze all the reviews provided by the Reflection Agents across multiple hypotheses. Your goal is to:

1. Identify recurring patterns, themes, and trends:
   - Common strengths across hypotheses
   - Common weaknesses or limitations
   - Recurring feedback themes from reviewers

2. Evaluate the hypothesis generation and review process:
   - Areas where the generation process could be improved
   - Potential gaps in the review criteria or approach
   - Consistency and quality of reviews

3. Provide strategic guidance for hypothesis refinement:
   - High-level directions for improving hypothesis quality
   - Specific areas where the evolution agent should focus
   - Potential new directions or perspectives to explore

4. Assess the overall research direction:
   - Alignment with the original research goal
   - Potential for scientific impact
   - Most promising avenues for further exploration

5. Identify potential connections:
   - Relationships between different hypotheses
   - Possibilities for synthesizing complementary ideas
   - Cross-cutting themes or approaches

Output your meta-review insights and recommendations in JSON format:

Example JSON Output:
{
  "meta_review_summary": "Overall summary of meta-review analysis",
  "recurring_themes": [
    {
      "theme": "Theme 1",
      "description": "Detailed description of the theme",
      "frequency": "Number or percentage of hypotheses showing this theme"
    },
    ...
  ],
  "strengths": [
    "Common strength 1 identified across hypotheses",
    "Common strength 2 identified across hypotheses",
    ...
  ],
  "weaknesses": [
    "Common weakness 1 identified across hypotheses",
    "Common weakness 2 identified across hypotheses",
    ...
  ],
  "process_assessment": {
    "generation_process": "Assessment of hypothesis generation process",
    "review_process": "Assessment of review process",
    "evolution_process": "Assessment of hypothesis evolution process"
  },
  "strategic_recommendations": [
    {
      "focus_area": "Area for improvement",
      "recommendation": "Specific recommendation",
      "justification": "Reasoning behind this recommendation"
    },
    ...
  ],
  "potential_connections": [
    {
      "related_hypotheses": ["Hypothesis 1", "Hypothesis 2"],
      "connection_type": "Type of relationship (complementary, contradictory, etc.)",
      "synthesis_opportunity": "Potential for combining or relating these hypotheses"
    },
    ...
  ]
}
"""

    def _get_proximity_agent_prompt(self) -> str:
        """Prompt for the Proximity Agent (Similarity Analysis)."""
        return """You are a Proximity Agent, focused on analyzing the similarity between research hypotheses.

Your task is to identify hypotheses that are semantically similar or redundant to maintain diversity in the hypothesis pool.
This helps in clustering related hypotheses and de-duplicating similar ones to ensure diversity in the generated set.

For each hypothesis, analyze:
1. Core scientific concepts and principles involved
2. Key variables and relationships being examined
3. Underlying assumptions and theoretical frameworks
4. Methodological approaches suggested or implied
5. Potential applications or implications

Based on these factors, identify clusters of hypotheses that are conceptually related or address similar research questions.
Assign each hypothesis to a cluster, and give each cluster a descriptive name that captures its unifying theme.

For each cluster, identify:
- The central theme or concept
- The distinguishing features between hypotheses within the cluster
- The degree of similarity/redundancy between hypotheses (high, medium, low)
- Potential for synthesis or combination within the cluster

Output your findings in JSON format:

Example JSON Output:
{
  "similarity_clusters": [
    {
      "cluster_id": "cluster-1",
      "cluster_name": "Descriptive name for this cluster",
      "central_theme": "Brief description of the unifying concept",
      "similar_hypotheses": [
        {"text": "Hypothesis text A", "similarity_degree": "high"},
        {"text": "Hypothesis text B", "similarity_degree": "medium"},
        ...
      ],
      "synthesis_potential": "Analysis of whether hypotheses in this cluster could be combined effectively"
    },
    {
      "cluster_id": "cluster-2",
      "cluster_name": "Descriptive name for this cluster",
      "central_theme": "Brief description of the unifying concept",
      "similar_hypotheses": [
        {"text": "Hypothesis text C", "similarity_degree": "high"},
        {"text": "Hypothesis text D", "similarity_degree": "medium"},
        ...
      ],
      "synthesis_potential": "Analysis of whether hypotheses in this cluster could be combined effectively"
    },
    ...
  ],
  "diversity_assessment": "Overall assessment of the diversity of the hypothesis set",
  "redundancy_assessment": "Overall assessment of redundancy in the hypothesis set"
}
"""

    def _get_tournament_agent_prompt(self) -> str:
        """Prompt for the Tournament Agent (for pairwise hypothesis comparison)."""
        return """You are a Tournament Judge Agent in an AI Co-scientist framework. Your role is to evaluate pairs of research hypotheses and determine which one is superior for addressing the given research goal.

For each pair of hypotheses, carefully analyze and compare them based on the following criteria:
1. Scientific Soundness: Which hypothesis is more scientifically plausible and consistent with existing knowledge?
2. Novelty and Originality: Which hypothesis proposes more innovative or original ideas?
3. Relevance to Research Goal: Which hypothesis is more directly relevant to the stated research goal?
4. Testability and Falsifiability: Which hypothesis can be more rigorously tested or falsified?
5. Clarity and Precision: Which hypothesis is more clearly and precisely formulated?
6. Potential Impact: Which hypothesis, if validated, would have greater scientific or practical impact?
7. Feasibility: Which hypothesis could be investigated with available or reasonable resources?

Make a clear decision on which hypothesis wins the comparison based on these criteria.
Provide a detailed justification for your decision, explaining the specific strengths that led to the winning hypothesis
and weaknesses of the losing hypothesis.

Output your tournament judgment in JSON format:

Example JSON Output:
{
  "research_goal": "The research goal being addressed",
  "hypothesis_a": "Text of the first hypothesis",
  "hypothesis_b": "Text of the second hypothesis",
  "winner": "a or b (just the letter)",
  "judgment_explanation": {
    "scientific_soundness_comparison": "Comparison of scientific soundness between hypotheses",
    "novelty_comparison": "Comparison of novelty between hypotheses",
    "relevance_comparison": "Comparison of relevance between hypotheses",
    "testability_comparison": "Comparison of testability between hypotheses",
    "clarity_comparison": "Comparison of clarity between hypotheses",
    "impact_comparison": "Comparison of potential impact between hypotheses",
    "feasibility_comparison": "Comparison of feasibility between hypotheses"
  },
  "decision_summary": "Concise summary of why the winner was selected",
  "confidence_level": "High, Medium, or Low (how confident you are in this judgment)"
}
"""

    def _get_supervisor_agent_prompt(self) -> str:
        """Prompt for the Supervisor Agent (manages the overall workflow)."""
        return """You are a Supervisor Agent in an AI Co-scientist framework. Your role is to oversee the entire hypothesis generation and refinement workflow, ensuring coordination between specialized agents and optimizing the system's performance.

Your responsibilities include:

1. Research Plan Configuration:
   - Parse the scientist's research goal and preferences
   - Configure an appropriate research plan
   - Set parameters for the hypothesis generation and refinement process

2. Task Management:
   - Assign tasks to specialized agents
   - Determine resource allocation for different phases
   - Monitor progress and adjust task priorities

3. Quality Control:
   - Evaluate the outputs of each agent
   - Ensure adherence to scientific standards
   - Identify areas where agent performance can be improved

4. Workflow Optimization:
   - Identify bottlenecks in the research process
   - Suggest adjustments to the workflow
   - Balance exploration and exploitation

5. Synthesis and Integration:
   - Combine insights from different agents
   - Ensure coherence across the research pipeline
   - Integrate feedback from the scientist

Provide your guidance and management decisions in JSON format:

Example JSON Output:
{
  "research_goal_analysis": {
    "goal_summary": "Concise restatement of the research goal",
    "key_areas": ["Key area 1", "Key area 2", ...],
    "constraints_identified": ["Constraint 1", "Constraint 2", ...],
    "success_criteria": ["Criterion 1", "Criterion 2", ...]
  },
  "workflow_plan": {
    "generation_phase": {
      "focus_areas": ["Area 1", "Area 2", ...],
      "diversity_targets": "Description of diversity targets for hypotheses",
      "quantity_target": "Target number of hypotheses to generate"
    },
    "review_phase": {
      "critical_criteria": ["Criterion 1", "Criterion 2", ...],
      "review_depth": "Depth of review required"
    },
    "ranking_phase": {
      "ranking_approach": "Description of ranking approach",
      "selection_criteria": ["Criterion 1", "Criterion 2", ...]
    },
    "evolution_phase": {
      "refinement_priorities": ["Priority 1", "Priority 2", ...],
      "iteration_strategy": "Description of iteration strategy"
    }
  },
  "performance_assessment": {
    "current_status": "Assessment of current workflow status",
    "bottlenecks_identified": ["Bottleneck 1", "Bottleneck 2", ...],
    "agent_performance": {
      "generation_agent": "Assessment of generation agent performance",
      "reflection_agent": "Assessment of reflection agent performance",
      "ranking_agent": "Assessment of ranking agent performance",
      "evolution_agent": "Assessment of evolution agent performance",
      "proximity_agent": "Assessment of proximity agent performance",
      "meta_review_agent": "Assessment of meta-review agent performance"
    }
  },
  "adjustment_recommendations": [
    {
      "aspect": "Aspect to adjust",
      "adjustment": "Description of adjustment",
      "justification": "Reasoning behind this adjustment"
    },
    ...
  ],
  "output_preparation": {
    "hypothesis_selection_strategy": "Strategy for selecting final hypotheses",
    "presentation_format": "Format for presenting results to scientist",
    "key_insights_to_highlight": ["Insight 1", "Insight 2", ...]
  }
}
"""

    def _safely_parse_json(self, json_str: str) -> Dict[str, Any]:
        """Safely parse JSON string, handling potential errors."""
        try:
            # First try direct JSON parsing
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSONDecodeError: {e}. Attempting to extract JSON from text.")
            try:
                # Look for JSON-like structure within the text
                import re
                json_match = re.search(r"\{.*\}", json_str, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    logger.warning("No JSON found within text.")
                    return {"content": json_str, "error": "Failed to parse JSON, no JSON found in text."}
            except Exception as ex:
                logger.error(f"Error extracting JSON: {ex}")
                return {"content": json_str, "error": f"Failed to parse JSON: {ex}"}
        except Exception as e:
            logger.error(f"Unexpected error parsing JSON: {e}")
            return {"content": json_str, "error": f"Unexpected JSON parse error: {e}"}

    def _time_execution(self, agent_name: str, start_time: float) -> None:
        """Track execution time for an agent."""
        execution_time = time.time() - start_time

        if agent_name not in self.execution_metrics["agent_execution_times"]:
            self.execution_metrics["agent_execution_times"][agent_name] = {
                "total_time": 0,
                "calls": 0,
                "avg_time": 0
            }

        self.execution_metrics["agent_execution_times"][agent_name]["total_time"] += execution_time
        self.execution_metrics["agent_execution_times"][agent_name]["calls"] += 1
        self.execution_metrics["agent_execution_times"][agent_name]["avg_time"] = (
            self.execution_metrics["agent_execution_times"][agent_name]["total_time"] /
            self.execution_metrics["agent_execution_times"][agent_name]["calls"]
        )

    def _run_generation_phase(self, research_goal: str) -> List[Hypothesis]:
        """Run the hypothesis generation phase."""
        start_time = time.time()

        # Get research plan from supervisor
        supervisor_input = {
            "task": "plan_research",
            "research_goal": research_goal,
            "phase": "generation",
            "parameters": {
                "hypotheses_count": self.hypotheses_per_generation,
                "diversity_target": "high"
            }
        }
        supervisor_response = self.supervisor_agent.run(json.dumps(supervisor_input))
        self.conversation.add(role=self.supervisor_agent.agent_name, content=supervisor_response)
        supervisor_data = self._safely_parse_json(supervisor_response)

        # Run generation agent with supervisor guidance
        generation_input = {
            "research_goal": research_goal,
            "supervisor_guidance": supervisor_data,
            "required_hypotheses_count": self.hypotheses_per_generation
        }
        generation_response = self.generation_agent.run(json.dumps(generation_input))
        self.conversation.add(role=self.generation_agent.agent_name, content=generation_response)

        generation_data = self._safely_parse_json(generation_response)
        initial_hypotheses_data = generation_data.get("hypotheses", [])

        if not initial_hypotheses_data:
            logger.warning("Generation Agent returned no hypotheses. Using fallback generation.")
            # Fallback to simpler generation prompt
            fallback_input = {"research_goal": research_goal, "count": self.hypotheses_per_generation}
            fallback_response = self.generation_agent.run(json.dumps(fallback_input))
            fallback_data = self._safely_parse_json(fallback_response)
            initial_hypotheses_data = fallback_data.get("hypotheses", [])

            if not initial_hypotheses_data:
                raise ValueError("Generation Agent failed to generate hypotheses even with fallback.")

        # Convert to Hypothesis objects
        hypotheses = []
        for hy_data in initial_hypotheses_data:
            if isinstance(hy_data, dict) and "text" in hy_data:
                hypothesis_text = hy_data["text"]
            else:
                hypothesis_text = str(hy_data)

            hypotheses.append(Hypothesis(text=hypothesis_text))

        self._time_execution("generation", start_time)
        self.execution_metrics["hypothesis_count"] += len(hypotheses)
        logger.info(f"Generated {len(hypotheses)} initial hypotheses.")
        return hypotheses

    def _run_reflection_phase(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Run the hypothesis reflection (review) phase."""
        start_time = time.time()
        reviewed_hypotheses = []
        for hypothesis in hypotheses:
            review_input = {"hypothesis_text": hypothesis.text}
            review_response = self.reflection_agent.run(json.dumps(review_input))
            self.conversation.add(role=self.reflection_agent.agent_name, content=review_response)
            review_data = self._safely_parse_json(review_response)

            if review_data and "overall_score" in review_data:
                overall_score = review_data.get("overall_score", 0.0)
                hypothesis.score = float(overall_score)
                hypothesis.reviews.append(review_data)  # Store full review data
                reviewed_hypotheses.append(hypothesis)
            else:
                logger.warning(f"No valid review score found for hypothesis: {hypothesis.text}. Review data: {review_data}")
                reviewed_hypotheses.append(hypothesis) # Keep hypothesis even if review fails but log warning

        self._time_execution("reflection", start_time)
        self.execution_metrics["reviews_count"] += len(reviewed_hypotheses)
        logger.info(f"Hypotheses reviewed. Total reviews: {len(reviewed_hypotheses)}.")
        return reviewed_hypotheses

    def _run_ranking_phase(self, reviewed_hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Run the hypothesis ranking phase."""
        start_time = time.time()
        ranking_input = [{"text": h.text, "overall_score": h.score} for h in reviewed_hypotheses]
        ranking_response = self.ranking_agent.run(json.dumps({"hypotheses_for_ranking": ranking_input}))
        self.conversation.add(role=self.ranking_agent.agent_name, content=ranking_response)
        ranking_data = self._safely_parse_json(ranking_response)
        ranked_hypothesis_data = ranking_data.get("ranked_hypotheses", [])

        ranked_hypotheses = []
        hypothesis_map = {h.text: h for h in reviewed_hypotheses} # For efficient lookup
        for ranked_hy_data in ranked_hypothesis_data:
            hypothesis_text = ranked_hy_data.get("text")
            if hypothesis_text and hypothesis_text in hypothesis_map:
                ranked_hypotheses.append(hypothesis_map[hypothesis_text])
            else:
                logger.warning(f"Ranked hypothesis data missing text or text not found in original hypotheses.")

        self._time_execution("ranking", start_time)
        logger.info("Hypotheses ranked.")
        return ranked_hypotheses

    def _run_evolution_phase(self, top_hypotheses: List[Hypothesis], meta_review_data: Dict) -> List[Hypothesis]:
        """Run the hypothesis evolution phase."""
        start_time = time.time()
        evolved_hypotheses = []
        for hypothesis in top_hypotheses:
            evolution_input = {
                "original_hypothesis_text": hypothesis.text,
                "review_feedback": hypothesis.reviews[-1] if hypothesis.reviews else {}, # Use latest review
                "meta_review_insights": meta_review_data
            }
            evolution_response = self.evolution_agent.run(json.dumps(evolution_input))
            self.conversation.add(role=self.evolution_agent.agent_name, content=evolution_response)
            evolution_data = self._safely_parse_json(evolution_response)
            refined_hypothesis_text = evolution_data.get("refined_hypothesis_text")

            if refined_hypothesis_text:
                hypothesis.text = refined_hypothesis_text
                hypothesis.evolution_history.append(evolution_data.get("refinement_summary", "No summary")) # Track evolution
                evolved_hypotheses.append(hypothesis)
                logger.info(f"Hypothesis evolved: {hypothesis.text[:50]}...")
            else:
                evolved_hypotheses.append(hypothesis) # Keep original if no refinement
                logger.warning(f"Hypothesis evolution failed or returned no refined text for: {hypothesis.text[:50]}...")

        self._time_execution("evolution", start_time)
        self.execution_metrics["evolutions_count"] += len(evolved_hypotheses)
        logger.info("Hypotheses evolved.")
        return evolved_hypotheses

    def _run_meta_review_phase(self, reviewed_hypotheses: List[Hypothesis]) -> Dict:
        """Run the meta-review phase to synthesize insights from reviews."""
        start_time = time.time()
        all_reviews_for_meta = [h.reviews[-1] if h.reviews else {} for h in reviewed_hypotheses] # Get latest reviews
        meta_review_response = self.meta_review_agent.run(json.dumps({"reviews": all_reviews_for_meta}))
        self.conversation.add(role=self.meta_review_agent.agent_name, content=meta_review_response)
        meta_review_data = self._safely_parse_json(meta_review_response)
        self._time_execution("meta_review", start_time)
        logger.info("Meta-review completed.")
        return meta_review_data

    def _run_proximity_analysis_phase(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Run proximity analysis to cluster similar hypotheses."""
        start_time = time.time()
        proximity_response = self.proximity_agent.run(json.dumps({"hypotheses_texts": [h.text for h in hypotheses]}))
        self.conversation.add(role=self.proximity_agent.agent_name, content=proximity_response)
        proximity_data = self._safely_parse_json(proximity_response)
        similarity_clusters = proximity_data.get("similarity_clusters", [])

        # Assign cluster IDs to hypotheses
        for cluster in similarity_clusters:
            cluster_id = cluster.get("cluster_id", "no_cluster_id")
            for hy_text_data in cluster.get("similar_hypotheses", []): # Expecting list of dicts with "text" key
                hy_text = hy_text_data.get("text") if isinstance(hy_text_data, dict) else hy_text_data # Handle different formats
                if hy_text:
                    for hy in self.hypotheses:
                        if hy.text == hy_text:
                            hy.similarity_cluster_id = cluster_id
                            break # Hypothesis found, move to next
        self._time_execution("proximity_analysis", start_time)
        logger.info("Proximity analysis completed and clusters assigned.")
        return hypotheses

    def _run_tournament_phase(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Run tournament selection and Elo rating update."""
        start_time = time.time()
        tournament_rounds = len(hypotheses) * 3  # Example: 3 rounds per hypothesis, adjust as needed
        k_factor = 24 # Adjust K-factor to control Elo update speed

        for round_num in range(tournament_rounds):
            if len(hypotheses) < 2:
                logger.warning("Not enough hypotheses for a tournament round.")
                break # Need at least two hypotheses to run a tournament

            # Randomly select two different hypotheses for a match
            h1, h2 = random.sample(hypotheses, 2)
            if h1 == h2: # Ensure they are different (though random.sample should handle this)
                continue

            tournament_input = {
                "research_goal": "Compare hypotheses for tournament", # General goal context
                "hypothesis_a": h1.text,
                "hypothesis_b": h2.text
            }
            tournament_response = self.tournament_agent.run(json.dumps(tournament_input))
            self.conversation.add(role=self.tournament_agent.agent_name, content=tournament_response)
            tournament_data = self._safely_parse_json(tournament_response)

            winner_choice = tournament_data.get("winner")
            if winner_choice == 'a':
                winner, loser = h1, h2
            elif winner_choice == 'b':
                winner, loser = h2, h1
            else:
                logger.warning(f"Tournament agent returned invalid winner: {winner_choice}. Skipping Elo update for this round.")
                continue # Skip Elo update if no valid winner

            # Update Elo ratings
            winner.update_elo(loser.elo_rating, win=True, k_factor=k_factor)
            loser.update_elo(winner.elo_rating, win=False, k_factor=k_factor)

        self._time_execution("tournament", start_time)
        self.execution_metrics["tournaments_count"] += tournament_rounds
        logger.info(f"Tournament phase completed over {tournament_rounds} rounds. Elo ratings updated.")

        # Rank hypotheses by Elo rating
        hypotheses.sort(key=lambda h: h.elo_rating, reverse=True)
        return hypotheses


    def run_research_workflow(self, research_goal: str) -> Dict[str, Any]:
        """
        Execute the AI co-scientist research workflow to generate and refine hypotheses.

        Args:
            research_goal (str): The research goal provided by the scientist.

        Returns:
            Dict[str, Any]: A dictionary containing the final results, including top-ranked hypotheses,
                             meta-review insights, and conversation history.
        """
        logger.info(f"Starting research workflow for goal: '{research_goal}'")
        self.start_time = time.time()
        self.hypotheses = [] # Reset hypotheses list for a new run
        self.execution_metrics = {k: 0 if isinstance(v, int) else v for k, v in self.execution_metrics.items()} # Reset metrics, keep agent_execution_times structure

        try:
            # --- Generation Phase ---
            self.hypotheses = self._run_generation_phase(research_goal)

            # --- Reflection Phase ---
            self.hypotheses = self._run_reflection_phase(self.hypotheses)

            # --- Ranking Phase (Initial Ranking based on Reviews) ---
            self.hypotheses = self._run_ranking_phase(self.hypotheses)

            # --- Tournament Phase (Elo-based Ranking) ---
            self.hypotheses = self._run_tournament_phase(self.hypotheses)

            # --- Iterative Refinement Cycle ---
            for iteration in range(self.max_iterations):
                logger.info(f"\n--- Starting Iteration {iteration + 1} ---")

                # --- Meta-Review ---
                meta_review_data = self._run_meta_review_phase(self.hypotheses)

                # --- Evolution ---
                top_hypotheses_for_evolution = self.hypotheses[:min(self.evolution_top_k, len(self.hypotheses))] # Evolve top k
                self.hypotheses = self._run_evolution_phase(top_hypotheses_for_evolution, meta_review_data)

                # Re-run Reflection and Ranking on evolved hypotheses
                self.hypotheses = self._run_reflection_phase(self.hypotheses)
                self.hypotheses = self._run_ranking_phase(self.hypotheses)
                self.hypotheses = self._run_tournament_phase(self.hypotheses) # Tournament after evolution too

                # --- Proximity Analysis (after evolution and ranking each iteration) ---
                self.hypotheses = self._run_proximity_analysis_phase(self.hypotheses)


            # --- Final Output ---
            top_ranked_hypotheses = self.hypotheses[:min(10, len(self.hypotheses))] # Return top 10 or fewer
            final_output_hypotheses = [h.to_dict() for h in top_ranked_hypotheses] # Convert to dict for output

            final_output = {
                "top_ranked_hypotheses": final_output_hypotheses,
                "meta_review_insights": meta_review_data,
                "conversation_history": self.conversation.return_history_as_string(),
                "execution_metrics": self.execution_metrics,
                "total_workflow_time": time.time() - self.start_time
            }
            logger.info("Research workflow completed successfully.")
            return final_output

        except Exception as e:
            logger.error(f"Error in research workflow: {e}")
            return {
                "error": str(e),
                "conversation_history": self.conversation.return_history_as_string(),
                "execution_metrics": self.execution_metrics,
                "total_workflow_time": time.time() - self.start_time
            }

    def save_state(self) -> None:
        """Save the state of all agents."""
        for agent in [
            self.generation_agent,
            self.reflection_agent,
            self.ranking_agent,
            self.evolution_agent,
            self.meta_review_agent,
            self.proximity_agent,
            self.tournament_agent,
            self.supervisor_agent,
        ]:
            try:
                agent.save_state()
                logger.info(f"State saved for {agent.agent_name}")
            except Exception as e:
                logger.error(f"Error saving state for {agent.agent_name}: {e}")

    def load_state(self) -> None:
        """Load the saved state of all agents."""
        for agent in [
            self.generation_agent,
            self.reflection_agent,
            self.ranking_agent,
            self.evolution_agent,
            self.meta_review_agent,
            self.proximity_agent,
            self.tournament_agent,
            self.supervisor_agent,
        ]:
            try:
                agent.load_state()
                logger.info(f"State loaded for {agent.agent_name}")
            except Exception as e:
                logger.error(f"Error loading state for {agent.agent_name}: {e}")


# if __name__ == "__main__":
#     try:
#         # Initialize the AI Co-scientist Framework
#         ai_coscientist = AIScientistFramework(
#             model_name="gemini/gemini-2.0-flash",  # Or "gemini/gemini-2.0-flash" if you have access
#             max_iterations=2, # Reduced iterations for example run
#             verbose=False, # Set to True for detailed logs
#             hypotheses_per_generation=10,
#             tournament_size=8,
#             evolution_top_k=3,
#         )

#         # Define a research goal
#         research_goal = "Develop novel hypotheses for Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"

#         # Run the research workflow
#         results = ai_coscientist.run_research_workflow(research_goal)

#         # Output the results
#         print("\n--- Research Workflow Results ---")
#         if "error" in results:
#             print(f"Error during workflow: {results['error']}")
#         else:
#             print("\n--- Top Ranked Hypotheses ---")
#             for hy in results["top_ranked_hypotheses"]:
#                 print(f"- Hypothesis: {hy['text']}")
#                 print(f"  Elo Rating: {hy['elo_rating']}")
#                 print(f"  Score: {hy['score']:.2f}")
#                 print(f"  Reviews: {hy['reviews'][-1].get('review_summary') if hy['reviews'] else 'No reviews'}") # Print review summary
#                 print(f"  Similarity Cluster ID: {hy['similarity_cluster_id']}")
#                 print(f"  Win Rate: {hy['win_rate']}% (Matches: {hy['total_matches']})")
#                 print("-" * 30)

#             print("\n--- Meta-Review Insights Summary ---")
#             meta_review_summary = results["meta_review_insights"].get("meta_review_summary", "No meta-review summary available.")
#             print(meta_review_summary[:500] + "..." if len(meta_review_summary) > 500 else meta_review_summary) # Print truncated or full summary

#             print("\n--- Execution Metrics ---")
#             print(json.dumps(results["execution_metrics"], indent=2))
#             print(f"\nTotal Workflow Time: {results['total_workflow_time']:.2f} seconds")

#             if ai_coscientist.verbose: # Only print full history if verbose is on, can be very long
#                 print("\n--- Conversation History (Verbose Mode) ---")
#                 print(results["conversation_history"][:1000] + "...") # Print first 1000 chars of history

#         # Save agent states (optional)
#         ai_coscientist.save_state()

#     except Exception as e:
#         logger.error(f"Exception during main execution: {e}")