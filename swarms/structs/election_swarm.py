"""ElectionSwarm: Multi-agent orchestrator selection with AGENTSNET communication."""

import hashlib
import json
import os
import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import history_output_formatter
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

election_logger = initialize_logger(log_folder="election_swarm")

DEFAULT_BUDGET_LIMIT = 200.0
DEFAULT_CONSENSUS_THRESHOLD = 0.6
DEFAULT_MAX_WORKERS = 10
DEFAULT_MAX_ROUNDS = 5
DEFAULT_BATCH_SIZE = 25


class ElectionConfigModel(BaseModel):
    """ElectionSwarm configuration model for orchestrator selection."""

    election_type: str = Field(default="orchestrator_selection")
    max_candidates: int = Field(default=5, ge=1, le=20)
    max_voters: int = Field(default=100, ge=1, le=1000)
    enable_consensus: bool = Field(default=True)
    enable_leader_election: bool = Field(default=True)
    enable_matching: bool = Field(default=True)
    enable_coloring: bool = Field(default=True)
    enable_vertex_cover: bool = Field(default=True)
    enable_caching: bool = Field(default=True)
    enable_voter_tool_calls: bool = Field(default=True)
    batch_size: int = Field(default=25, ge=1, le=100)
    max_workers: int = Field(default=DEFAULT_MAX_WORKERS, ge=1, le=50)
    budget_limit: float = Field(default=DEFAULT_BUDGET_LIMIT, ge=0.0)
    default_model: str = Field(default="gpt-4o-mini")
    verbose_logging: bool = Field(default=False)


@dataclass
class ElectionConfig:
    """Election configuration manager."""

    config_file_path: Optional[str] = None
    config_data: Optional[Dict[str, Any]] = None
    config: ElectionConfigModel = field(init=False)

    def __post_init__(self) -> None:
        self._load_config()

    def _load_config(self) -> None:
        try:
            self.config = ElectionConfigModel()
            if self.config_file_path and os.path.exists(self.config_file_path):
                self._load_from_file()
            if self.config_data:
                self._load_from_dict(self.config_data)
        except Exception as e:
            election_logger.error(f"Failed to load configuration: {str(e)}")
            raise

    def _load_from_file(self) -> None:
        try:
            import yaml

            with open(self.config_file_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f)
                self._load_from_dict(file_config)
        except Exception as e:
            election_logger.warning(f"Failed to load config file: {e}")
            raise

    def _load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                try:
                    setattr(self.config, key, value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid config {key}: {e}")

    def get_config(self) -> ElectionConfigModel:
        return self.config

    def update_config(self, updates: Dict[str, Any]) -> None:
        try:
            self._load_from_dict(updates)
        except ValueError as e:
            election_logger.error(f"Failed to update configuration: {e}")
            raise

    def save_config(self, file_path: Optional[str] = None) -> None:
        save_path = file_path or self.config_file_path
        if not save_path:
            return
        try:
            import yaml

            config_dict = self.config.model_dump()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        except Exception as e:
            election_logger.error(f"Failed to save config: {e}")
            raise

    def validate_config(self) -> List[str]:
        errors = []
        try:
            self.config.model_validate(self.config.model_dump())
        except Exception as e:
            errors.append(f"Configuration validation failed: {e}")
        if self.config.budget_limit < 0:
            errors.append("Budget limit must be non-negative")
        if self.config.max_candidates < 1:
            errors.append("Max candidates must be at least 1")
        if self.config.max_voters < 1:
            errors.append("Max voters must be at least 1")
        return errors


class VoterType(str, Enum):
    """Voter types in the election system."""

    INDIVIDUAL = "individual"
    GROUP = "group"
    EXPERT = "expert"
    DELEGATE = "delegate"


class CandidateType(str, Enum):
    """Candidate types in the election system."""

    INDIVIDUAL = "individual"
    COALITION = "coalition"
    PARTY = "party"
    MOVEMENT = "movement"


class ElectionAlgorithm(str, Enum):
    """AGENTSNET algorithms for coordination."""

    CONSENSUS = "consensus"
    LEADER_ELECTION = "leader_election"
    MATCHING = "matching"
    COLORING = "coloring"
    VERTEX_COVER = "vertex_cover"


class MessagePassingProtocol:
    """AGENTSNET-inspired message-passing protocol."""

    def __init__(self, rounds: int = 5, synchronous: bool = True):
        self.rounds = rounds
        self.synchronous = synchronous
        self.current_round = 0
        self.message_history: List[Dict[str, Any]] = []

    def create_system_prompt(
        self,
        agent_name: str,
        neighbors: List[str],
        task_description: str,
    ) -> str:
        neighbors_str = ", ".join(neighbors)
        return f"""You are an agent named {agent_name} connected with neighbors: {neighbors_str}.

{task_description}

Communication Rules:
1. You can only communicate with immediate neighbors: {neighbors_str}
2. Synchronous message-passing in {self.rounds} rounds
3. Output JSON messages: {{"neighbor_name": "message"}}
4. After {self.rounds} rounds, provide your final answer
5. Base decisions on information from neighbors and your own reasoning

Think step-by-step about your strategy and communicate it clearly to neighbors."""

    def send_message(self, from_agent: str, to_agent: str, message: str) -> None:
        self.message_history.append(
            {
                "round": self.current_round,
                "from": from_agent,
                "to": to_agent,
                "message": message,
                "timestamp": datetime.now(),
            }
        )

    def get_messages_for_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        return [msg for msg in self.message_history if msg["to"] == agent_name]

    def advance_round(self) -> None:
        self.current_round += 1

    def reset(self) -> None:
        self.current_round = 0
        self.message_history.clear()


class VoteResult(str, Enum):
    """Possible vote results."""

    FOR = "for"
    AGAINST = "against"
    ABSTAIN = "abstain"
    INVALID = "invalid"


@dataclass
class CostTracker:
    """Track costs and usage for budget management."""

    total_tokens_used: int = 0
    total_cost_estimate: float = 0.0
    budget_limit: float = DEFAULT_BUDGET_LIMIT
    token_cost_per_1m: float = 0.15
    requests_made: int = 0
    cache_hits: int = 0

    def add_tokens(self, tokens: int) -> None:
        self.total_tokens_used += tokens
        self.total_cost_estimate = (
            self.total_tokens_used / 1_000_000
        ) * self.token_cost_per_1m
        self.requests_made += 1

    def add_cache_hit(self) -> None:
        self.cache_hits += 1

    def check_budget(self) -> bool:
        return self.total_cost_estimate <= self.budget_limit

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens_used,
            "total_cost": self.total_cost_estimate,
            "requests_made": self.requests_made,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits
            / max(1, self.requests_made + self.cache_hits),
            "budget_remaining": max(0, self.budget_limit - self.total_cost_estimate),
        }


@dataclass
class VoterProfile:
    """Represents an agent voter in the orchestrator selection system."""

    voter_id: str
    name: str
    voter_type: VoterType
    preferences: Dict[str, Any] = field(default_factory=dict)
    expertise_areas: List[str] = field(default_factory=list)
    voting_weight: float = 1.0
    agent: Optional[Agent] = None
    is_loaded: bool = False
    demographics: Dict[str, Any] = field(default_factory=dict)
    past_voting_history: List[Dict[str, Any]] = field(default_factory=list)
    neighbors: List[str] = field(default_factory=list)
    coordination_style: str = "collaborative"
    leadership_preferences: Dict[str, float] = field(default_factory=dict)


@dataclass
class CandidateProfile:
    """Represents an orchestrator candidate in the selection system."""

    candidate_id: str
    name: str
    candidate_type: CandidateType
    party_affiliation: Optional[str] = None
    policy_positions: Dict[str, Any] = field(default_factory=dict)
    campaign_promises: List[str] = field(default_factory=list)
    experience: List[str] = field(default_factory=list)
    agent: Optional[Agent] = None
    is_loaded: bool = False
    support_base: Dict[str, float] = field(default_factory=dict)
    campaign_strategy: Dict[str, Any] = field(default_factory=dict)
    leadership_style: str = "collaborative"
    coordination_approach: Dict[str, Any] = field(default_factory=dict)
    technical_expertise: List[str] = field(default_factory=list)


@dataclass
class VoteCounterProfile:
    """Represents a vote counter agent responsible for counting votes and presenting results."""

    counter_id: str
    name: str
    role: str = "Vote Counter"
    credentials: List[str] = field(default_factory=list)
    counting_methodology: str = "transparent"
    reporting_style: str = "comprehensive"
    agent: Optional[Agent] = None
    is_loaded: bool = False
    counting_experience: List[str] = field(default_factory=list)
    verification_protocols: List[str] = field(default_factory=list)
    documentation_standards: List[str] = field(default_factory=list)
    result_presentation_style: str = "detailed"


@dataclass
class VoterDecision:
    """Structured output from voter agents."""

    voter_id: str
    rationality: str
    vote: VoteResult
    confidence: float = 0.0
    reasoning_factors: List[str] = field(default_factory=list)
    candidate_rankings: Dict[str, int] = field(default_factory=dict)
    tool_call_explanation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VoteCountingResult:
    """Results of vote counting process."""

    counter_id: str
    counter_name: str
    counting_timestamp: datetime = field(default_factory=datetime.now)
    total_votes_counted: int = 0
    valid_votes: int = 0
    invalid_votes: int = 0
    abstentions: int = 0
    vote_breakdown: Dict[str, int] = field(default_factory=dict)
    counting_notes: List[str] = field(default_factory=list)
    verification_completed: bool = False
    counting_methodology: str = "transparent"
    documentation_provided: bool = False


@dataclass
class ElectionResult:
    """Results of an election."""

    election_id: str
    algorithm_used: ElectionAlgorithm
    total_voters: int
    total_candidates: int
    votes_cast: int
    winner: Optional[str] = None
    vote_distribution: Dict[str, int] = field(default_factory=dict)
    voter_decisions: List[VoterDecision] = field(default_factory=list)
    consensus_reached: bool = False
    rounds_to_consensus: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    vote_counting_result: Optional[VoteCountingResult] = None


class ElectionSwarm:
    """Multi-agent orchestrator selection system with AGENTSNET coordination algorithms."""

    def __init__(
        self,
        name: str = "ElectionSwarm",
        description: str = "Orchestrator selection with AGENTSNET algorithms",
        voters: Optional[List[VoterProfile]] = None,
        candidates: Optional[List[CandidateProfile]] = None,
        vote_counter: Optional[VoteCounterProfile] = None,
        election_config: Optional[ElectionConfig] = None,
        max_loops: int = 1,
        output_type: OutputType = "dict-all-except-first",
        verbose: bool = False,
        enable_lazy_loading: bool = True,
        enable_caching: bool = True,
        batch_size: int = 25,
        budget_limit: float = 200.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.description = description
        self.voters = voters or []
        self.candidates = candidates or []
        self.vote_counter = vote_counter
        self.election_config = election_config or ElectionConfig()
        self.max_loops = max_loops
        self.output_type = output_type
        self.verbose = verbose
        self.enable_lazy_loading = enable_lazy_loading
        self.enable_caching = enable_caching
        self.batch_size = batch_size
        self.budget_limit = budget_limit

        self.conversation = Conversation(time_enabled=False)
        self.cost_tracker = CostTracker(budget_limit=budget_limit)
        self.cache: Dict[str, str] = {}
        cpu_count = os.cpu_count() or 4
        self.max_workers = min(self.election_config.config.max_workers, cpu_count)
        self.message_protocol = MessagePassingProtocol(rounds=5)

        self._init_election_swarm()

    def _init_election_swarm(self) -> None:
        if self.verbose:
            election_logger.info(f"Initializing ElectionSwarm: {self.name}")

        self._perform_reliability_checks()

        if not self.voters:
            self._setup_default_voters()

        if not self.candidates:
            self._setup_default_candidates()

        if not self.vote_counter:
            self._setup_default_vote_counter()

        self._add_context_to_agents()

        if self.verbose:
            election_logger.info(f"ElectionSwarm initialized successfully: {self.name}")

    def _perform_reliability_checks(self) -> None:
        try:
            if self.verbose:
                election_logger.info(f"Running reliability checks for: {self.name}")

            # Default voters and candidates will be set up in _setup_default_* methods
            pass

            if self.max_loops <= 0:
                raise ValueError("Max loops must be greater than 0.")

            if self.verbose:
                election_logger.info(f"Reliability checks passed for: {self.name}")

        except Exception as e:
            error_msg = f"Failed reliability checks: {str(e)}\nTraceback: {traceback.format_exc()}"
            election_logger.error(error_msg)
            raise

    def _setup_default_voters(self) -> None:
        if self.verbose:
            election_logger.info("Setting up default voters with neighbor connections")

        default_voters = [
            VoterProfile(
                voter_id="voter_1",
                name="Alice Johnson",
                voter_type=VoterType.INDIVIDUAL,
                preferences={
                    "economy": 0.8,
                    "environment": 0.6,
                    "healthcare": 0.9,
                },
                expertise_areas=["economics", "healthcare"],
                demographics={
                    "age": 35,
                    "education": "college",
                    "income": "middle",
                },
                neighbors=["Bob Smith", "Carol Davis"],
            ),
            VoterProfile(
                voter_id="voter_2",
                name="Bob Smith",
                voter_type=VoterType.INDIVIDUAL,
                preferences={
                    "economy": 0.9,
                    "environment": 0.4,
                    "security": 0.8,
                },
                expertise_areas=["economics", "security"],
                demographics={
                    "age": 42,
                    "education": "graduate",
                    "income": "high",
                },
                neighbors=["Alice Johnson", "Carol Davis"],
            ),
            VoterProfile(
                voter_id="voter_3",
                name="Carol Davis",
                voter_type=VoterType.EXPERT,
                preferences={
                    "environment": 0.9,
                    "education": 0.8,
                    "social_justice": 0.7,
                },
                expertise_areas=["environment", "education"],
                demographics={
                    "age": 28,
                    "education": "phd",
                    "income": "middle",
                },
                neighbors=["Alice Johnson", "Bob Smith"],
            ),
        ]

        self.voters = default_voters

        if self.verbose:
            election_logger.info(
                f"Set up {len(default_voters)} default voters with neighbor connections"
            )

    def _setup_default_candidates(self) -> None:
        if self.verbose:
            election_logger.info("Setting up default candidates")

        default_candidates = [
            CandidateProfile(
                candidate_id="candidate_1",
                name="John Progressive",
                candidate_type=CandidateType.INDIVIDUAL,
                party_affiliation="Progressive Party",
                policy_positions={
                    "economy": "stimulus",
                    "environment": "green_new_deal",
                    "healthcare": "universal",
                },
                campaign_promises=[
                    "Universal healthcare",
                    "Green energy transition",
                    "Education reform",
                ],
                experience=[
                    "Mayor",
                    "State Senator",
                    "Business Leader",
                ],
            ),
            CandidateProfile(
                candidate_id="candidate_2",
                name="Sarah Conservative",
                candidate_type=CandidateType.INDIVIDUAL,
                party_affiliation="Conservative Party",
                policy_positions={
                    "economy": "tax_cuts",
                    "environment": "balanced",
                    "security": "strong_defense",
                },
                campaign_promises=[
                    "Tax reduction",
                    "Strong defense",
                    "Traditional values",
                ],
                experience=[
                    "Governor",
                    "Business Executive",
                    "Military Officer",
                ],
            ),
        ]

        self.candidates = default_candidates

        if self.verbose:
            election_logger.info(f"Set up {len(default_candidates)} default candidates")

    def _setup_default_vote_counter(self) -> None:
        if self.verbose:
            election_logger.info("Setting up default vote counter")

        default_vote_counter = VoteCounterProfile(
            counter_id="counter_001",
            name="Election Commissioner",
            role="Vote Counter",
            credentials=["Certified Election Official", "Transparency Specialist"],
            counting_methodology="transparent",
            reporting_style="comprehensive",
            counting_experience=[
                "Election oversight (10 years)",
                "Vote counting and verification",
                "Result documentation and reporting"
            ],
            verification_protocols=[
                "Double-count verification",
                "Cross-reference validation",
                "Audit trail documentation"
            ],
            documentation_standards=[
                "Detailed vote breakdown",
                "Verification documentation",
                "Transparent reporting"
            ],
            result_presentation_style="detailed"
        )

        self.vote_counter = default_vote_counter

        if self.verbose:
            election_logger.info("Set up default vote counter")

    def _add_context_to_agents(self) -> None:
        try:
            if self.verbose:
                election_logger.info("Adding context to agents")

            for voter in self.voters:
                if voter.agent:
                    self._add_voter_context(voter)

            for candidate in self.candidates:
                if candidate.agent:
                    self._add_candidate_context(candidate)

            if self.vote_counter and self.vote_counter.agent:
                self._add_vote_counter_context(self.vote_counter)

            if self.verbose:
                election_logger.info("Context added to agents successfully")

        except Exception as e:
            error_msg = f"Failed to add context to agents: {str(e)}\nTraceback: {traceback.format_exc()}"
            election_logger.error(error_msg)
            raise

    def _add_voter_context(self, voter: VoterProfile) -> None:
        if not voter.agent:
            return

        context = f"""
        Voter Context:
        - Name: {voter.name}
        - Type: {voter.voter_type.value}
        - Preferences: {voter.preferences}
        - Expertise: {voter.expertise_areas}
        - Demographics: {voter.demographics}
        """

        voter.agent.system_prompt += context

    def _add_candidate_context(self, candidate: CandidateProfile) -> None:
        if not candidate.agent:
            return

        context = f"""
        Candidate Context:
        - Name: {candidate.name}
        - Type: {candidate.candidate_type.value}
        - Party: {candidate.party_affiliation}
        - Policies: {candidate.policy_positions}
        - Promises: {candidate.campaign_promises}
        - Experience: {candidate.experience}
        """

        candidate.agent.system_prompt += context

    def _add_vote_counter_context(self, vote_counter: VoteCounterProfile) -> None:
        if not vote_counter.agent:
            return

        context = f"""
        Vote Counter Context:
        - Name: {vote_counter.name}
        - Role: {vote_counter.role}
        - Credentials: {vote_counter.credentials}
        - Methodology: {vote_counter.counting_methodology}
        - Reporting Style: {vote_counter.reporting_style}
        - Experience: {vote_counter.counting_experience}
        - Verification Protocols: {vote_counter.verification_protocols}
        """

        vote_counter.agent.system_prompt += context

    def _load_voter_agent(self, voter: VoterProfile) -> Agent:
        """
        Load an agent for a voter with AGENTSNET-style communication.

        Args:
            voter: The voter profile to create an agent for

        Returns:
            Agent: The loaded voter agent
        """
        if voter.agent and voter.is_loaded:
            return voter.agent

        # Create AGENTSNET-style system prompt
        task_description = f"""You are {voter.name}, a {voter.voter_type.value} agent in an orchestrator selection process.

Your profile:
- Preferences: {voter.preferences}
- Expertise areas: {voter.expertise_areas}
- Demographics: {voter.demographics}
- Coordination style: {voter.coordination_style}
- Leadership preferences: {voter.leadership_preferences}

Your task is to select the best orchestrator candidate after communicating with neighbors.
The orchestrator will coordinate multi-agent workflows and manage team collaboration.
When voting, provide structured output with:
1. Rationality: Your detailed reasoning for your orchestrator selection
2. Vote: Your actual vote (for/against/abstain)
3. Confidence: Your confidence level (0.0-1.0)
4. Reasoning factors: Key factors that influenced your decision
5. Candidate rankings: How you rank each orchestrator candidate"""

        system_prompt = self.message_protocol.create_system_prompt(
            agent_name=voter.name,
            neighbors=voter.neighbors,
            task_description=task_description,
        )

        # Create tools for voter explanation if enabled
        tools = []
        # Note: Tool creation disabled due to BaseTool validation issues
        # if self.election_config.config.enable_voter_tool_calls:
        #     tools.append(self._create_voting_explanation_tool(voter))

        agent = Agent(
            agent_name=voter.name,
            agent_description=f"{voter.voter_type.value} voter with expertise in {', '.join(voter.expertise_areas)}",
            model_name=self.election_config.config.default_model,
            max_loops=1,
            system_prompt=system_prompt,
        )

        voter.agent = agent
        voter.is_loaded = True

        return agent

    def _create_voting_explanation_tool(self, voter: VoterProfile) -> Dict[str, Any]:
        """
        Create a tool for voter to explain their voting decision.

        Args:
            voter: The voter profile to create the tool for

        Returns:
            Dict[str, Any]: Tool definition for voting explanation
        """
        return {
            "type": "function",
            "function": {
                "name": "explain_voting_decision",
                "description": f"Explain why {voter.name} is voting for a specific candidate",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "voter_name": {
                            "type": "string",
                            "description": f"The name of the voter: {voter.name}",
                        },
                        "voter_id": {
                            "type": "string",
                            "description": f"The ID of the voter: {voter.voter_id}",
                        },
                        "chosen_candidate": {
                            "type": "string",
                            "description": "The name of the candidate being voted for",
                        },
                        "voting_reasoning": {
                            "type": "string",
                            "description": "Detailed explanation of why this candidate was chosen",
                        },
                        "key_factors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of key factors that influenced the voting decision",
                        },
                        "confidence_level": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence level in the voting decision (0.0-1.0)",
                        },
                        "alternative_considerations": {
                            "type": "string",
                            "description": "What other candidates were considered and why they were not chosen",
                        },
                    },
                    "required": [
                        "voter_name",
                        "voter_id",
                        "chosen_candidate",
                        "voting_reasoning",
                        "key_factors",
                        "confidence_level",
                    ],
                },
            },
        }

    def _load_candidate_agent(self, candidate: CandidateProfile) -> Agent:
        """
        Load an agent for a candidate.

        Args:
            candidate: The candidate profile to create an agent for

        Returns:
            Agent: The loaded candidate agent
        """
        if candidate.agent and candidate.is_loaded:
            return candidate.agent

        system_prompt = f"""You are {candidate.name}, a candidate for election.

Your profile:
- Party affiliation: {candidate.party_affiliation}
- Policy positions: {candidate.policy_positions}
- Campaign promises: {candidate.campaign_promises}
- Experience: {candidate.experience}

Your role is to:
1. Present your platform and policies clearly
2. Respond to voter questions and concerns
3. Make compelling arguments for your candidacy
4. Address criticisms and challenges

Always be professional, articulate, and focused on your key messages."""

        agent = Agent(
            agent_name=candidate.name,
            agent_description=f"Candidate for election representing {candidate.party_affiliation}",
            model_name=self.election_config.config.default_model,
            max_loops=1,
            system_prompt=system_prompt,
        )

        candidate.agent = agent
        candidate.is_loaded = True

        return agent

    def _load_vote_counter_agent(self, vote_counter: VoteCounterProfile) -> Agent:
        """
        Load an agent for the vote counter.

        Args:
            vote_counter: The vote counter profile to create an agent for

        Returns:
            Agent: The loaded vote counter agent
        """
        if vote_counter.agent and vote_counter.is_loaded:
            return vote_counter.agent

        system_prompt = f"""You are {vote_counter.name}, the official vote counter for this election.

Your role and responsibilities:
- Count all votes accurately and transparently
- Verify vote integrity and validity
- Document the counting process thoroughly
- Present results in a clear, comprehensive manner
- Ensure all procedures follow {vote_counter.counting_methodology} methodology
- Maintain {vote_counter.reporting_style} reporting standards

Your credentials: {vote_counter.credentials}
Your experience: {vote_counter.counting_experience}
Your verification protocols: {vote_counter.verification_protocols}
Your documentation standards: {vote_counter.documentation_standards}

When counting votes, you must:
1. Verify each vote is valid and properly cast
2. Count votes for each candidate accurately
3. Document the counting process step-by-step
4. Provide detailed breakdown of results
5. Ensure transparency and auditability
6. Present results in a professional, comprehensive manner

Your counting methodology: {vote_counter.counting_methodology}
Your result presentation style: {vote_counter.result_presentation_style}

Always maintain the highest standards of accuracy, transparency, and documentation."""

        agent = Agent(
            agent_name=vote_counter.name,
            agent_description=f"Official vote counter with {vote_counter.counting_methodology} methodology",
            model_name=self.election_config.config.default_model,
            max_loops=1,
            system_prompt=system_prompt,
        )

        vote_counter.agent = agent
        vote_counter.is_loaded = True

        return agent

    def _get_cache_key(self, task: str, participants: List[str]) -> str:
        """
        Generate cache key for a task.

        Args:
            task: The task description
            participants: List of participant names

        Returns:
            str: MD5 hash of the cache key
        """
        content = f"{task}:{':'.join(sorted(participants))}"
        return hashlib.md5(content.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[str]:
        """
        Check if result is cached.

        Args:
            cache_key: The cache key to check

        Returns:
            Optional[str]: Cached result if found, None otherwise
        """
        if not self.enable_caching:
            return None

        if cache_key in self.cache:
            self.cost_tracker.add_cache_hit()
            if self.verbose:
                election_logger.info(f"Cache hit for key: {cache_key[:8]}...")
            return self.cache[cache_key]

        return None

    def _cache_response(self, cache_key: str, response: str) -> None:
        """
        Cache a response.

        Args:
            cache_key: The cache key to store under
            response: The response to cache
        """
        if self.enable_caching:
            self.cache[cache_key] = response

    def conduct_election(
        self,
        election_type: ElectionAlgorithm = ElectionAlgorithm.CONSENSUS,
        participants: Optional[List[str]] = None,
        max_rounds: int = 5,
    ) -> ElectionResult:
        """
        Conduct an election using the specified algorithm.

        Args:
            election_type: The AGENTSNET algorithm to use
            participants: List of participant IDs (optional)
            max_rounds: Maximum number of rounds for consensus algorithms

        Returns:
            ElectionResult: The results of the election
        """
        try:
            if self.verbose:
                election_logger.info(f"Conducting {election_type.value} election")

            # Initialize election result
            election_id = f"election_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            result = ElectionResult(
                election_id=election_id,
                algorithm_used=election_type,
                total_voters=len(self.voters),
                total_candidates=len(self.candidates),
                votes_cast=0,
            )

            # Conduct election based on algorithm type
            if election_type == ElectionAlgorithm.CONSENSUS:
                result = self._conduct_consensus_election(result, max_rounds)
            elif election_type == ElectionAlgorithm.LEADER_ELECTION:
                result = self._conduct_leader_election(result)
            elif election_type == ElectionAlgorithm.MATCHING:
                result = self._conduct_matching_election(result)
            elif election_type == ElectionAlgorithm.COLORING:
                result = self._conduct_coloring_election(result)
            elif election_type == ElectionAlgorithm.VERTEX_COVER:
                result = self._conduct_vertex_cover_election(result)
            else:
                raise ValueError(f"Unsupported election algorithm: {election_type}")

            if self.verbose:
                election_logger.info(f"Election completed: {election_id}")
                if result.winner:
                    election_logger.info(f"Winner: {result.winner}")

            return result

        except Exception as e:
            error_msg = (
                f"Failed to conduct election: {str(e)}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            election_logger.error(error_msg)
            raise

    def _conduct_consensus_election(
        self, result: ElectionResult, max_rounds: int
    ) -> ElectionResult:
        """
        Conduct a consensus-based election.

        Args:
            result: The election result object to populate
            max_rounds: Maximum number of consensus rounds

        Returns:
            ElectionResult: The completed election result
        """
        if self.verbose:
            election_logger.info("Conducting consensus election")

        round_num = 0
        consensus_reached = False

        while round_num < max_rounds and not consensus_reached:
            round_num += 1

            if self.verbose:
                election_logger.info(f"Consensus round {round_num}/{max_rounds}")

            # Collect votes from all voters
            voter_decisions = self._collect_voter_decisions()
            result.voter_decisions.extend(voter_decisions)

            # Check for consensus
            consensus_reached = self._check_consensus(voter_decisions)

            if consensus_reached:
                result.consensus_reached = True
                result.rounds_to_consensus = round_num
                result.winner = self._determine_winner(voter_decisions)
                break

        if not consensus_reached:
            # Use majority vote as fallback
            result.winner = self._determine_winner(result.voter_decisions)

        result.votes_cast = len(result.voter_decisions)
        
        # Count votes using the vote counter
        result.vote_counting_result = self._count_votes(result.voter_decisions)
        
        return result

    def _conduct_leader_election(self, result: ElectionResult) -> ElectionResult:
        """
        Conduct a leader election using distributed algorithms.

        Args:
            result: The election result object to populate

        Returns:
            ElectionResult: The completed election result
        """
        if self.verbose:
            election_logger.info("Conducting leader election")

        # Collect votes from all voters
        voter_decisions = self._collect_voter_decisions()
        result.voter_decisions = voter_decisions

        # Determine winner based on votes
        result.winner = self._determine_winner(voter_decisions)
        result.votes_cast = len(voter_decisions)

        return result

    def _conduct_matching_election(self, result: ElectionResult) -> ElectionResult:
        """
        Conduct a matching-based election.

        This method implements maximal matching algorithms for pairing
        voters with candidates based on compatibility.

        Args:
            result: The election result object to populate

        Returns:
            ElectionResult: The completed election result
        """
        if self.verbose:
            election_logger.info("Conducting matching election")

        # This would implement maximal matching algorithms
        # For now, use standard voting
        voter_decisions = self._collect_voter_decisions()
        result.voter_decisions = voter_decisions
        result.winner = self._determine_winner(voter_decisions)
        result.votes_cast = len(voter_decisions)

        return result

    def _conduct_coloring_election(self, result: ElectionResult) -> ElectionResult:
        """
        Conduct a coloring-based election.

        This method implements (Δ+1)-coloring algorithms for grouping
        voters and candidates based on compatibility constraints.

        Args:
            result: The election result object to populate

        Returns:
            ElectionResult: The completed election result
        """
        if self.verbose:
            election_logger.info("Conducting coloring election")

        # This would implement (Δ+1)-coloring algorithms
        # For now, use standard voting
        voter_decisions = self._collect_voter_decisions()
        result.voter_decisions = voter_decisions
        result.winner = self._determine_winner(voter_decisions)
        result.votes_cast = len(voter_decisions)

        return result

    def _conduct_vertex_cover_election(self, result: ElectionResult) -> ElectionResult:
        """
        Conduct a vertex cover-based election.

        This method implements minimal vertex cover algorithms for
        selecting a minimal set of coordinators to cover all voters.

        Args:
            result: The election result object to populate

        Returns:
            ElectionResult: The completed election result
        """
        if self.verbose:
            election_logger.info("Conducting vertex cover election")

        # This would implement minimal vertex cover algorithms
        # For now, use standard voting
        voter_decisions = self._collect_voter_decisions()
        result.voter_decisions = voter_decisions
        result.winner = self._determine_winner(voter_decisions)
        result.votes_cast = len(voter_decisions)
        
        # Count votes using the vote counter
        result.vote_counting_result = self._count_votes(voter_decisions)

        return result

    def _count_votes(self, voter_decisions: List[VoterDecision]) -> VoteCountingResult:
        """
        Count votes using the vote counter agent.

        Args:
            voter_decisions: List of voter decisions to count

        Returns:
            VoteCountingResult: The vote counting results
        """
        if not self.vote_counter:
            # Create a basic counting result if no vote counter is available
            return VoteCountingResult(
                counter_id="default",
                counter_name="System Counter",
                total_votes_counted=len(voter_decisions),
                valid_votes=len([d for d in voter_decisions if d.vote != VoteResult.INVALID]),
                invalid_votes=len([d for d in voter_decisions if d.vote == VoteResult.INVALID]),
                abstentions=len([d for d in voter_decisions if d.vote == VoteResult.ABSTAIN]),
                vote_breakdown=self._calculate_vote_breakdown(voter_decisions),
                verification_completed=True,
                documentation_provided=True
            )

        try:
            # Load vote counter agent
            counter_agent = self._load_vote_counter_agent(self.vote_counter)

            # Create vote counting prompt
            counting_prompt = self._create_vote_counting_prompt(voter_decisions)

            # Get counting response from vote counter
            counting_response = counter_agent.run(task=counting_prompt)

            # Parse counting results
            counting_result = self._parse_vote_counting_response(counting_response, voter_decisions)

            return counting_result

        except Exception as e:
            election_logger.error(f"Failed to count votes with vote counter: {str(e)}")
            # Fallback to basic counting
            return VoteCountingResult(
                counter_id=self.vote_counter.counter_id,
                counter_name=self.vote_counter.name,
                total_votes_counted=len(voter_decisions),
                valid_votes=len([d for d in voter_decisions if d.vote != VoteResult.INVALID]),
                invalid_votes=len([d for d in voter_decisions if d.vote == VoteResult.INVALID]),
                abstentions=len([d for d in voter_decisions if d.vote == VoteResult.ABSTAIN]),
                vote_breakdown=self._calculate_vote_breakdown(voter_decisions),
                verification_completed=False,
                documentation_provided=False,
                counting_notes=[f"Error in vote counting: {str(e)}"]
            )

    def _create_vote_counting_prompt(self, voter_decisions: List[VoterDecision]) -> str:
        """
        Create a prompt for the vote counter to count votes.

        Args:
            voter_decisions: List of voter decisions to count

        Returns:
            str: Vote counting prompt
        """
        prompt = f"""OFFICIAL VOTE COUNTING

You are tasked with counting and verifying all votes in this election.

VOTER DECISIONS TO COUNT:
"""
        
        for i, decision in enumerate(voter_decisions, 1):
            prompt += f"""
Vote #{i}:
- Voter ID: {decision.voter_id}
- Vote: {decision.vote.value}
- Confidence: {decision.confidence}
- Reasoning: {decision.rationality}
- Candidate Rankings: {decision.candidate_rankings}
"""

        prompt += f"""
CANDIDATES IN THIS ELECTION:
"""
        for candidate in self.candidates:
            prompt += f"- {candidate.name} ({candidate.party_affiliation})\n"

        prompt += """
YOUR TASK:
1. Count all votes accurately
2. Verify vote validity
3. Calculate vote distribution by candidate
4. Identify the winner
5. Document the counting process
6. Provide detailed breakdown

Please provide your counting results in the following JSON format:
{
    "total_votes_counted": number,
    "valid_votes": number,
    "invalid_votes": number,
    "abstentions": number,
    "vote_breakdown": {
        "candidate_name": vote_count
    },
    "winner": "candidate_name",
    "counting_notes": ["note1", "note2"],
    "verification_completed": true,
    "documentation_provided": true
}

Ensure accuracy, transparency, and complete documentation of the counting process."""

        return prompt

    def _parse_vote_counting_response(self, response: str, voter_decisions: List[VoterDecision]) -> VoteCountingResult:
        """
        Parse the vote counter's response.

        Args:
            response: Response from the vote counter agent
            voter_decisions: Original voter decisions for validation

        Returns:
            VoteCountingResult: Parsed counting results
        """
        try:
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                # Fallback parsing
                parsed = {}

            # Create counting result
            counting_result = VoteCountingResult(
                counter_id=self.vote_counter.counter_id,
                counter_name=self.vote_counter.name,
                total_votes_counted=parsed.get("total_votes_counted", len(voter_decisions)),
                valid_votes=parsed.get("valid_votes", len([d for d in voter_decisions if d.vote != VoteResult.INVALID])),
                invalid_votes=parsed.get("invalid_votes", len([d for d in voter_decisions if d.vote == VoteResult.INVALID])),
                abstentions=parsed.get("abstentions", len([d for d in voter_decisions if d.vote == VoteResult.ABSTAIN])),
                vote_breakdown=parsed.get("vote_breakdown", self._calculate_vote_breakdown(voter_decisions)),
                counting_notes=parsed.get("counting_notes", []),
                verification_completed=parsed.get("verification_completed", True),
                documentation_provided=parsed.get("documentation_provided", True)
            )

            return counting_result

        except Exception as e:
            election_logger.error(f"Failed to parse vote counting response: {str(e)}")
            # Return basic counting result
            return VoteCountingResult(
                counter_id=self.vote_counter.counter_id,
                counter_name=self.vote_counter.name,
                total_votes_counted=len(voter_decisions),
                valid_votes=len([d for d in voter_decisions if d.vote != VoteResult.INVALID]),
                invalid_votes=len([d for d in voter_decisions if d.vote == VoteResult.INVALID]),
                abstentions=len([d for d in voter_decisions if d.vote == VoteResult.ABSTAIN]),
                vote_breakdown=self._calculate_vote_breakdown(voter_decisions),
                verification_completed=False,
                documentation_provided=False,
                counting_notes=[f"Error parsing counting response: {str(e)}"]
            )

    def _calculate_vote_breakdown(self, voter_decisions: List[VoterDecision]) -> Dict[str, int]:
        """
        Calculate vote breakdown by candidate.

        Args:
            voter_decisions: List of voter decisions

        Returns:
            Dict[str, int]: Vote breakdown by candidate
        """
        breakdown = {}
        
        for decision in voter_decisions:
            if decision.vote == VoteResult.FOR and decision.candidate_rankings:
                # Find the highest ranked candidate
                winner = min(decision.candidate_rankings.items(), key=lambda x: x[1])[0]
                breakdown[winner] = breakdown.get(winner, 0) + 1
        
        return breakdown

    def present_election_results(self, result: ElectionResult) -> str:
        """
        Present election results using the vote counter.

        Args:
            result: The election result to present

        Returns:
            str: Formatted presentation of results
        """
        if not self.vote_counter:
            return self._create_basic_result_presentation(result)

        try:
            # Load vote counter agent
            counter_agent = self._load_vote_counter_agent(self.vote_counter)

            # Create result presentation prompt
            presentation_prompt = self._create_result_presentation_prompt(result)

            # Get presentation from vote counter
            presentation = counter_agent.run(task=presentation_prompt)

            return presentation

        except Exception as e:
            election_logger.error(f"Failed to present results with vote counter: {str(e)}")
            return self._create_basic_result_presentation(result)

    def _create_result_presentation_prompt(self, result: ElectionResult) -> str:
        """
        Create a prompt for the vote counter to present results.

        Args:
            result: The election result to present

        Returns:
            str: Result presentation prompt
        """
        prompt = f"""OFFICIAL ELECTION RESULTS PRESENTATION

You are tasked with presenting the official results of this election in a professional, comprehensive manner.

ELECTION INFORMATION:
- Election ID: {result.election_id}
- Algorithm Used: {result.algorithm_used.value}
- Total Voters: {result.total_voters}
- Total Candidates: {result.total_candidates}
- Votes Cast: {result.votes_cast}
- Winner: {result.winner}
- Consensus Reached: {result.consensus_reached}
- Rounds to Consensus: {result.rounds_to_consensus}

VOTE COUNTING RESULTS:
"""
        
        if result.vote_counting_result:
            counting = result.vote_counting_result
            prompt += f"""
- Counter: {counting.counter_name}
- Total Votes Counted: {counting.total_votes_counted}
- Valid Votes: {counting.valid_votes}
- Invalid Votes: {counting.invalid_votes}
- Abstentions: {counting.abstentions}
- Vote Breakdown: {counting.vote_breakdown}
- Verification Completed: {counting.verification_completed}
- Documentation Provided: {counting.documentation_provided}
- Counting Notes: {counting.counting_notes}
"""

        prompt += f"""
CANDIDATES:
"""
        for candidate in self.candidates:
            prompt += f"- {candidate.name} ({candidate.party_affiliation})\n"

        prompt += f"""
VOTER DECISIONS:
"""
        for i, decision in enumerate(result.voter_decisions, 1):
            prompt += f"""
Vote #{i}:
- Voter: {decision.voter_id}
- Vote: {decision.vote.value}
- Confidence: {decision.confidence}
- Reasoning: {decision.rationality}
- Candidate Rankings: {decision.candidate_rankings}
"""

        prompt += """
YOUR TASK:
Present the official election results in a professional, comprehensive format that includes:
1. Executive summary of the election
2. Detailed vote counting results
3. Winner announcement with margin of victory
4. Vote distribution analysis
5. Voter participation statistics
6. Verification and documentation status
7. Any notable patterns or insights
8. Official certification of results

Format your presentation professionally and ensure it meets the highest standards of transparency and documentation."""

        return prompt

    def _create_basic_result_presentation(self, result: ElectionResult) -> str:
        """
        Create a basic result presentation if no vote counter is available.

        Args:
            result: The election result to present

        Returns:
            str: Basic result presentation
        """
        presentation = f"""
OFFICIAL ELECTION RESULTS
========================

Election ID: {result.election_id}
Algorithm Used: {result.algorithm_used.value}
Date: {result.timestamp}

PARTICIPANTS:
- Total Voters: {result.total_voters}
- Total Candidates: {result.total_candidates}
- Votes Cast: {result.votes_cast}

RESULTS:
- Winner: {result.winner}
- Consensus Reached: {result.consensus_reached}
- Rounds to Consensus: {result.rounds_to_consensus}

VOTE DISTRIBUTION:
{result.vote_distribution}

VOTER DECISIONS:
"""
        for i, decision in enumerate(result.voter_decisions, 1):
            presentation += f"""
Vote #{i}:
- Voter: {decision.voter_id}
- Vote: {decision.vote.value}
- Confidence: {decision.confidence}
- Reasoning: {decision.rationality}
"""

        if result.vote_counting_result:
            counting = result.vote_counting_result
            presentation += f"""

VOTE COUNTING DETAILS:
- Counter: {counting.counter_name}
- Total Votes Counted: {counting.total_votes_counted}
- Valid Votes: {counting.valid_votes}
- Invalid Votes: {counting.invalid_votes}
- Abstentions: {counting.abstentions}
- Vote Breakdown: {counting.vote_breakdown}
- Verification Completed: {counting.verification_completed}
"""

        presentation += "\n--- END OF OFFICIAL RESULTS ---"
        return presentation

    def _collect_voter_decisions(self) -> List[VoterDecision]:
        """
        Collect structured decisions from all voters using simplified approach.

        Returns:
            List[VoterDecision]: List of voter decisions
        """
        if self.verbose:
            election_logger.info("Collecting voter decisions")

        # Collect decisions directly from voters (simplified approach)
        decisions = []
        for voter in self.voters:
            decision = self._get_voter_decision(voter)
            if decision:
                decisions.append(decision)

        if self.verbose:
            election_logger.info(f"Collected {len(decisions)} voter decisions")

        return decisions

    def _conduct_message_passing_round(self) -> None:
        """
        Conduct a single round of message-passing between voters.

        This method implements the core AGENTSNET communication mechanism
        where voters exchange messages with their neighbors.
        """
        # Collect messages from all voters
        for voter in self.voters:
            if voter.neighbors:  # Only if voter has neighbors
                agent = self._load_voter_agent(voter)

                # Get messages from neighbors
                neighbor_messages = self.message_protocol.get_messages_for_agent(
                    voter.name
                )

                # Create prompt for this round
                prompt = self._create_message_passing_prompt(voter, neighbor_messages)

                # Get agent response
                response = agent.run(task=prompt)

                # Parse and store messages (simplified)
                try:
                    # Try to extract JSON from response
                    json_match = re.search(r"\{.*\}", response, re.DOTALL)
                    if json_match:
                        messages = json.loads(json_match.group())
                        # Store messages
                        for neighbor, message in messages.items():
                            self.message_protocol.send_message(voter.name, neighbor, message)
                except Exception as e:
                    if self.verbose:
                        election_logger.warning(f"Failed to parse messages from {voter.name}: {e}")

    def _create_message_passing_prompt(
        self,
        voter: VoterProfile,
        neighbor_messages: List[Dict[str, Any]],
    ) -> str:
        """
        Create prompt for message-passing round.

        Args:
            voter: The voter profile
            neighbor_messages: Messages received from neighbors

        Returns:
            str: Prompt for the voter agent
        """
        round_num = self.message_protocol.current_round + 1
        total_rounds = self.message_protocol.rounds
        prompt = f"Round {round_num} of {total_rounds}\n\n"

        if neighbor_messages:
            prompt += "Messages from your neighbors:\n"
            for msg in neighbor_messages:
                prompt += f"From {msg['from']}: {msg['message']}\n"
        else:
            prompt += "No messages from neighbors yet.\n"

        prompt += f"\nCandidates: {[c.name for c in self.candidates]}\n"
        prompt += (
            "\nPlease respond with JSON messages to your neighbors "
            "and your reasoning."
        )

        return prompt


    def _get_voter_decision_with_context(
        self, voter: VoterProfile
    ) -> Optional[VoterDecision]:
        """
        Get voter decision with message-passing context.

        Args:
            voter: The voter profile

        Returns:
            Optional[VoterDecision]: The voter's decision if successful
        """
        try:
            agent = self._load_voter_agent(voter)

            # Get final voting prompt with context
            prompt = self._create_final_voting_prompt(voter)

            # Get response from voter
            response = agent.run(task=prompt)

            # Parse structured response
            decision = self._parse_voter_response(voter, response)

            return decision

        except Exception as e:
            election_logger.error(f"Failed to get decision from {voter.name}: {str(e)}")
            return None

    def _create_final_voting_prompt(self, voter: VoterProfile) -> str:
        """
        Create final voting prompt with message-passing context.

        Args:
            voter: The voter profile

        Returns:
            str: Final voting prompt
        """
        prompt = "FINAL VOTING DECISION\n\n"

        # Include message history
        messages = self.message_protocol.get_messages_for_agent(voter.name)
        if messages:
            prompt += "Based on your communication with neighbors:\n"
            for msg in messages:
                prompt += f"- {msg['from']}: {msg['message']}\n"

        prompt += "\nCandidates:\n"
        for candidate in self.candidates:
            prompt += f"- {candidate.name}: {candidate.policy_positions}\n"

        prompt += """\nProvide your final decision in JSON format:
{
    "rationality": "Your detailed reasoning",
    "vote": "for/against/abstain",
    "confidence": 0.0-1.0,
    "reasoning_factors": ["factor1", "factor2"],
    "candidate_rankings": {"candidate_name": ranking}
}"""

        return prompt

    def _get_voter_decision(self, voter: VoterProfile) -> Optional[VoterDecision]:
        """
        Get a structured decision from a single voter.

        Args:
            voter: The voter profile to get a decision from

        Returns:
            Optional[VoterDecision]: The voter's decision if successful, None otherwise
        """
        try:
            # Load voter agent if needed
            agent = self._load_voter_agent(voter)

            # Create voting prompt
            prompt = self._create_voting_prompt()

            # Get response from voter
            response = agent.run(task=prompt)

            # Parse structured response
            decision = self._parse_voter_response(voter, response)

            return decision

        except Exception as e:
            election_logger.error(f"Failed to get decision from {voter.name}: {str(e)}")
            return None

    def _create_voting_prompt(self) -> str:
        """
        Create a prompt for voters to make decisions.

        Returns:
            str: The voting prompt with candidate information
        """
        candidates_info = []
        for candidate in self.candidates:
            candidates_info.append(
                f"""
            Candidate: {candidate.name}
            Party: {candidate.party_affiliation}
            Policies: {candidate.policy_positions}
            Promises: {candidate.campaign_promises}
            Experience: {candidate.experience}
            """
            )

        candidates_text = chr(10).join(candidates_info)
        
        return f"""You are voting in an election. Analyze the candidates and provide your vote.

CANDIDATES:
{candidates_text}

IMPORTANT: You must respond with ONLY a valid JSON object in this exact format:
{{
    "rationality": "Your reasoning for your vote",
    "vote": "for",
    "confidence": 0.8,
    "reasoning_factors": ["factor1", "factor2"],
    "candidate_rankings": {{
        "Innovation Leader": 1,
        "Stability Leader": 2
    }}
}}

Choose "for" if you support a candidate, "against" if you oppose, or "abstain" if you have no preference."""

    def _parse_voter_response(
        self, voter: VoterProfile, response: str
    ) -> Optional[VoterDecision]:
        """
        Parse a voter's structured response.

        Args:
            voter: The voter profile
            response: The raw response from the voter agent

        Returns:
            Optional[VoterDecision]: Parsed decision if successful, None otherwise
        """
        try:
            # Try to extract JSON from response - look for the first complete JSON object
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                # Clean up any extra text or formatting
                json_str = json_str.strip()
                parsed = json.loads(json_str)
            else:
                # Fallback: try to find any JSON-like structure
                json_match = re.search(r"\{.*?\}", response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed = json.loads(json_str)
                else:
                    raise ValueError("No valid JSON found in response")

            # Create VoterDecision object
            decision = VoterDecision(
                voter_id=voter.voter_id,
                rationality=parsed.get("rationality", ""),
                vote=VoteResult(parsed.get("vote", "abstain")),
                confidence=float(parsed.get("confidence", 0.0)),
                reasoning_factors=parsed.get("reasoning_factors", []),
                candidate_rankings=parsed.get("candidate_rankings", {}),
                tool_call_explanation=None,
            )

            return decision

        except Exception as e:
            election_logger.error(
                f"Failed to parse response from {voter.name}: {str(e)}"
            )
            if self.verbose:
                election_logger.error(f"Raw response from {voter.name}: {response[:500]}...")
            return None

    def _extract_tool_call_explanation(self, response: str) -> Optional[str]:
        """
        Extract tool call explanation from voter response.

        Args:
            response: The raw response from the voter agent

        Returns:
            Optional[str]: Tool call explanation if found, None otherwise
        """
        try:
            # Look for tool call patterns in the response
            # This is a simplified extraction - in practice, you might want to
            # parse the actual tool call structure more carefully
            if "explain_voting_decision" in response:
                # Extract the explanation part
                explanation_match = re.search(
                    r"voting_reasoning[:\s]*([^\n]+)", response, re.IGNORECASE
                )
                if explanation_match:
                    return explanation_match.group(1).strip()
                
                # Fallback: extract any detailed explanation
                explanation_match = re.search(
                    r"reasoning[:\s]*([^\n]+)", response, re.IGNORECASE
                )
                if explanation_match:
                    return explanation_match.group(1).strip()
            
            return None
            
        except Exception as e:
            if self.verbose:
                election_logger.warning(f"Failed to extract tool call explanation: {e}")
            return None

    def _check_consensus(self, decisions: List[VoterDecision]) -> bool:
        """
        Check if consensus has been reached.

        Args:
            decisions: List of voter decisions to check

        Returns:
            bool: True if consensus reached, False otherwise
        """
        if not decisions:
            return False

        # Count votes
        vote_counts = {}
        for decision in decisions:
            vote = decision.vote.value
            vote_counts[vote] = vote_counts.get(vote, 0) + 1

        # Check if any vote has majority
        total_votes = len(decisions)
        for vote, count in vote_counts.items():
            if count / total_votes >= DEFAULT_CONSENSUS_THRESHOLD:
                return True

        return False

    def _determine_winner(self, decisions: List[VoterDecision]) -> Optional[str]:
        """
        Determine the winner based on voter decisions.

        Args:
            decisions: List of voter decisions

        Returns:
            Optional[str]: Name of the winning candidate if found, None otherwise
        """
        if not decisions:
            return None

        # Count votes for each candidate
        candidate_votes = {}
        for decision in decisions:
            if decision.vote == VoteResult.FOR:
                # Find the highest ranked candidate
                if decision.candidate_rankings:
                    winner = min(
                        decision.candidate_rankings.items(),
                        key=lambda x: x[1],
                    )[0]
                    candidate_votes[winner] = candidate_votes.get(winner, 0) + 1

        if not candidate_votes:
            return None

        # Return candidate with most votes
        return max(candidate_votes.items(), key=lambda x: x[1])[0]

    def step(
        self,
        task: str,
        img: Optional[str] = None,
        election_type: ElectionAlgorithm = ElectionAlgorithm.CONSENSUS,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a single step of the ElectionSwarm.

        This method runs one complete election cycle, including voter decision
        collection, candidate evaluation, and result determination.

        Args:
            task: The election task or question to be voted on
            img: Optional image input for the election
            election_type: The AGENTSNET algorithm to use for the election
            max_rounds: Maximum rounds for consensus algorithms
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Any: The result of the election step

        Raises:
            Exception: If step execution fails
        """
        try:
            if self.verbose:
                election_logger.info(
                    f"Executing election step for task: {task[:100]}..."
                )

            # Conduct the election
            result = self.conduct_election(
                election_type=election_type, max_rounds=max_rounds
            )

            # Add to conversation history
            self.conversation.add(
                role="ElectionSwarm",
                content=f"Election completed: {result.winner} won with {result.votes_cast} votes",
            )

            if self.verbose:
                election_logger.info("Election step completed successfully")

            return result

        except Exception as e:
            error_msg = (
                f"Failed to execute election step: {str(e)}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            election_logger.error(error_msg)
            raise

    def run(
        self,
        task: str,
        img: Optional[str] = None,
        election_type: ElectionAlgorithm = ElectionAlgorithm.CONSENSUS,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Run the ElectionSwarm for the specified number of loops.

        This method executes the complete election workflow, including multiple
        iterations if max_loops is greater than 1. Each iteration includes
        election execution and result analysis.

        Args:
            task: The election task or question to be voted on
            img: Optional image input for the election
            election_type: The AGENTSNET algorithm to use for the election
            max_rounds: Maximum rounds for consensus algorithms
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Any: The final result of the election swarm execution

        Raises:
            Exception: If election swarm execution fails
        """
        try:
            if self.verbose:
                election_logger.info(f"Starting ElectionSwarm execution: {self.name}")
                election_logger.info(f"Task: {task[:100]}...")

            current_loop = 0
            while current_loop < self.max_loops:
                if self.verbose:
                    election_logger.info(
                        f"Executing loop {current_loop + 1}/{self.max_loops}"
                    )

                # Execute step
                self.step(
                    task=task,
                    img=img,
                    election_type=election_type,
                    max_rounds=max_rounds,
                    *args,
                    **kwargs,
                )

                # Add to conversation
                self.conversation.add(
                    role="System",
                    content=f"Loop {current_loop + 1} completed",
                )

                current_loop += 1

            if self.verbose:
                election_logger.info(f"ElectionSwarm run completed: {self.name}")
                election_logger.info(f"Total loops executed: {current_loop}")

            return history_output_formatter(
                conversation=self.conversation, type=self.output_type
            )

        except Exception as e:
            error_msg = (
                f"Failed to run ElectionSwarm: {str(e)}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            election_logger.error(error_msg)
            raise

    def run_election_session(
        self,
        election_type: ElectionAlgorithm = ElectionAlgorithm.CONSENSUS,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> Dict[str, Any]:
        """
        Run a complete election session.

        Args:
            election_type: The algorithm to use for the election
            max_rounds: Maximum rounds for consensus algorithms

        Returns:
            Dict containing election results and analysis
        """
        try:
            if self.verbose:
                election_logger.info(
                    f"Starting election session: {election_type.value}"
                )

            # Conduct the election
            result = self.conduct_election(
                election_type=election_type, max_rounds=max_rounds
            )

            # Generate analysis
            analysis = self._generate_election_analysis(result)

            # Get cost statistics
            cost_stats = self.cost_tracker.get_stats()

            session_result = {
                "election_result": result,
                "analysis": analysis,
                "cost_statistics": cost_stats,
                "timestamp": datetime.now().isoformat(),
            }

            if self.verbose:
                election_logger.info("Election session completed successfully")

            return session_result

        except Exception as e:
            error_msg = (
                f"Failed to run election session: {str(e)}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            election_logger.error(error_msg)
            raise

    def _generate_election_analysis(self, result: ElectionResult) -> Dict[str, Any]:
        """
        Generate comprehensive analysis of election results.

        Args:
            result: The election result to analyze

        Returns:
            Dict[str, Any]: Comprehensive analysis of the election
        """
        analysis = {
            "election_summary": {
                "algorithm_used": result.algorithm_used.value,
                "total_voters": result.total_voters,
                "total_candidates": result.total_candidates,
                "votes_cast": result.votes_cast,
                "winner": result.winner,
                "consensus_reached": result.consensus_reached,
                "rounds_to_consensus": result.rounds_to_consensus,
            },
            "vote_analysis": {
                "vote_distribution": {},
                "confidence_stats": {},
                "reasoning_analysis": {},
            },
            "candidate_analysis": {},
            "voter_analysis": {},
        }

        # Analyze votes
        vote_counts = {}
        confidence_scores = []
        reasoning_factors = []

        for decision in result.voter_decisions:
            vote = decision.vote.value
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
            confidence_scores.append(decision.confidence)
            reasoning_factors.extend(decision.reasoning_factors)

        analysis["vote_analysis"]["vote_distribution"] = vote_counts
        analysis["vote_analysis"]["confidence_stats"] = {
            "average": (
                sum(confidence_scores) / len(confidence_scores)
                if confidence_scores
                else 0
            ),
            "min": min(confidence_scores) if confidence_scores else 0,
            "max": max(confidence_scores) if confidence_scores else 0,
        }

        # Count reasoning factors
        factor_counts = {}
        for factor in reasoning_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        analysis["vote_analysis"]["reasoning_analysis"] = factor_counts

        return analysis

    def get_election_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive election statistics.

        Returns:
            Dict[str, Any]: Comprehensive statistics about the election system
        """
        return {
            "swarm_info": {
                "name": self.name,
                "description": self.description,
                "total_voters": len(self.voters),
                "total_candidates": len(self.candidates),
            },
            "cost_statistics": self.cost_tracker.get_stats(),
            "configuration": self.election_config.config.model_dump(),
            "cache_stats": {
                "cache_size": len(self.cache),
                "cache_enabled": self.enable_caching,
            },
        }

    def add_voter(self, voter: VoterProfile) -> None:
        """
        Add a new voter to the election.

        Args:
            voter: The voter profile to add
        """
        self.voters.append(voter)
        if self.verbose:
            election_logger.info(f"Added voter: {voter.name}")

    def remove_voter(self, voter_id: str) -> None:
        """
        Remove a voter from the election.

        Args:
            voter_id: The ID of the voter to remove
        """
        self.voters = [v for v in self.voters if v.voter_id != voter_id]
        if self.verbose:
            election_logger.info(f"Removed voter: {voter_id}")

    def add_candidate(self, candidate: CandidateProfile) -> None:
        """
        Add a new candidate to the election.

        Args:
            candidate: The candidate profile to add
        """
        self.candidates.append(candidate)
        if self.verbose:
            election_logger.info(f"Added candidate: {candidate.name}")

    def remove_candidate(self, candidate_id: str) -> None:
        """
        Remove a candidate from the election.

        Args:
            candidate_id: The ID of the candidate to remove
        """
        self.candidates = [c for c in self.candidates if c.candidate_id != candidate_id]
        if self.verbose:
            election_logger.info(f"Removed candidate: {candidate_id}")

    def get_voter(self, voter_id: str) -> Optional[VoterProfile]:
        """
        Get a voter by ID.

        Args:
            voter_id: The ID of the voter to retrieve

        Returns:
            Optional[VoterProfile]: The voter profile if found, None otherwise
        """
        for voter in self.voters:
            if voter.voter_id == voter_id:
                return voter
        return None

    def get_candidate(self, candidate_id: str) -> Optional[CandidateProfile]:
        """
        Get a candidate by ID.

        Args:
            candidate_id: The ID of the candidate to retrieve

        Returns:
            Optional[CandidateProfile]: The candidate profile if found, None otherwise
        """
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        return None

    def reset(self) -> None:
        """
        Reset the election swarm to its initial state.

        This method clears conversation history, resets cost tracking,
        message protocol, and prepares the swarm for a new election.
        """
        self.conversation = Conversation(time_enabled=False)
        self.cost_tracker = CostTracker(budget_limit=self.budget_limit)
        self.cache.clear()
        self.message_protocol.reset()

        if self.verbose:
            election_logger.info("ElectionSwarm reset to initial state")

    def get_swarm_size(self) -> int:
        """
        Get the total size of the swarm (voters + candidates).

        Returns:
            int: Total number of participants in the election
        """
        return len(self.voters) + len(self.candidates)

    def get_swarm_status(self) -> Dict[str, Any]:
        """
        Get the current status of the election swarm.

        Returns:
            Dict[str, Any]: Status information about the swarm
        """
        return {
            "name": self.name,
            "status": "active",
            "total_participants": self.get_swarm_size(),
            "voters": len(self.voters),
            "candidates": len(self.candidates),
            "max_loops": self.max_loops,
            "budget_remaining": self.cost_tracker.budget_limit
            - self.cost_tracker.total_cost_estimate,
            "cache_size": len(self.cache),
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def create_default_election_config(
    file_path: str = "election_config.yaml",
) -> None:
    """
    Create a default election configuration file.

    Args:
        file_path: Path where to create the configuration file
    """
    default_config = {
        "election_type": "democratic",
        "max_candidates": 5,
        "max_voters": 100,
        "enable_consensus": True,
        "enable_leader_election": True,
        "enable_matching": True,
        "enable_coloring": True,
        "enable_vertex_cover": True,
        "enable_caching": True,
        "enable_voter_tool_calls": True,
        "batch_size": DEFAULT_BATCH_SIZE,
        "max_workers": DEFAULT_MAX_WORKERS,
        "budget_limit": DEFAULT_BUDGET_LIMIT,
        "default_model": "gpt-4o-mini",
        "verbose_logging": False,
    }

    config = ElectionConfig(config_data=default_config)
    config.save_config(file_path)

    election_logger.info(f"Created default election config file: {file_path}")
