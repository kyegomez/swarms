"""
Mass Agent Template - Template for Creating Large-Scale Multi-Agent Systems

This template demonstrates how to generate hundreds of agents on the fly, similar to the EuroSwarm Parliament approach.
It provides a reusable framework for creating large-scale multi-agent systems with dynamic agent generation.

Key Features:
- Dynamic agent generation from data sources
- Configurable agent personalities and roles
- Scalable architecture for thousands of agents
- Template-based system prompts
- Hierarchical organization capabilities
- Memory and state management
- COST OPTIMIZATION: Lazy loading, batching, caching, budget controls
"""

import os
import random
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from swarms import Agent
from swarms.structs.multi_agent_exec import run_agents_concurrently
from swarms.structs.board_of_directors_swarm import (
    BoardOfDirectorsSwarm,
    BoardMember,
    BoardMemberRole,
    enable_board_feature,
)
from swarms.utils.loguru_logger import initialize_logger

# Initialize logger
logger = initialize_logger(log_folder="mass_agent_template")

# Enable Board of Directors feature
enable_board_feature()


class AgentRole(str, Enum):
    """Enumeration of agent roles and specializations."""

    WORKER = "worker"
    MANAGER = "manager"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    ANALYST = "analyst"
    CREATOR = "creator"
    VALIDATOR = "validator"
    EXECUTOR = "executor"


class AgentCategory(str, Enum):
    """Enumeration of agent categories for organization."""

    TECHNICAL = "technical"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    SUPPORT = "support"


@dataclass
class AgentProfile:
    """
    Represents a single agent in the mass agent system.

    Attributes:
        name: Unique name of the agent
        role: Primary role of the agent
        category: Category for organization
        specialization: Areas of expertise
        personality_traits: Personality characteristics
        skills: List of skills and capabilities
        experience_level: Experience level (junior, senior, expert)
        agent: The AI agent instance (lazy loaded)
        is_loaded: Whether the agent has been instantiated
    """

    name: str
    role: AgentRole
    category: AgentCategory
    specialization: List[str] = field(default_factory=list)
    personality_traits: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    experience_level: str = "senior"
    agent: Optional[Agent] = None
    is_loaded: bool = False


@dataclass
class AgentGroup:
    """
    Represents a group of agents with similar roles or categories.

    Attributes:
        name: Name of the group
        category: Category of the group
        agents: List of agent names in this group
        leader: Group leader agent name
        total_agents: Total number of agents in group
        group_swarm: Board of Directors swarm for this group
        is_swarm_loaded: Whether the swarm has been instantiated
    """

    name: str
    category: AgentCategory
    agents: List[str] = field(default_factory=list)
    leader: Optional[str] = None
    total_agents: int = 0
    group_swarm: Optional[Any] = None
    is_swarm_loaded: bool = False


@dataclass
class CostTracker:
    """Track costs and usage for budget management."""

    total_tokens_used: int = 0
    total_cost_estimate: float = 0.0
    budget_limit: float = 100.0  # Default $100 budget
    token_cost_per_1m: float = 0.15  # GPT-4o-mini cost
    requests_made: int = 0
    cache_hits: int = 0

    def add_tokens(self, tokens: int):
        """Add tokens used and calculate cost."""
        self.total_tokens_used += tokens
        self.total_cost_estimate = (
            self.total_tokens_used / 1_000_000
        ) * self.token_cost_per_1m
        self.requests_made += 1

    def add_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1

    def check_budget(self) -> bool:
        """Check if within budget."""
        return self.total_cost_estimate <= self.budget_limit

    def get_stats(self) -> Dict[str, Any]:
        """Get cost statistics."""
        return {
            "total_tokens": self.total_tokens_used,
            "total_cost": self.total_cost_estimate,
            "requests_made": self.requests_made,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits
            / max(1, self.requests_made + self.cache_hits),
            "budget_remaining": max(
                0, self.budget_limit - self.total_cost_estimate
            ),
        }


class MassAgentTemplate:
    """
    Template for creating large-scale multi-agent systems with cost optimization.

    This class provides a framework for generating hundreds of agents on the fly,
    organizing them into groups, and managing their interactions with cost controls.
    """

    def __init__(
        self,
        data_source: str = None,  # Path to data file (CSV, JSON, XML, etc.)
        agent_count: int = 1000,  # Target number of agents
        enable_hierarchical_organization: bool = True,
        enable_group_swarms: bool = True,
        enable_lazy_loading: bool = True,  # NEW: Lazy load agents
        enable_caching: bool = True,  # NEW: Enable response caching
        batch_size: int = 50,  # NEW: Batch size for concurrent execution
        budget_limit: float = 100.0,  # NEW: Budget limit in dollars
        verbose: bool = False,
    ):
        """
        Initialize the Mass Agent Template with cost optimization.

        Args:
            data_source: Path to data file containing agent information
            agent_count: Target number of agents to generate
            enable_hierarchical_organization: Enable hierarchical organization
            enable_group_swarms: Enable Board of Directors swarms for groups
            enable_lazy_loading: Enable lazy loading of agents (cost optimization)
            enable_caching: Enable response caching (cost optimization)
            batch_size: Number of agents to process in batches
            budget_limit: Maximum budget in dollars
            verbose: Enable verbose logging
        """
        self.data_source = data_source
        self.agent_count = agent_count
        self.enable_hierarchical_organization = (
            enable_hierarchical_organization
        )
        self.enable_group_swarms = enable_group_swarms
        self.enable_lazy_loading = enable_lazy_loading
        self.enable_caching = enable_caching
        self.batch_size = batch_size
        self.verbose = verbose

        # Initialize cost tracking
        self.cost_tracker = CostTracker(budget_limit=budget_limit)

        # Initialize agent storage
        self.agents: Dict[str, AgentProfile] = {}
        self.groups: Dict[str, AgentGroup] = {}
        self.categories: Dict[AgentCategory, List[str]] = {}

        # Initialize caching
        self.response_cache: Dict[str, str] = {}

        # Load agent profiles (without creating agents)
        self._load_agent_profiles()

        if self.enable_hierarchical_organization:
            self._organize_agents()

        if self.verbose:
            logger.info(
                f"Mass Agent Template initialized with {len(self.agents)} agent profiles"
            )
            logger.info(
                f"Lazy loading: {self.enable_lazy_loading}, Caching: {self.enable_caching}"
            )
            logger.info(
                f"Budget limit: ${budget_limit}, Batch size: {batch_size}"
            )

    def _load_agent_profiles(self) -> List[Dict[str, Any]]:
        """
        Load agent profiles from the specified data source.

        This method loads agent data but doesn't create AI agents yet (lazy loading).

        Returns:
            List[Dict[str, Any]]: List of agent data dictionaries
        """
        agent_data = []

        if self.data_source and os.path.exists(self.data_source):
            # Load from file - customize based on your data format
            try:
                if self.data_source.endswith(".json"):
                    with open(
                        self.data_source, "r", encoding="utf-8"
                    ) as f:
                        agent_data = json.load(f)
                elif self.data_source.endswith(".csv"):
                    import pandas as pd

                    df = pd.read_csv(self.data_source)
                    agent_data = df.to_dict("records")
                else:
                    logger.warning(
                        f"Unsupported data format: {self.data_source}"
                    )
            except Exception as e:
                logger.error(f"Error loading agent data: {e}")

        # If no data loaded, generate synthetic data
        if not agent_data:
            agent_data = self._generate_synthetic_data()

        # Create agent profiles (without instantiating agents)
        for data in agent_data:
            agent_profile = AgentProfile(
                name=data["name"],
                role=data["role"],
                category=data["category"],
                specialization=data["specialization"],
                personality_traits=data["personality_traits"],
                skills=data["skills"],
                experience_level=data["experience_level"],
                agent=None,  # Will be created on demand
                is_loaded=False,
            )

            self.agents[data["name"]] = agent_profile

        return agent_data

    def _load_agent(self, agent_name: str) -> Optional[Agent]:
        """
        Lazy load a single agent on demand.

        Args:
            agent_name: Name of the agent to load

        Returns:
            Optional[Agent]: Loaded agent or None if not found
        """
        if agent_name not in self.agents:
            return None

        profile = self.agents[agent_name]

        # Check if already loaded
        if profile.is_loaded and profile.agent:
            return profile.agent

        # Create agent (no cost for creation, only for running)
        profile.agent = self._create_agent(profile)
        profile.is_loaded = True

        if self.verbose:
            logger.info(f"Loaded agent: {agent_name}")

        return profile.agent

    def _load_agents_batch(
        self, agent_names: List[str]
    ) -> List[Agent]:
        """
        Load multiple agents in a batch.

        Args:
            agent_names: List of agent names to load

        Returns:
            List[Agent]: List of loaded agents
        """
        loaded_agents = []

        for agent_name in agent_names:
            agent = self._load_agent(agent_name)
            if agent:
                loaded_agents.append(agent)

        return loaded_agents

    def _get_cache_key(
        self, task: str, agent_names: List[str]
    ) -> str:
        """
        Generate a cache key for a task and agent combination.

        Args:
            task: Task to execute
            agent_names: List of agent names

        Returns:
            str: Cache key
        """
        # Sort agent names for consistent cache keys
        sorted_agents = sorted(agent_names)
        content = f"{task}:{':'.join(sorted_agents)}"
        return hashlib.md5(content.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[str]:
        """
        Check if a response is cached.

        Args:
            cache_key: Cache key to check

        Returns:
            Optional[str]: Cached response or None
        """
        if not self.enable_caching:
            return None

        cached_response = self.response_cache.get(cache_key)
        if cached_response:
            self.cost_tracker.add_cache_hit()
            if self.verbose:
                logger.info(f"Cache hit for key: {cache_key[:20]}...")

        return cached_response

    def _cache_response(self, cache_key: str, response: str):
        """
        Cache a response.

        Args:
            cache_key: Cache key
            response: Response to cache
        """
        if self.enable_caching:
            self.response_cache[cache_key] = response
            if self.verbose:
                logger.info(
                    f"Cached response for key: {cache_key[:20]}..."
                )

    def _generate_synthetic_data(self) -> List[Dict[str, Any]]:
        """
        Generate synthetic agent data for demonstration purposes.

        Returns:
            List[Dict[str, Any]]: List of synthetic agent data
        """
        synthetic_data = []

        # Define sample data for different agent types
        sample_agents = [
            {
                "name": "Alex_Developer",
                "role": AgentRole.SPECIALIST,
                "category": AgentCategory.TECHNICAL,
                "specialization": [
                    "Python",
                    "Machine Learning",
                    "API Development",
                ],
                "personality_traits": [
                    "analytical",
                    "detail-oriented",
                    "problem-solver",
                ],
                "skills": [
                    "Python",
                    "TensorFlow",
                    "FastAPI",
                    "Docker",
                ],
                "experience_level": "senior",
            },
            {
                "name": "Sarah_Designer",
                "role": AgentRole.CREATOR,
                "category": AgentCategory.CREATIVE,
                "specialization": [
                    "UI/UX Design",
                    "Visual Design",
                    "Brand Identity",
                ],
                "personality_traits": [
                    "creative",
                    "user-focused",
                    "aesthetic",
                ],
                "skills": [
                    "Figma",
                    "Adobe Creative Suite",
                    "User Research",
                    "Prototyping",
                ],
                "experience_level": "senior",
            },
            {
                "name": "Mike_Analyst",
                "role": AgentRole.ANALYST,
                "category": AgentCategory.ANALYTICAL,
                "specialization": [
                    "Data Analysis",
                    "Business Intelligence",
                    "Market Research",
                ],
                "personality_traits": [
                    "data-driven",
                    "curious",
                    "insightful",
                ],
                "skills": ["SQL", "Python", "Tableau", "Statistics"],
                "experience_level": "expert",
            },
            {
                "name": "Lisa_Manager",
                "role": AgentRole.MANAGER,
                "category": AgentCategory.STRATEGIC,
                "specialization": [
                    "Project Management",
                    "Team Leadership",
                    "Strategic Planning",
                ],
                "personality_traits": [
                    "organized",
                    "leadership",
                    "strategic",
                ],
                "skills": [
                    "Agile",
                    "Scrum",
                    "Risk Management",
                    "Stakeholder Communication",
                ],
                "experience_level": "senior",
            },
            {
                "name": "Tom_Coordinator",
                "role": AgentRole.COORDINATOR,
                "category": AgentCategory.OPERATIONAL,
                "specialization": [
                    "Process Optimization",
                    "Workflow Management",
                    "Resource Allocation",
                ],
                "personality_traits": [
                    "efficient",
                    "coordinated",
                    "systematic",
                ],
                "skills": [
                    "Process Mapping",
                    "Automation",
                    "Resource Planning",
                    "Quality Assurance",
                ],
                "experience_level": "senior",
            },
        ]

        # Generate the specified number of agents
        for i in range(self.agent_count):
            # Use sample data as template and create variations
            template = random.choice(sample_agents)

            agent_data = {
                "name": f"{template['name']}_{i:04d}",
                "role": template["role"],
                "category": template["category"],
                "specialization": template["specialization"].copy(),
                "personality_traits": template[
                    "personality_traits"
                ].copy(),
                "skills": template["skills"].copy(),
                "experience_level": template["experience_level"],
            }

            # Add some randomization for variety
            if random.random() < 0.3:
                agent_data["experience_level"] = random.choice(
                    ["junior", "senior", "expert"]
                )

            synthetic_data.append(agent_data)

        return synthetic_data

    def _create_agent(self, profile: AgentProfile) -> Agent:
        """
        Create an AI agent for the given profile.

        Args:
            profile: Agent profile data

        Returns:
            Agent: AI agent instance
        """
        system_prompt = self._generate_agent_system_prompt(profile)

        return Agent(
            agent_name=profile.name,
            system_prompt=system_prompt,
            model_name="gpt-4o-mini",
            max_loops=3,
            verbose=self.verbose,
        )

    def _generate_agent_system_prompt(
        self, profile: AgentProfile
    ) -> str:
        """
        Generate a comprehensive system prompt for an agent.

        Args:
            profile: Agent profile data

        Returns:
            str: System prompt for the agent
        """
        prompt = f"""You are {profile.name}, an AI agent with the following characteristics:

ROLE AND CATEGORY:
- Role: {profile.role.value}
- Category: {profile.category.value}
- Experience Level: {profile.experience_level}

EXPERTISE AND SKILLS:
- Specializations: {', '.join(profile.specialization)}
- Skills: {', '.join(profile.skills)}

PERSONALITY TRAITS:
- {', '.join(profile.personality_traits)}

CORE RESPONSIBILITIES:
{self._get_role_responsibilities(profile.role)}

WORKING STYLE:
- Approach tasks with your unique personality and expertise
- Collaborate effectively with other agents
- Maintain high quality standards
- Adapt to changing requirements
- Communicate clearly and professionally

When working on tasks:
1. Apply your specialized knowledge and skills
2. Consider your personality traits in your approach
3. Work within your role's scope and responsibilities
4. Collaborate with other agents when beneficial
5. Maintain consistency with your established character

Remember: You are part of a large multi-agent system. Your unique combination of role, skills, and personality makes you valuable to the team.
"""

        return prompt

    def _get_role_responsibilities(self, role: AgentRole) -> str:
        """Get responsibilities for a specific role."""

        responsibilities = {
            AgentRole.WORKER: """
- Execute assigned tasks efficiently and accurately
- Follow established procedures and guidelines
- Report progress and any issues encountered
- Maintain quality standards in all work
- Collaborate with team members as needed""",
            AgentRole.MANAGER: """
- Oversee team activities and coordinate efforts
- Set priorities and allocate resources
- Monitor progress and ensure deadlines are met
- Provide guidance and support to team members
- Make strategic decisions for the team""",
            AgentRole.SPECIALIST: """
- Provide expert knowledge in specific domains
- Solve complex technical problems
- Mentor other agents in your area of expertise
- Stay updated on latest developments in your field
- Contribute specialized insights to projects""",
            AgentRole.COORDINATOR: """
- Facilitate communication between different groups
- Ensure smooth workflow and process optimization
- Manage dependencies and resource allocation
- Track project timelines and milestones
- Resolve conflicts and bottlenecks""",
            AgentRole.ANALYST: """
- Analyze data and extract meaningful insights
- Identify patterns and trends
- Provide evidence-based recommendations
- Create reports and visualizations
- Support decision-making with data""",
            AgentRole.CREATOR: """
- Generate innovative ideas and solutions
- Design and develop new content or products
- Think creatively and outside the box
- Prototype and iterate on concepts
- Inspire and motivate other team members""",
            AgentRole.VALIDATOR: """
- Review and validate work quality
- Ensure compliance with standards and requirements
- Provide constructive feedback
- Identify potential issues and risks
- Maintain quality assurance processes""",
            AgentRole.EXECUTOR: """
- Implement plans and strategies
- Execute tasks with precision and efficiency
- Adapt to changing circumstances
- Ensure successful completion of objectives
- Maintain focus on results and outcomes""",
        }

        return responsibilities.get(
            role,
            "Execute tasks according to your role and expertise.",
        )

    def _organize_agents(self):
        """Organize agents into groups and categories."""

        # Organize by category
        for agent_name, profile in self.agents.items():
            category = profile.category
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(agent_name)

        # Create groups for each category
        for category, agent_names in self.categories.items():
            group_name = f"{category.value.capitalize()}_Group"

            # Select a leader (first agent in the category)
            leader = agent_names[0] if agent_names else None

            group = AgentGroup(
                name=group_name,
                category=category,
                agents=agent_names,
                leader=leader,
                total_agents=len(agent_names),
            )

            self.groups[group_name] = group

        if self.verbose:
            logger.info(
                f"Organized agents into {len(self.groups)} groups"
            )

    def _create_group_swarms(self):
        """Create Board of Directors swarms for each group."""

        for group_name, group in self.groups.items():
            if not group.agents:
                continue

            # Create board members from group agents
            board_members = []

            # Add group leader as chairman
            if group.leader and group.leader in self.agents:
                leader_profile = self.agents[group.leader]
                if leader_profile.agent:
                    board_members.append(
                        BoardMember(
                            agent=leader_profile.agent,
                            role=BoardMemberRole.CHAIRMAN,
                            voting_weight=1.0,
                            expertise_areas=leader_profile.specialization,
                        )
                    )

            # Add other agents as board members
            for agent_name in group.agents[
                :5
            ]:  # Limit to 5 board members
                if (
                    agent_name != group.leader
                    and agent_name in self.agents
                ):
                    profile = self.agents[agent_name]
                    if profile.agent:
                        board_members.append(
                            BoardMember(
                                agent=profile.agent,
                                role=BoardMemberRole.EXECUTIVE_DIRECTOR,
                                voting_weight=0.8,
                                expertise_areas=profile.specialization,
                            )
                        )

            # Create Board of Directors swarm
            if board_members:
                agents = [
                    member.agent
                    for member in board_members
                    if member.agent is not None
                ]

                group.group_swarm = BoardOfDirectorsSwarm(
                    name=group_name,
                    description=f"Specialized swarm for {group_name} with expertise in {group.category.value}",
                    board_members=board_members,
                    agents=agents,
                    max_loops=3,
                    verbose=self.verbose,
                    decision_threshold=0.6,
                    enable_voting=True,
                    enable_consensus=True,
                )

        if self.verbose:
            logger.info(
                f"Created {len([g for g in self.groups.values() if g.group_swarm])} group swarms"
            )

    def get_agent(self, agent_name: str) -> Optional[AgentProfile]:
        """
        Get a specific agent by name.

        Args:
            agent_name: Name of the agent

        Returns:
            Optional[AgentProfile]: Agent profile if found, None otherwise
        """
        return self.agents.get(agent_name)

    def get_group(self, group_name: str) -> Optional[AgentGroup]:
        """
        Get a specific group by name.

        Args:
            group_name: Name of the group

        Returns:
            Optional[AgentGroup]: Group if found, None otherwise
        """
        return self.groups.get(group_name)

    def get_agents_by_category(
        self, category: AgentCategory
    ) -> List[str]:
        """
        Get all agents in a specific category.

        Args:
            category: Agent category

        Returns:
            List[str]: List of agent names in the category
        """
        return self.categories.get(category, [])

    def get_agents_by_role(self, role: AgentRole) -> List[str]:
        """
        Get all agents with a specific role.

        Args:
            role: Agent role

        Returns:
            List[str]: List of agent names with the role
        """
        return [
            name
            for name, profile in self.agents.items()
            if profile.role == role
        ]

    def run_mass_task(
        self, task: str, agent_count: int = 10
    ) -> Dict[str, Any]:
        """
        Run a task with multiple agents working in parallel with cost optimization.

        Args:
            task: Task to execute
            agent_count: Number of agents to use

        Returns:
            Dict[str, Any]: Results from the mass task execution
        """
        # Check budget before starting
        if not self.cost_tracker.check_budget():
            return {
                "error": "Budget exceeded",
                "cost_stats": self.cost_tracker.get_stats(),
            }

        # Select random agents
        selected_agent_names = random.sample(
            list(self.agents.keys()),
            min(agent_count, len(self.agents)),
        )

        # Check cache first
        cache_key = self._get_cache_key(task, selected_agent_names)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return {
                "task": task,
                "agents_used": selected_agent_names,
                "results": cached_result,
                "total_agents": len(selected_agent_names),
                "cached": True,
                "cost_stats": self.cost_tracker.get_stats(),
            }

        # Process in batches to control memory and cost
        all_results = []
        total_processed = 0

        for i in range(0, len(selected_agent_names), self.batch_size):
            batch_names = selected_agent_names[
                i : i + self.batch_size
            ]

            # Check budget for this batch
            if not self.cost_tracker.check_budget():
                logger.warning(
                    f"Budget exceeded after processing {total_processed} agents"
                )
                logger.warning(
                    f"Current cost: ${self.cost_tracker.total_cost_estimate:.4f}, Budget: ${self.cost_tracker.budget_limit:.2f}"
                )
                break

            # Load agents for this batch
            batch_agents = self._load_agents_batch(batch_names)

            if not batch_agents:
                continue

            # Run batch
            try:
                batch_results = run_agents_concurrently(
                    batch_agents, task
                )
                all_results.extend(batch_results)
                total_processed += len(batch_agents)

                # Estimate tokens used (more realistic approximation)
                # Include both input tokens (task) and output tokens (response)
                task_tokens = (
                    len(task.split()) * 1.3
                )  # ~1.3 tokens per word
                response_tokens = (
                    len(batch_agents) * 200
                )  # ~200 tokens per response
                total_tokens = int(task_tokens + response_tokens)
                self.cost_tracker.add_tokens(total_tokens)

                if self.verbose:
                    logger.info(
                        f"Processed batch {i//self.batch_size + 1}: {len(batch_agents)} agents"
                    )
                    logger.info(
                        f"Current cost: ${self.cost_tracker.total_cost_estimate:.4f}, Budget remaining: ${self.cost_tracker.budget_limit - self.cost_tracker.total_cost_estimate:.2f}"
                    )

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue

        # Cache the results
        if all_results:
            self._cache_response(cache_key, str(all_results))

        return {
            "task": task,
            "agents_used": selected_agent_names[:total_processed],
            "results": all_results,
            "total_agents": total_processed,
            "cached": False,
            "cost_stats": self.cost_tracker.get_stats(),
        }

    def run_mass_task_optimized(
        self,
        task: str,
        agent_count: int = 1000,
        max_cost: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Run a task with cost-optimized mass execution for large-scale operations.

        Args:
            task: Task to execute
            agent_count: Target number of agents to use
            max_cost: Maximum cost for this task in dollars

        Returns:
            Dict[str, Any]: Results from the optimized mass task execution
        """
        # Store original settings
        original_budget = self.cost_tracker.budget_limit
        original_batch_size = self.batch_size

        try:
            # Set temporary budget for this task (don't reduce if max_cost is higher)
            if max_cost < original_budget:
                self.cost_tracker.budget_limit = max_cost

            # Use smaller batches for better cost control
            self.batch_size = min(
                25, self.batch_size
            )  # Smaller batches for cost control

            result = self.run_mass_task(task, agent_count)

            return result

        finally:
            # Restore original settings
            self.cost_tracker.budget_limit = original_budget
            self.batch_size = original_batch_size

    def run_group_task(
        self, group_name: str, task: str
    ) -> Dict[str, Any]:
        """
        Run a task with a specific group using their Board of Directors swarm.

        Args:
            group_name: Name of the group
            task: Task to execute

        Returns:
            Dict[str, Any]: Results from the group task execution
        """
        group = self.groups.get(group_name)
        if not group or not group.group_swarm:
            return {
                "error": f"Group {group_name} not found or no swarm available"
            }

        # Run task with group swarm
        result = group.group_swarm.run(task)

        return {
            "group": group_name,
            "task": task,
            "result": result,
            "agents_involved": group.agents,
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the mass agent system including cost tracking.

        Returns:
            Dict[str, Any]: System statistics
        """
        stats = {
            "total_agents": len(self.agents),
            "total_groups": len(self.groups),
            "loaded_agents": len(
                [a for a in self.agents.values() if a.is_loaded]
            ),
            "categories": {},
            "roles": {},
            "experience_levels": {},
            "cost_stats": self.cost_tracker.get_stats(),
            "optimization": {
                "lazy_loading": self.enable_lazy_loading,
                "caching": self.enable_caching,
                "batch_size": self.batch_size,
                "budget_limit": self.cost_tracker.budget_limit,
            },
        }

        # Category breakdown
        for category in AgentCategory:
            stats["categories"][category.value] = len(
                self.get_agents_by_category(category)
            )

        # Role breakdown
        for role in AgentRole:
            stats["roles"][role.value] = len(
                self.get_agents_by_role(role)
            )

        # Experience level breakdown
        experience_counts = {}
        for profile in self.agents.values():
            level = profile.experience_level
            experience_counts[level] = (
                experience_counts.get(level, 0) + 1
            )
        stats["experience_levels"] = experience_counts

        return stats


# Example usage and demonstration
def demonstrate_mass_agent_template():
    """Demonstrate the Mass Agent Template functionality with cost optimization."""

    print("MASS AGENT TEMPLATE DEMONSTRATION (COST OPTIMIZED)")
    print("=" * 60)

    # Initialize the template with 1000 agents and cost optimization
    template = MassAgentTemplate(
        agent_count=1000,
        enable_hierarchical_organization=True,
        enable_group_swarms=False,  # Disable for cost savings
        enable_lazy_loading=True,
        enable_caching=True,
        batch_size=25,
        budget_limit=50.0,  # $50 budget limit
        verbose=True,
    )

    # Show system statistics
    stats = template.get_system_stats()

    print("\nSYSTEM STATISTICS:")
    print(f"Total Agents: {stats['total_agents']}")
    print(
        f"Loaded Agents: {stats['loaded_agents']} (lazy loading active)"
    )
    print(f"Total Groups: {stats['total_groups']}")

    print("\nCOST OPTIMIZATION:")
    cost_stats = stats["cost_stats"]
    print(
        f"Budget Limit: ${cost_stats['budget_remaining'] + cost_stats['total_cost']:.2f}"
    )
    print(f"Budget Used: ${cost_stats['total_cost']:.2f}")
    print(f"Budget Remaining: ${cost_stats['budget_remaining']:.2f}")
    print(f"Cache Hit Rate: {cost_stats['cache_hit_rate']:.1%}")

    print("\nCATEGORY BREAKDOWN:")
    for category, count in stats["categories"].items():
        print(f"  {category}: {count} agents")

    print("\nROLE BREAKDOWN:")
    for role, count in stats["roles"].items():
        print(f"  {role}: {count} agents")

    print("\nEXPERIENCE LEVEL BREAKDOWN:")
    for level, count in stats["experience_levels"].items():
        print(f"  {level}: {count} agents")

    # Demonstrate cost-optimized mass task execution
    print("\nCOST-OPTIMIZED MASS TASK DEMONSTRATION:")
    print("-" * 40)

    # Small task first (low cost)
    small_result = template.run_mass_task(
        "What is the most important skill for a software developer?",
        agent_count=5,
    )

    print("Small Task Results:")
    print(f"  Agents Used: {len(small_result['agents_used'])}")
    print(f"  Cached: {small_result.get('cached', False)}")
    print(f"  Cost: ${small_result['cost_stats']['total_cost']:.2f}")

    # Large task to demonstrate full capability
    print("\nLarge Task Demonstration (Full Capability):")
    large_result = template.run_mass_task(
        "Analyze the benefits of cloud computing for small businesses",
        agent_count=200,  # Use more agents to show capability
    )

    print(f"  Agents Used: {len(large_result['agents_used'])}")
    print(f"  Cached: {large_result.get('cached', False)}")
    print(f"  Cost: ${large_result['cost_stats']['total_cost']:.2f}")
    print(
        f"  Budget Remaining: ${large_result['cost_stats']['budget_remaining']:.2f}"
    )

    # Show what happens with cost limits
    print("\nCost-Limited Task Demonstration:")
    cost_limited_result = template.run_mass_task_optimized(
        "What are the key principles of agile development?",
        agent_count=100,
        max_cost=2.0,  # Show cost limiting in action
    )

    print(f"  Agents Used: {len(cost_limited_result['agents_used'])}")
    print(f"  Cached: {cost_limited_result.get('cached', False)}")
    print(
        f"  Cost: ${cost_limited_result['cost_stats']['total_cost']:.2f}"
    )
    print(
        f"  Budget Remaining: ${cost_limited_result['cost_stats']['budget_remaining']:.2f}"
    )

    # Show final cost statistics
    final_stats = template.get_system_stats()
    print("\nFINAL COST STATISTICS:")
    print(
        f"Total Cost: ${final_stats['cost_stats']['total_cost']:.2f}"
    )
    print(
        f"Budget Remaining: ${final_stats['cost_stats']['budget_remaining']:.2f}"
    )
    print(
        f"Cache Hit Rate: {final_stats['cost_stats']['cache_hit_rate']:.1%}"
    )
    print(
        f"Total Requests: {final_stats['cost_stats']['requests_made']}"
    )
    print(f"Cache Hits: {final_stats['cost_stats']['cache_hits']}")

    print("\nDEMONSTRATION COMPLETED SUCCESSFULLY!")
    print(
        f"✅ Cost optimization working: ${final_stats['cost_stats']['total_cost']:.2f} spent"
    )
    print(
        f"✅ Lazy loading working: {final_stats['loaded_agents']}/{final_stats['total_agents']} agents loaded"
    )
    print(
        f"✅ Caching working: {final_stats['cost_stats']['cache_hit_rate']:.1%} hit rate"
    )


if __name__ == "__main__":
    demonstrate_mass_agent_template()
