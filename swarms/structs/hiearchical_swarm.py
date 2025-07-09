import json
import os
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Union, Dict, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque
import threading

from pydantic import BaseModel, Field, validator

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.conversation import Conversation
from swarms.utils.output_types import OutputType
from swarms.utils.any_to_str import any_to_str
from swarms.utils.formatter import formatter

from swarms.utils.function_caller_model import OpenAIFunctionCaller
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.structs.ma_utils import list_all_agents

logger = initialize_logger(log_folder="hierarchical_swarm")


class AgentState(Enum):
    """Agent state enumeration for tracking agent status"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    DISABLED = "disabled"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class TaskResult:
    """Result of a task execution"""
    agent_name: str
    task: str
    output: str
    success: bool
    execution_time: float
    timestamp: float
    error: Optional[str] = None


@dataclass
class AgentHealth:
    """Agent health monitoring data"""
    agent_name: str
    state: AgentState
    last_activity: float
    task_count: int
    success_rate: float
    avg_response_time: float
    consecutive_failures: int


class HierarchicalOrder(BaseModel):
    agent_name: str = Field(
        ...,
        description="Specifies the name of the agent to which the task is assigned. This is a crucial element in the hierarchical structure of the swarm, as it determines the specific agent responsible for the task execution.",
    )
    task: str = Field(
        ...,
        description="Defines the specific task to be executed by the assigned agent. This task is a key component of the swarm's plan and is essential for achieving the swarm's goals.",
    )
    priority: TaskPriority = Field(
        default=TaskPriority.MEDIUM,
        description="Priority level of the task affecting execution order"
    )
    timeout: Optional[int] = Field(
        default=300,
        description="Timeout in seconds for task execution"
    )
    retry_count: int = Field(
        default=3,
        description="Number of retry attempts for failed tasks"
    )
    depends_on: Optional[List[str]] = Field(
        default=None,
        description="List of agent names whose tasks must complete before this one"
    )

    @validator('retry_count')
    def validate_retry_count(cls, v):
        if v < 0:
            raise ValueError('retry_count must be non-negative')
        return v

    @validator('timeout')
    def validate_timeout(cls, v):
        if v is not None and v <= 0:
            raise ValueError('timeout must be positive')
        return v


class SwarmSpec(BaseModel):
    goals: str = Field(
        ...,
        description="The goal of the swarm. This is the overarching objective that the swarm is designed to achieve. It guides the swarm's plan and the tasks assigned to the agents.",
    )
    plan: str = Field(
        ...,
        description="Outlines the sequence of actions to be taken by the swarm. This plan is a detailed roadmap that guides the swarm's behavior and decision-making.",
    )
    rules: str = Field(
        ...,
        description="Defines the governing principles for swarm behavior and decision-making. These rules are the foundation of the swarm's operations and ensure that the swarm operates in a coordinated and efficient manner.",
    )
    orders: List[HierarchicalOrder] = Field(
        ...,
        description="A collection of task assignments to specific agents within the swarm. These orders are the specific instructions that guide the agents in their task execution and are a key element in the swarm's plan.",
    )
    max_concurrent_tasks: int = Field(
        default=5,
        description="Maximum number of tasks that can be executed concurrently"
    )
    failure_threshold: float = Field(
        default=0.3,
        description="Maximum failure rate before swarm enters degraded mode"
    )

    @validator('max_concurrent_tasks')
    def validate_max_concurrent_tasks(cls, v):
        if v <= 0:
            raise ValueError('max_concurrent_tasks must be positive')
        return v

    @validator('failure_threshold')
    def validate_failure_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('failure_threshold must be between 0 and 1')
        return v


HIEARCHICAL_SWARM_SYSTEM_PROMPT = """
**SYSTEM PROMPT: HIERARCHICAL AGENT DIRECTOR**

**I. Introduction and Context**

You are the Hierarchical Agent Director – the central orchestrator responsible for breaking down overarching goals into granular tasks and intelligently assigning these tasks to the most suitable worker agents within the swarm. Your objective is to maximize the overall performance of the system by ensuring that every agent is given a task aligned with its strengths, expertise, and available resources.

---

**II. Core Operating Principles**

1. **Goal Alignment and Context Awareness:**  
   - **Overarching Goals:** Begin every operation by clearly reviewing the swarm's overall goals. Understand the mission statement and ensure that every assigned task contributes directly to these objectives.
   - **Context Sensitivity:** Evaluate the context provided in the "plan" and "rules" sections of the SwarmSpec. These instructions provide the operational boundaries and behavioral constraints within which you must work.

2. **Task Decomposition and Prioritization:**  
   - **Hierarchical Decomposition:** Break down the overarching plan into granular tasks. For each major objective, identify subtasks that logically lead toward the goal. This decomposition should be structured in a hierarchical manner, where complex tasks are subdivided into simpler, manageable tasks.
   - **Task Priority:** Assign a priority level to each task based on urgency, complexity, and impact. Use HIGH priority for critical tasks, MEDIUM for standard tasks, and LOW for background tasks.

3. **Agent Profiling and Matching:**  
   - **Agent Specialization:** Maintain an up-to-date registry of worker agents, each with defined capabilities, specializations, and performance histories. When assigning tasks, consider the specific strengths of each agent.
   - **Performance Metrics:** Utilize historical performance metrics and available workload data to select the most suitable agent for each task. If an agent is overburdened or has lower efficiency on a specific type of task, consider alternate agents.
   - **Dynamic Reassignment:** Allow for real-time reassignments based on the evolving state of the system. If an agent encounters issues or delays, reassign tasks to ensure continuity.

4. **Reliability and Error Handling:**  
   - **Failure Tolerance:** Design tasks with appropriate retry mechanisms and fallback options.
   - **Dependency Management:** Properly sequence tasks with dependencies to ensure logical execution order.
   - **Timeout Management:** Set appropriate timeouts for tasks based on complexity and criticality.

5. **Concurrency and Performance:**  
   - **Parallel Execution:** Identify tasks that can be executed concurrently to maximize throughput.
   - **Resource Optimization:** Balance workload across available agents to prevent bottlenecks.
   - **Scalability:** Design task distribution to handle varying workloads efficiently.

---

**III. Enhanced Task Assignment Process**

1. **Input Analysis and Context Setting:**
   - **Goal Review:** Begin by carefully reading the "goals" string within the SwarmSpec. This is your north star for every decision you make.
   - **Plan Comprehension:** Analyze the "plan" string for detailed instructions. Identify key milestones, deliverables, and dependencies within the roadmap.
   - **Rule Enforcement:** Read through the "rules" string to understand the non-negotiable guidelines that govern task assignments.

2. **Advanced Task Breakdown:**
   - **Decompose the Plan:** Using a systematic approach, decompose the overall plan into discrete tasks with clear dependencies.
   - **Task Granularity:** Ensure tasks are actionable and appropriately sized for individual agents.
   - **Priority Assignment:** Assign priority levels (HIGH, MEDIUM, LOW) based on criticality and impact.
   - **Dependency Mapping:** Identify and specify task dependencies to ensure proper execution order.

3. **Intelligent Agent Selection:**
   - **Capabilities Matching:** Match task requirements with agent capabilities and specializations.
   - **Load Balancing:** Consider current agent workload and distribute tasks evenly.
   - **Performance History:** Use historical performance data to make optimal assignments.
   - **Fallback Options:** Identify backup agents for critical tasks.

4. **Enhanced Order Construction:**
   - **Comprehensive Orders:** Create HierarchicalOrder objects with all necessary parameters including priority, timeout, retry count, and dependencies.
   - **Validation:** Ensure all orders are valid and executable within the swarm's constraints.
   - **Optimization:** Optimize task ordering for maximum efficiency and minimal resource contention.

Remember: You must create orders that are executable, well-prioritized, and designed for reliability. Consider agent capabilities, task dependencies, and system constraints when making assignments.
"""


class TeamUnit(BaseModel):
    """Represents a team within a department."""

    name: Optional[str] = Field(
        None, description="The name of the team."
    )
    description: Optional[str] = Field(
        None, description="A brief description of the team's purpose."
    )
    agents: Optional[List[Union[Agent, Any]]] = Field(
        None,
        description="A list of agents that are part of the team.",
    )
    team_leader: Optional[Union[Agent, Any]] = Field(
        None, description="The team leader of the team."
    )
    max_concurrent_tasks: int = Field(
        default=3,
        description="Maximum concurrent tasks for this team"
    )

    class Config:
        arbitrary_types_allowed = True


class HierarchicalSwarm(BaseSwarm):
    """
    Enhanced hierarchical swarm with improved reliability, error handling, and performance.
    
    Features:
    - Concurrent task execution
    - Retry mechanisms with exponential backoff
    - Agent health monitoring
    - Graceful degradation
    - Task dependency management
    - Performance metrics tracking
    - Load balancing
    """

    def __init__(
        self,
        name: str = "HierarchicalAgentSwarm",
        description: str = "Enhanced distributed task swarm",
        director: Optional[Union[Agent, Any]] = None,
        agents: List[Union[Agent, Any]] = None,
        max_loops: int = 1,
        output_type: OutputType = "dict-all-except-first",
        director_model_name: str = "gpt-4o",
        teams: Optional[List[TeamUnit]] = None,
        inter_agent_loops: int = 1,
        max_concurrent_tasks: int = 5,
        task_timeout: int = 300,
        retry_attempts: int = 3,
        health_check_interval: float = 30.0,
        failure_threshold: float = 0.3,
        enable_monitoring: bool = True,
        enable_load_balancing: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialize the enhanced HierarchicalSwarm.

        Args:
            name: The name of the swarm
            description: A description of the swarm
            director: The director agent that orchestrates tasks
            agents: A list of agents within the swarm
            max_loops: Maximum number of feedback loops
            output_type: Format for output return
            director_model_name: Model name for the director
            teams: Optional list of team units
            inter_agent_loops: Number of inter-agent loops
            max_concurrent_tasks: Maximum concurrent tasks
            task_timeout: Default timeout for tasks
            retry_attempts: Default retry attempts
            health_check_interval: Interval for health checks
            failure_threshold: Failure threshold for degraded mode
            enable_monitoring: Enable agent monitoring
            enable_load_balancing: Enable load balancing
        """
        super().__init__(
            name=name,
            description=description,
            agents=agents,
        )
        self.director = director
        self.agents = agents or []
        self.max_loops = max_loops
        self.output_type = output_type
        self.director_model_name = director_model_name
        self.teams = teams or []
        self.inter_agent_loops = inter_agent_loops
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_timeout = task_timeout
        self.retry_attempts = retry_attempts
        self.health_check_interval = health_check_interval
        self.failure_threshold = failure_threshold
        self.enable_monitoring = enable_monitoring
        self.enable_load_balancing = enable_load_balancing

        # Initialize enhanced components
        self.conversation = Conversation(time_enabled=True)
        self.current_loop = 0
        self.agent_outputs = {}
        self.task_results = []
        self.agent_health = {}
        self.task_queue = deque()
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.performance_metrics = {}
        self._shutdown_event = threading.Event()
        self._health_monitor_thread = None
        self._task_executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)

        # Initialize swarm
        self._initialize_swarm()

    def _initialize_swarm(self):
        """Initialize the swarm with enhanced features"""
        self.add_name_and_description()
        self.reliability_checks()
        self.handle_teams()
        self._initialize_agent_health()
        list_all_agents(self.agents, self.conversation, self.name)
        self.director = self.setup_director()
        
        if self.enable_monitoring:
            self._start_health_monitor()
            
        logger.info(f"Enhanced hierarchical swarm '{self.name}' initialized successfully")

    def _initialize_agent_health(self):
        """Initialize health monitoring for all agents"""
        current_time = time.time()
        for agent in self.agents:
            if hasattr(agent, 'agent_name') and agent.agent_name:
                self.agent_health[agent.agent_name] = AgentHealth(
                    agent_name=agent.agent_name,
                    state=AgentState.IDLE,
                    last_activity=current_time,
                    task_count=0,
                    success_rate=1.0,
                    avg_response_time=0.0,
                    consecutive_failures=0
                )

    def _start_health_monitor(self):
        """Start the health monitoring thread"""
        if self._health_monitor_thread is None:
            self._health_monitor_thread = threading.Thread(
                target=self._health_monitor_loop,
                daemon=True
            )
            self._health_monitor_thread.start()

    def _health_monitor_loop(self):
        """Health monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                self._check_agent_health()
                self._update_performance_metrics()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    def _check_agent_health(self):
        """Check health of all agents"""
        current_time = time.time()
        for agent_name, health in self.agent_health.items():
            # Check for inactive agents
            if current_time - health.last_activity > self.health_check_interval * 2:
                if health.state == AgentState.RUNNING:
                    logger.warning(f"Agent {agent_name} appears to be stuck")
                    health.state = AgentState.FAILED
                    health.consecutive_failures += 1

            # Check failure rate
            if health.success_rate < self.failure_threshold:
                logger.warning(f"Agent {agent_name} has high failure rate: {health.success_rate:.2f}")
                if health.consecutive_failures >= 3:
                    health.state = AgentState.DISABLED
                    logger.error(f"Agent {agent_name} disabled due to consecutive failures")

    def _update_performance_metrics(self):
        """Update performance metrics"""
        total_tasks = len(self.task_results)
        if total_tasks > 0:
            successful_tasks = sum(1 for result in self.task_results if result.success)
            self.performance_metrics['success_rate'] = successful_tasks / total_tasks
            self.performance_metrics['total_tasks'] = total_tasks
            self.performance_metrics['avg_execution_time'] = sum(
                result.execution_time for result in self.task_results
            ) / total_tasks

    def handle_teams(self):
        """Enhanced team handling with load balancing"""
        if not self.teams:
            return

        team_agents = []
        for team in self.teams:
            if team.agents:
                team_agents.extend(team.agents)
                # Set up team-specific configurations
                for agent in team.agents:
                    # Only set team_name if agent supports it
                    if hasattr(agent, '__dict__'):
                        agent.__dict__['team_name'] = team.name

        self.agents.extend(team_agents)
        
        # Add team information to conversation
        team_info = [
            {
                "name": team.name,
                "description": team.description,
                "agent_count": len(team.agents) if team.agents else 0,
                "max_concurrent_tasks": team.max_concurrent_tasks
            }
            for team in self.teams
        ]
        
        self.conversation.add(
            role="System",
            content=f"Teams Available: {any_to_str(team_info)}",
        )

    def setup_director(self):
        """Set up the director with enhanced capabilities"""
        director = OpenAIFunctionCaller(
            model_name=self.director_model_name,
            system_prompt=HIEARCHICAL_SWARM_SYSTEM_PROMPT,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.5,
            base_model=SwarmSpec,
            max_tokens=10000,
        )
        return director

    def reliability_checks(self):
        """Enhanced reliability checks"""
        logger.info(f"🔍 PERFORMING ENHANCED RELIABILITY CHECKS: {self.name}")

        # Basic checks
        if not self.agents:
            raise ValueError("No agents found in the swarm. At least one agent must be provided.")

        if self.max_loops <= 0:
            raise ValueError("Max loops must be greater than 0.")

        # Enhanced checks
        if self.max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks must be greater than 0.")

        if self.task_timeout <= 0:
            raise ValueError("task_timeout must be greater than 0.")

        if not 0 <= self.failure_threshold <= 1:
            raise ValueError("failure_threshold must be between 0 and 1.")

        # Validate agents
        for i, agent in enumerate(self.agents):
            if not hasattr(agent, 'agent_name') or not agent.agent_name:
                raise ValueError(f"Agent {i} must have a valid agent_name.")
            if not hasattr(agent, 'run') or not callable(agent.run):
                raise ValueError(f"Agent {i} must have a callable 'run' method.")

        # Set up director
        if self.director is None:
            self.director = self.agents[0]
            logger.info(f"Director not specified, using first agent: {self.director.agent_name}")

        logger.info(f"✅ ENHANCED RELIABILITY CHECKS PASSED: {self.name}")

    def get_healthy_agents(self) -> List[Agent]:
        """Get list of healthy agents available for task assignment"""
        healthy_agents = []
        for agent in self.agents:
            health = self.agent_health.get(agent.agent_name)
            if health and health.state not in [AgentState.DISABLED, AgentState.FAILED]:
                healthy_agents.append(agent)
        return healthy_agents

    def select_best_agent(self, task: str, exclude_agents: Optional[List[str]] = None) -> Optional[Agent]:
        """Select the best agent for a task based on health and load balancing"""
        if exclude_agents is None:
            exclude_agents = []
            
        available_agents = [
            agent for agent in self.get_healthy_agents()
            if agent.agent_name not in exclude_agents
        ]

        if not available_agents:
            return None

        if not self.enable_load_balancing:
            return available_agents[0]

        # Select agent with lowest current load and best performance
        best_agent = None
        best_score = float('inf')

        for agent in available_agents:
            health = self.agent_health.get(agent.agent_name)
            if health is None:
                continue
                
            # Calculate load score (lower is better)
            current_load = sum(1 for task_name, task_info in self.running_tasks.items()
                             if task_info.get('agent_name') == agent.agent_name)
            load_score = current_load / max(health.success_rate, 0.1)
            
            if load_score < best_score:
                best_score = load_score
                best_agent = agent

        return best_agent

    def run_director(self, task: str, loop_context: str = "", img: str = None) -> SwarmSpec:
        """Run the director with enhanced context and error handling"""
        try:
            # Build comprehensive context
            agent_status = self._get_agent_status_summary()
            performance_summary = self._get_performance_summary()
            
            director_context = f"""
Swarm Status: {agent_status}
Performance: {performance_summary}
History: {self.conversation.get_str()}
"""
            
            if loop_context:
                director_context += f"\n\nCurrent Loop ({self.current_loop}/{self.max_loops}): {loop_context}"
            
            director_context += f"\n\nYour Task: {task}"

            # Run director with timeout
            start_time = time.time()
            function_call = self.director.run(task=director_context)
            execution_time = time.time() - start_time

            # Log director output
            formatter.print_panel(
                f"Director Output (Loop {self.current_loop}/{self.max_loops}):\n{function_call}",
                title="Director's Orders",
            )

            # Add to conversation
            self.conversation.add(
                role="Director",
                content=f"Loop {self.current_loop}/{self.max_loops}: {function_call}",
            )

            # Validate SwarmSpec
            if not isinstance(function_call, SwarmSpec):
                logger.error("Director did not return a valid SwarmSpec")
                raise ValueError("Invalid director response")

            logger.info(f"Director executed successfully in {execution_time:.2f}s")
            return function_call

        except Exception as e:
            logger.error(f"Director execution failed: {e}")
            # Return a fallback SwarmSpec
            return SwarmSpec(
                goals="Emergency fallback mode",
                plan="Execute available tasks with reduced functionality",
                rules="Follow basic operational guidelines",
                orders=[]
            )

    def _get_agent_status_summary(self) -> str:
        """Get summary of agent status"""
        status_counts = {}
        for health in self.agent_health.values():
            status_counts[health.state.value] = status_counts.get(health.state.value, 0) + 1
        
        return f"Agents: {dict(status_counts)}"

    def _get_performance_summary(self) -> str:
        """Get performance summary"""
        if not self.performance_metrics:
            return "No performance data available"
        
        return f"Success Rate: {self.performance_metrics.get('success_rate', 0):.2f}, " \
               f"Total Tasks: {self.performance_metrics.get('total_tasks', 0)}"

    def run_agent_with_retry(self, agent: Agent, task: str, order: HierarchicalOrder, img: str = None) -> TaskResult:
        """Run agent with retry mechanism and timeout"""
        agent_name = agent.agent_name
        if not agent_name:
            logger.error("Agent has no name, cannot execute task")
            return TaskResult(
                agent_name="unknown",
                task=task,
                output="",
                success=False,
                execution_time=0.0,
                timestamp=time.time(),
                error="Agent has no name"
            )
            
        start_time = time.time()
        
        for attempt in range(order.retry_count + 1):
            try:
                # Update agent health
                health = self.agent_health.get(agent_name)
                if health is None:
                    # Initialize health if not found
                    health = AgentHealth(
                        agent_name=agent_name,
                        state=AgentState.IDLE,
                        last_activity=time.time(),
                        task_count=0,
                        success_rate=1.0,
                        avg_response_time=0.0,
                        consecutive_failures=0
                    )
                    self.agent_health[agent_name] = health
                
                health.state = AgentState.RUNNING
                health.last_activity = time.time()

                # Prepare context
                agent_context = f"""
Loop: {self.current_loop}/{self.max_loops}
Attempt: {attempt + 1}/{order.retry_count + 1}
Priority: {order.priority.name}
History: {self.conversation.get_str()}
Your Task: {task}
"""

                # Run agent with timeout
                future = self._task_executor.submit(agent.run, agent_context)
                
                try:
                    timeout_value = order.timeout or self.task_timeout
                    output = future.result(timeout=timeout_value)
                    
                    # Task succeeded
                    execution_time = time.time() - start_time
                    health.state = AgentState.COMPLETED
                    health.task_count += 1
                    health.consecutive_failures = 0
                    
                    # Update success rate
                    total_tasks = health.task_count
                    if total_tasks > 0:
                        previous_successes = (total_tasks - 1) * health.success_rate
                        health.success_rate = (previous_successes + 1) / total_tasks
                    
                    # Update average response time
                    if health.avg_response_time == 0:
                        health.avg_response_time = execution_time
                    else:
                        health.avg_response_time = (health.avg_response_time + execution_time) / 2

                    # Add to conversation
                    self.conversation.add(
                        role=agent_name,
                        content=f"Loop {self.current_loop}/{self.max_loops}: {output}",
                    )

                    # Create successful result
                    result = TaskResult(
                        agent_name=agent_name,
                        task=task,
                        output=output,
                        success=True,
                        execution_time=execution_time,
                        timestamp=time.time()
                    )
                    
                    self.task_results.append(result)
                    logger.info(f"Agent {agent_name} completed task successfully in {execution_time:.2f}s")
                    
                    return result
                    
                except Exception as timeout_e:
                    logger.warning(f"Agent {agent_name} timed out on attempt {attempt + 1}: {timeout_e}")
                    if attempt < order.retry_count:
                        time.sleep(min(2 ** attempt, 10))  # Exponential backoff
                        continue
                    else:
                        raise

            except Exception as e:
                logger.error(f"Agent {agent_name} failed on attempt {attempt + 1}: {e}")
                
                # Update health on failure
                health = self.agent_health.get(agent_name)
                if health is not None:
                    health.consecutive_failures += 1
                    health.task_count += 1
                    
                    # Update success rate
                    total_tasks = health.task_count
                    if total_tasks > 0:
                        previous_successes = (total_tasks - 1) * health.success_rate
                        health.success_rate = previous_successes / total_tasks
                
                if attempt < order.retry_count:
                    time.sleep(min(2 ** attempt, 10))  # Exponential backoff
                    continue
                else:
                    # All attempts failed
                    if health is not None:
                        health.state = AgentState.FAILED
                    
                    execution_time = time.time() - start_time
                    
                    result = TaskResult(
                        agent_name=agent_name,
                        task=task,
                        output="",
                        success=False,
                        execution_time=execution_time,
                        timestamp=time.time(),
                        error=str(e)
                    )
                    
                    self.task_results.append(result)
                    logger.error(f"Agent {agent_name} failed after {order.retry_count + 1} attempts")
                    
                    return result

        # This should never be reached, but adding for completeness
        return TaskResult(
            agent_name=agent_name,
            task=task,
            output="",
            success=False,
            execution_time=time.time() - start_time,
            timestamp=time.time(),
            error="Unexpected execution path"
        )

    def execute_orders_concurrently(self, orders: List[HierarchicalOrder], img: str = None) -> Dict[str, TaskResult]:
        """Execute orders concurrently with dependency management"""
        results = {}
        completed_agents = set()
        
        # Sort orders by priority
        sorted_orders = sorted(orders, key=lambda x: x.priority.value, reverse=True)
        
        # Group orders by dependency level
        dependency_groups = self._group_orders_by_dependencies(sorted_orders)
        
        for group in dependency_groups:
            # Execute current group concurrently
            futures = {}
            
            for order in group:
                agent = self.find_agent(order.agent_name)
                if agent is None:
                    logger.error(f"Agent {order.agent_name} not found")
                    continue
                
                # Check if agent is healthy
                health = self.agent_health.get(order.agent_name)
                if health and health.state == AgentState.DISABLED:
                    logger.warning(f"Agent {order.agent_name} is disabled, skipping task")
                    continue
                
                # Submit task
                future = self._task_executor.submit(
                    self.run_agent_with_retry, agent, order.task, order, img
                )
                futures[future] = order
            
            # Wait for all tasks in current group to complete
            for future in as_completed(futures):
                order = futures[future]
                try:
                    result = future.result()
                    results[order.agent_name] = result
                    completed_agents.add(order.agent_name)
                    
                    formatter.print_panel(
                        result.output if result.success else f"FAILED: {result.error}",
                        title=f"Output from {order.agent_name} - Loop {self.current_loop}/{self.max_loops}",
                    )
                    
                except Exception as e:
                    logger.error(f"Unexpected error executing order for {order.agent_name}: {e}")
                    results[order.agent_name] = TaskResult(
                        agent_name=order.agent_name,
                        task=order.task,
                        output="",
                        success=False,
                        execution_time=0,
                        timestamp=time.time(),
                        error=str(e)
                    )
        
        return results

    def _group_orders_by_dependencies(self, orders: List[HierarchicalOrder]) -> List[List[HierarchicalOrder]]:
        """Group orders by their dependencies to enable proper execution order"""
        dependency_groups = []
        remaining_orders = orders.copy()
        completed_agents = set()
        
        while remaining_orders:
            current_group = []
            
            # Find orders that can be executed (no pending dependencies)
            for order in remaining_orders[:]:
                if not order.depends_on or all(dep in completed_agents for dep in order.depends_on):
                    current_group.append(order)
                    remaining_orders.remove(order)
            
            if not current_group:
                # Deadlock or circular dependency - execute remaining orders anyway
                logger.warning("Potential circular dependency detected, executing remaining orders")
                current_group = remaining_orders.copy()
                remaining_orders.clear()
            
            dependency_groups.append(current_group)
            completed_agents.update(order.agent_name for order in current_group)
        
        return dependency_groups

    def run(self, task: str, img: str = None, *args, **kwargs) -> Union[str, Dict, List]:
        """Enhanced run method with improved reliability and performance"""
        logger.info(f"Starting enhanced hierarchical swarm execution for task: {task}")
        
        # Add initial task to conversation
        self.conversation.add(role="User", content=f"Task: {task}")
        
        # Reset execution state
        self.current_loop = 0
        self.agent_outputs = {}
        self.task_results = []
        
        # Initialize loop context
        loop_context = "Initial planning phase"
        
        try:
            # Execute loops
            for loop_idx in range(self.max_loops):
                self.current_loop = loop_idx + 1
                logger.info(f"Starting loop {self.current_loop}/{self.max_loops}")
                
                # Check swarm health
                healthy_agents = self.get_healthy_agents()
                if len(healthy_agents) < len(self.agents) * (1 - self.failure_threshold):
                    logger.warning("Swarm is in degraded mode due to agent failures")
                
                # Get director's orders
                swarm_spec = self.run_director(task=task, loop_context=loop_context, img=img)
                
                # Add swarm spec to conversation
                self.add_goal_and_more_in_conversation(swarm_spec)
                
                # Parse orders
                orders = self.parse_swarm_spec(swarm_spec)
                if not orders:
                    logger.warning("No orders received from director")
                    continue
                
                # Execute orders concurrently
                loop_results = self.execute_orders_concurrently(orders, img=img)
                
                # Store results for this loop
                self.agent_outputs[self.current_loop] = {
                    name: result.output for name, result in loop_results.items()
                }
                
                # Prepare context for next loop
                loop_context = self.compile_loop_context(self.current_loop)
                
                # Check if we should continue
                if self.current_loop >= self.max_loops:
                    break
                
                # Brief pause between loops
                time.sleep(0.1)
            
            # Return formatted results
            logger.info(f"Hierarchical swarm execution completed after {self.current_loop} loops")
            return history_output_formatter(self.conversation, self.output_type)
            
        except Exception as e:
            logger.error(f"Hierarchical swarm execution failed: {e}")
            # Return error information
            return {
                "error": str(e),
                "completed_loops": self.current_loop,
                "partial_results": self.agent_outputs
            }

    def compile_loop_context(self, loop_number: int) -> str:
        """Enhanced loop context compilation with performance metrics"""
        if loop_number not in self.agent_outputs:
            return "No agent outputs available for this loop."
        
        context = f"Results from loop {loop_number}:\n"
        
        # Add agent outputs
        for agent_name, output in self.agent_outputs[loop_number].items():
            context += f"\n--- {agent_name}'s Output ---\n{output}\n"
        
        # Add performance metrics
        successful_tasks = sum(1 for result in self.task_results 
                             if result.success and result.timestamp > time.time() - 60)
        total_tasks = len([result for result in self.task_results 
                          if result.timestamp > time.time() - 60])
        
        if total_tasks > 0:
            success_rate = successful_tasks / total_tasks
            context += f"\n--- Performance Metrics ---\nSuccess Rate: {success_rate:.2f}\n"
        
        return context

    def add_name_and_description(self):
        """Add swarm name and description to conversation"""
        self.conversation.add(
            role="System",
            content=f"Enhanced Swarm Name: {self.name}\nSwarm Description: {self.description}",
        )

    def find_agent(self, name: str) -> Optional[Agent]:
        """Find agent by name with enhanced error handling"""
        try:
            matching_agents = [
                agent for agent in self.agents
                if agent.agent_name == name
            ]
            
            if not matching_agents:
                logger.error(f"Agent '{name}' not found in swarm '{self.name}'")
                return None
            
            return matching_agents[0]
        except Exception as e:
            logger.error(f"Error finding agent '{name}': {e}")
            return None

    def parse_swarm_spec(self, swarm_spec: SwarmSpec) -> List[HierarchicalOrder]:
        """Parse SwarmSpec with enhanced validation"""
        try:
            if not isinstance(swarm_spec, SwarmSpec):
                logger.error("Invalid SwarmSpec format")
                return []
            
            orders = swarm_spec.orders
            if not orders:
                logger.warning("No orders found in SwarmSpec")
                return []
            
            # Validate orders
            valid_orders = []
            for order in orders:
                if self.find_agent(order.agent_name) is None:
                    logger.warning(f"Skipping order for unknown agent: {order.agent_name}")
                    continue
                valid_orders.append(order)
            
            logger.info(f"Parsed {len(valid_orders)} valid orders from SwarmSpec")
            return valid_orders
            
        except Exception as e:
            logger.error(f"Error parsing SwarmSpec: {e}")
            return []

    def add_goal_and_more_in_conversation(self, swarm_spec: SwarmSpec) -> None:
        """Add swarm goals, plan, and rules to conversation"""
        try:
            self.conversation.add(
                role="Director",
                content=f"Goals:\n{swarm_spec.goals}\n\nPlan:\n{swarm_spec.plan}\n\nRules:\n{swarm_spec.rules}",
            )
        except Exception as e:
            logger.error(f"Error adding goals and plan to conversation: {e}")

    def batch_run(self, tasks: List[str], img: str = None) -> List[Union[str, Dict, List]]:
        """Enhanced batch run with concurrent execution"""
        logger.info(f"Starting batch run with {len(tasks)} tasks")
        
        if not tasks:
            return []
        
        # Execute tasks concurrently
        with ThreadPoolExecutor(max_workers=min(len(tasks), self.max_concurrent_tasks)) as executor:
            futures = [executor.submit(self.run, task, img) for task in tasks]
            results = []
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch task failed: {e}")
                    results.append({"error": str(e)})
        
        logger.info(f"Batch run completed with {len(results)} results")
        return results

    def get_swarm_metrics(self) -> Dict:
        """Get comprehensive swarm metrics"""
        metrics = {
            "agent_count": len(self.agents),
            "healthy_agents": len(self.get_healthy_agents()),
            "total_tasks": len(self.task_results),
            "successful_tasks": sum(1 for result in self.task_results if result.success),
            "failed_tasks": sum(1 for result in self.task_results if not result.success),
            "average_execution_time": sum(result.execution_time for result in self.task_results) / len(self.task_results) if self.task_results else 0,
            "success_rate": sum(1 for result in self.task_results if result.success) / len(self.task_results) if self.task_results else 0,
            "agent_health": {name: health.state.value for name, health in self.agent_health.items()},
            "performance_metrics": self.performance_metrics
        }
        return metrics

    def shutdown(self):
        """Graceful shutdown of the swarm"""
        logger.info("Shutting down hierarchical swarm...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for health monitor to stop
        if self._health_monitor_thread:
            self._health_monitor_thread.join(timeout=5)
        
        # Shutdown task executor
        self._task_executor.shutdown(wait=True)
        
        logger.info("Hierarchical swarm shutdown complete")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
