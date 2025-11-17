# Social Algorithms

The Social Algorithms framework provides a flexible system for defining custom social algorithms that control how agents communicate and interact with each other in multi-agent systems. This framework allows you to upload any arbitrary social algorithm as a callable that defines the sequence of communication between agents.

## Overview

Features

| Feature                                              | Description                                                        |
|------------------------------------------------------|--------------------------------------------------------------------|
| Custom Communication Patterns                        | Define custom communication patterns between agents                |
| Complex Multi-Agent Workflows                        | Implement complex multi-agent workflows                            |
| Emergent Behaviors                                   | Create emergent behaviors through agent interactions               |
| Communication Logging                                | Log and track all communication between agents                     |
| Timeout & Error Handling                             | Execute algorithms with timeout protection and error handling      |

## Core Classes

### SocialAlgorithms

The main class for creating and executing social algorithms.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm_id` | `str` | `None` | Unique identifier for the algorithm. If None, a UUID will be generated. |
| `name` | `str` | `"SocialAlgorithm"` | Human-readable name for the algorithm. |
| `description` | `str` | `"A custom social algorithm for agent communication"` | Description of what the algorithm does. |
| `agents` | `List[AgentType]` | `None` | List of agents that will participate in the algorithm. |
| `social_algorithm` | `Callable` | `None` | The callable that defines the communication sequence. Must accept (agents, task, **kwargs) as parameters. |
| `max_execution_time` | `float` | `300.0` | Maximum time allowed for algorithm execution in seconds. |
| `output_type` | `OutputType` | `"dict"` | Format of the output from the algorithm. |
| `verbose` | `bool` | `False` | Whether to enable verbose logging. |
| `enable_communication_logging` | `bool` | `True` | Whether to log communication steps. |
| `parallel_execution` | `bool` | `False` | Whether to enable parallel execution where possible. |
| `max_workers` | `int` | `None` | Maximum number of workers for parallel execution. |

#### Key Methods

##### `run(task: str, algorithm_args: Optional[Dict[str, Any]] = None, **kwargs) -> SocialAlgorithmResult`

Execute the social algorithm with the given task.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | `str` | The task to execute using the social algorithm. |
| `algorithm_args` | `Dict[str, Any]` | Additional arguments for the algorithm. |
| `**kwargs` | `Any` | Additional keyword arguments. |

**Returns:** `SocialAlgorithmResult` - The result of executing the social algorithm.

**Raises:**

- `InvalidAlgorithmError`: If no social algorithm is defined.
- `TimeoutError`: If the algorithm execution exceeds max_execution_time.
- `Exception`: If the algorithm execution fails.

##### `run_async(task: str, algorithm_args: Optional[Dict[str, Any]] = None, **kwargs) -> SocialAlgorithmResult`

Execute the social algorithm asynchronously.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | `str` | The task to execute using the social algorithm. |
| `algorithm_args` | `Dict[str, Any]` | Additional arguments for the algorithm. |
| `**kwargs` | `Any` | Additional keyword arguments. |

**Returns:** `SocialAlgorithmResult` - The result of executing the social algorithm.

##### `add_agent(agent: Agent) -> None`

Add an agent to the social algorithm.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent` | `Agent` | The agent to add. |

**Raises:**

- `ValueError`: If agent is not an instance of the Agent class.

##### `remove_agent(agent_name: str) -> None`

Remove an agent from the social algorithm.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_name` | `str` | Name of the agent to remove. |

**Raises:**

- `AgentNotFoundError`: If no agent with the given name is found.

##### `get_agent_names() -> List[str]`

Get a list of all agent names in the algorithm.

**Returns:** `List[str]` - List of agent names.

##### `get_communication_history() -> List[CommunicationStep]`

Get the communication history for this algorithm execution.

**Returns:** `List[CommunicationStep]` - List of communication steps.

##### `clear_communication_history() -> None`

Clear the communication history.

##### `get_algorithm_info() -> Dict[str, Any]`

Get information about the social algorithm.

**Returns:** `Dict[str, Any]` - Information about the algorithm including ID, name, description, agent count, and configuration.

### CommunicationStep

Represents a single step in a social algorithm.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `step_id` | `str` | Unique identifier for the communication step. |
| `sender_agent` | `str` | Name of the sending agent. |
| `receiver_agent` | `str` | Name of the receiving agent. |
| `message` | `str` | The message being sent. |
| `timestamp` | `float` | Timestamp when the communication occurred. |
| `metadata` | `Optional[Dict[str, Any]]` | Additional metadata about the communication. |

### SocialAlgorithmResult

Result of executing a social algorithm.

#### Result Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `algorithm_id` | `str` | Unique identifier for the algorithm. |
| `execution_time` | `float` | Time taken to execute the algorithm in seconds. |
| `total_steps` | `int` | Total number of communication steps. |
| `successful_steps` | `int` | Number of successful communication steps. |
| `failed_steps` | `int` | Number of failed communication steps. |
| `communication_history` | `List[CommunicationStep]` | Complete history of all communications. |
| `final_outputs` | `Dict[str, Any]` | The final outputs from the algorithm. |
| `metadata` | `Optional[Dict[str, Any]]` | Additional metadata about the execution. |

## Social Algorithm Types

### SocialAlgorithmType (Enum)

Predefined types of social algorithms.

| Type | Description |
|------|-------------|
| `CUSTOM` | Custom user-defined algorithm |
| `SEQUENTIAL` | Sequential execution pattern |
| `CONCURRENT` | Concurrent execution pattern |
| `HIERARCHICAL` | Hierarchical execution pattern |
| `MESH` | Mesh network pattern |
| `ROUND_ROBIN` | Round-robin execution pattern |
| `BROADCAST` | Broadcast communication pattern |

## Exception Classes

### SocialAlgorithmError

Base exception for social algorithm errors.

### InvalidAlgorithmError

Raised when an invalid algorithm is provided.

### AgentNotFoundError

Raised when a required agent is not found.

## Usage Examples

### Example 1: Research and Development Team Algorithm

This example demonstrates a complete R&D team workflow with multiple specialized agents working together on a research project.

```python
from swarms import Agent, SocialAlgorithms
import time

def research_development_algorithm(agents, task, **kwargs):
    """
    A comprehensive R&D team algorithm with multiple phases and specialized roles.
    """
    # Define agent roles
    project_manager = next(agent for agent in agents if "ProjectManager" in agent.agent_name)
    researcher = next(agent for agent in agents if "Researcher" in agent.agent_name)
    analyst = next(agent for agent in agents if "Analyst" in agent.agent_name)
    developer = next(agent for agent in agents if "Developer" in agent.agent_name)
    tester = next(agent for agent in agents if "Tester" in agent.agent_name)
    reviewer = next(agent for agent in agents if "Reviewer" in agent.agent_name)
    
    # Initialize project state
    project_state = {
        "phase": "initialization",
        "requirements": {},
        "research_findings": {},
        "analysis_results": {},
        "prototype": {},
        "test_results": {},
        "final_review": {},
        "deliverables": {}
    }
    
    # Phase 1: Project Planning
    project_manager.run(f"Initialize project: {task}")
    
    planning_prompt = f"""
    As Project Manager, create a comprehensive project plan for: {task}
    
    Include:
    1. Project objectives and success criteria
    2. Key deliverables and milestones
    3. Resource requirements
    4. Timeline and deadlines
    5. Risk assessment and mitigation strategies
    6. Quality standards and review processes
    """
    
    project_plan = project_manager.run(planning_prompt)
    project_state["requirements"] = project_plan
    
    # Phase 2: Research Phase
    researcher.run("Begin research phase")
    
    research_prompt = f"""
    As Researcher, conduct comprehensive research on: {task}
    
    Research areas:
    1. Current state of the art
    2. Existing solutions and their limitations
    3. Emerging technologies and trends
    4. Market analysis and user needs
    5. Technical feasibility studies
    6. Competitive landscape analysis
    
    Provide detailed findings with sources and evidence.
    """
    
    research_findings = researcher.run(research_prompt)
    project_state["research_findings"] = research_findings
    
    # Phase 3: Analysis and Design
    analyst.run("Begin analysis phase")
    
    analysis_prompt = f"""
    As Analyst, analyze the research findings and design the solution:
    
    Research Findings: {research_findings}
    
    Analysis tasks:
    1. Requirements analysis and prioritization
    2. Technical architecture design
    3. User experience design considerations
    4. Implementation strategy and approach
    5. Resource and timeline estimation
    6. Risk analysis and mitigation plans
    
    Provide detailed analysis and design recommendations.
    """
    
    analysis_results = analyst.run(analysis_prompt)
    project_state["analysis_results"] = analysis_results
    
    # Phase 4: Development
    developer.run("Begin development phase")
    
    development_prompt = f"""
    As Developer, create a prototype based on the analysis:
    
    Analysis Results: {analysis_results}
    Project Requirements: {project_plan}
    
    Development tasks:
    1. Create technical specifications
    2. Implement core functionality
    3. Build user interface components
    4. Integrate with external systems
    5. Implement security measures
    6. Create documentation and code comments
    
    Provide detailed implementation and code samples.
    """
    
    prototype = developer.run(development_prompt)
    project_state["prototype"] = prototype
    
    # Phase 5: Testing
    tester.run("Begin testing phase")
    
    testing_prompt = f"""
    As Tester, create comprehensive test plans and execute testing:
    
    Prototype: {prototype}
    Requirements: {project_plan}
    
    Testing activities:
    1. Unit testing and code review
    2. Integration testing
    3. User acceptance testing
    4. Performance testing
    5. Security testing
    6. Bug tracking and resolution
    
    Provide detailed test results and recommendations.
    """
    
    test_results = tester.run(testing_prompt)
    project_state["test_results"] = test_results
    
    # Phase 6: Final Review
    reviewer.run("Begin final review")
    
    review_prompt = f"""
    As Reviewer, conduct final review of the entire project:
    
    Project Plan: {project_plan}
    Research: {research_findings}
    Analysis: {analysis_results}
    Prototype: {prototype}
    Testing: {test_results}
    
    Review criteria:
    1. Requirements fulfillment
    2. Quality and completeness
    3. Technical excellence
    4. User experience
    5. Documentation quality
    6. Deliverable readiness
    
    Provide final assessment and recommendations.
    """
    
    final_review = reviewer.run(review_prompt)
    project_state["final_review"] = final_review
    
    # Phase 7: Project Closure
    closure_prompt = f"""
    As Project Manager, create final project deliverables:
    
    Complete Project State: {project_state}
    
    Create:
    1. Executive summary
    2. Technical documentation
    3. User guide
    4. Deployment instructions
    5. Maintenance procedures
    6. Lessons learned and recommendations
    """
    
    deliverables = project_manager.run(closure_prompt)
    project_state["deliverables"] = deliverables
    
    return {
        "task": task,
        "project_state": project_state,
        "algorithm_type": "research_development",
        "phases_completed": 7,
        "total_agents": len(agents)
    }

# Create specialized agents
project_manager = Agent(
    agent_name="ProjectManager",
    system_prompt="You are an experienced project manager focused on planning, coordination, and delivery excellence.",
    model_name="gpt-4o-mini",
    max_loops=1
)

researcher = Agent(
    agent_name="Researcher",
    system_prompt="You are a research specialist focused on gathering comprehensive information and insights.",
    model_name="gpt-4o-mini",
    max_loops=1
)

analyst = Agent(
    agent_name="Analyst",
    system_prompt="You are a systems analyst focused on requirements analysis and solution design.",
    model_name="gpt-4o-mini",
    max_loops=1
)

developer = Agent(
    agent_name="Developer",
    system_prompt="You are a senior developer focused on creating high-quality, maintainable code.",
    model_name="gpt-4o-mini",
    max_loops=1
)

tester = Agent(
    agent_name="Tester",
    system_prompt="You are a QA specialist focused on ensuring quality and reliability.",
    model_name="gpt-4o-mini",
    max_loops=1
)

reviewer = Agent(
    agent_name="Reviewer",
    system_prompt="You are a technical reviewer focused on quality assurance and best practices.",
    model_name="gpt-4o-mini",
    max_loops=1
)

# Create and run the R&D algorithm
rd_algorithm = SocialAlgorithms(
    name="Research-Development-Team",
    description="Complete R&D workflow with specialized team members",
    agents=[project_manager, researcher, analyst, developer, tester, reviewer],
    social_algorithm=research_development_algorithm,
    verbose=True,
    max_execution_time=600  # 10 minutes
)

# Execute the algorithm
result = rd_algorithm.run("Develop a sustainable energy management system for smart cities")
```

### Example 2: Creative Collaboration Algorithm

This example shows a creative team working together on a marketing campaign with iterative feedback and refinement.

```python
from swarms import Agent, SocialAlgorithms
import random

def creative_collaboration_algorithm(agents, task, **kwargs):
    """
    A creative collaboration algorithm with iterative feedback and refinement.
    """
    # Define creative team roles
    creative_director = next(agent for agent in agents if "CreativeDirector" in agent.agent_name)
    copywriter = next(agent for agent in agents if "Copywriter" in agent.agent_name)
    designer = next(agent for agent in agents if "Designer" in agent.agent_name)
    strategist = next(agent for agent in agents if "Strategist" in agent.agent_name)
    client = next(agent for agent in agents if "Client" in agent.agent_name)
    
    # Algorithm parameters
    max_iterations = kwargs.get("max_iterations", 5)
    feedback_threshold = kwargs.get("feedback_threshold", 0.8)
    
    # Initialize creative state
    creative_state = {
        "iteration": 0,
        "concepts": [],
        "feedback_history": [],
        "refinements": [],
        "final_concept": None,
        "client_satisfaction": 0.0
    }
    
    # Phase 1: Initial Briefing
    creative_director.run(f"Begin creative project: {task}")
    
    briefing_prompt = f"""
    As Creative Director, establish the creative brief for: {task}
    
    Include:
    1. Project objectives and goals
    2. Target audience definition
    3. Key messages and value propositions
    4. Brand guidelines and constraints
    5. Success metrics and KPIs
    6. Timeline and deliverables
    """
    
    creative_brief = creative_director.run(briefing_prompt)
    
    # Phase 2: Strategy Development
    strategist.run("Develop marketing strategy")
    
    strategy_prompt = f"""
    As Strategist, develop comprehensive marketing strategy:
    
    Creative Brief: {creative_brief}
    Task: {task}
    
    Strategy elements:
    1. Market positioning and differentiation
    2. Channel strategy and media mix
    3. Messaging framework and tone
    4. Campaign structure and phases
    5. Budget allocation and ROI projections
    6. Measurement and optimization plan
    """
    
    marketing_strategy = strategist.run(strategy_prompt)
    
    # Phase 3: Iterative Creative Development
    for iteration in range(1, max_iterations + 1):
        creative_state["iteration"] = iteration
        
        # Generate creative concepts
        concept_prompt = f"""
        As Creative Team, develop creative concepts for iteration {iteration}:
        
        Creative Brief: {creative_brief}
        Marketing Strategy: {marketing_strategy}
        Previous Feedback: {creative_state['feedback_history'][-1] if creative_state['feedback_history'] else 'None'}
        
        Create:
        1. Multiple creative concepts and approaches
        2. Visual design directions and mockups
        3. Copy variations and messaging options
        4. Channel-specific adaptations
        5. Implementation considerations
        6. Creative rationale and insights
        """
        
        # Get input from creative team
        copywriter_concepts = copywriter.run(concept_prompt)
        designer_concepts = designer.run(concept_prompt)
        
        # Combine concepts
        combined_concepts = f"""
        Copywriter Concepts: {copywriter_concepts}
        Designer Concepts: {designer_concepts}
        """
        
        creative_state["concepts"].append({
            "iteration": iteration,
            "concepts": combined_concepts,
            "timestamp": time.time()
        })
        
        # Client feedback
        feedback_prompt = f"""
        As Client, provide feedback on creative concepts for iteration {iteration}:
        
        Concepts: {combined_concepts}
        Original Brief: {creative_brief}
        
        Evaluate:
        1. Alignment with brand and objectives
        2. Appeal to target audience
        3. Clarity and effectiveness of messaging
        4. Visual impact and design quality
        5. Feasibility and implementation
        6. Overall satisfaction (rate 0-1)
        
        Provide specific feedback and suggestions for improvement.
        """
        
        client_feedback = client.run(feedback_prompt)
        creative_state["feedback_history"].append({
            "iteration": iteration,
            "feedback": client_feedback,
            "timestamp": time.time()
        })
        
        # Calculate satisfaction score (simplified)
        satisfaction_score = random.uniform(0.6, 0.9)  # Simulated client satisfaction
        creative_state["client_satisfaction"] = satisfaction_score
        
        # Check if we've reached satisfaction threshold
        if satisfaction_score >= feedback_threshold:
            creative_director.run(f"Client satisfaction reached: {satisfaction_score:.2f}")
            break
        
        # Refinement phase
        refinement_prompt = f"""
        As Creative Director, refine concepts based on feedback:
        
        Current Concepts: {combined_concepts}
        Client Feedback: {client_feedback}
        Satisfaction Score: {satisfaction_score:.2f}
        
        Refine:
        1. Address specific feedback points
        2. Improve weak areas identified
        3. Enhance strong elements
        4. Explore new creative directions
        5. Optimize for better client appeal
        6. Prepare for next iteration
        """
        
        refinements = creative_director.run(refinement_prompt)
        creative_state["refinements"].append({
            "iteration": iteration,
            "refinements": refinements,
            "timestamp": time.time()
        })
    
    # Phase 4: Final Concept Selection
    final_prompt = f"""
    As Creative Director, select and finalize the best concept:
    
    All Concepts: {creative_state['concepts']}
    All Feedback: {creative_state['feedback_history']}
    All Refinements: {creative_state['refinements']}
    Final Satisfaction: {creative_state['client_satisfaction']:.2f}
    
    Create final deliverable:
    1. Selected concept with rationale
    2. Final creative assets and specifications
    3. Implementation guidelines
    4. Brand compliance checklist
    5. Quality assurance standards
    6. Launch recommendations
    """
    
    final_concept = creative_director.run(final_prompt)
    creative_state["final_concept"] = final_concept
    
    return {
        "task": task,
        "creative_state": creative_state,
        "algorithm_type": "creative_collaboration",
        "iterations_completed": creative_state["iteration"],
        "final_satisfaction": creative_state["client_satisfaction"]
    }

# Create creative team agents
creative_director = Agent(
    agent_name="CreativeDirector",
    system_prompt="You are a creative director focused on leading creative vision and ensuring brand consistency.",
    model_name="gpt-4o-mini",
    max_loops=1
)

copywriter = Agent(
    agent_name="Copywriter",
    system_prompt="You are a creative copywriter focused on compelling messaging and storytelling.",
    model_name="gpt-4o-mini",
    max_loops=1
)

designer = Agent(
    agent_name="Designer",
    system_prompt="You are a visual designer focused on creating impactful and beautiful designs.",
    model_name="gpt-4o-mini",
    max_loops=1
)

strategist = Agent(
    agent_name="Strategist",
    system_prompt="You are a marketing strategist focused on data-driven insights and campaign optimization.",
    model_name="gpt-4o-mini",
    max_loops=1
)

client = Agent(
    agent_name="Client",
    system_prompt="You are a client representative focused on brand requirements and business objectives.",
    model_name="gpt-4o-mini",
    max_loops=1
)

# Create and run creative collaboration algorithm
creative_algorithm = SocialAlgorithms(
    name="Creative-Collaboration-Team",
    description="Creative team collaboration with iterative feedback",
    agents=[creative_director, copywriter, designer, strategist, client],
    social_algorithm=creative_collaboration_algorithm,
    verbose=True,
    max_execution_time=800  # 13+ minutes
)

# Execute the algorithm
result = creative_algorithm.run(
    "Create a comprehensive marketing campaign for a new eco-friendly smartphone",
    algorithm_args={"max_iterations": 5, "feedback_threshold": 0.8}
)
```

### Example 3: Emergency Response Algorithm

This example demonstrates a crisis management system with real-time coordination and decision-making.

```python
from swarms import Agent, SocialAlgorithms
import time
import random

def emergency_response_algorithm(agents, task, **kwargs):
    """
    An emergency response algorithm with real-time coordination and decision-making.
    """
    # Define emergency response roles
    incident_commander = next(agent for agent in agents if "IncidentCommander" in agent.agent_name)
    operations_chief = next(agent for agent in agents if "OperationsChief" in agent.agent_name)
    safety_officer = next(agent for agent in agents if "SafetyOfficer" in agent.agent_name)
    communications_officer = next(agent for agent in agents if "CommunicationsOfficer" in agent.agent_name)
    logistics_coordinator = next(agent for agent in agents if "LogisticsCoordinator" in agent.agent_name)
    medical_officer = next(agent for agent in agents if "MedicalOfficer" in agent.agent_name)
    
    # Emergency parameters
    severity_level = kwargs.get("severity_level", "high")
    max_response_time = kwargs.get("max_response_time", 300)  # 5 minutes
    update_frequency = kwargs.get("update_frequency", 30)  # 30 seconds
    
    # Initialize emergency state
    emergency_state = {
        "incident_id": f"INC-{int(time.time())}",
        "severity": severity_level,
        "status": "active",
        "start_time": time.time(),
        "response_teams": [],
        "resources_deployed": [],
        "casualties": {"injured": 0, "fatalities": 0},
        "evacuations": {"completed": 0, "in_progress": 0},
        "communications": [],
        "safety_assessments": [],
        "medical_status": {},
        "logistics_status": {},
        "resolution_status": "ongoing"
    }
    
    # Phase 1: Initial Assessment and Activation
    incident_commander.run(f"Emergency incident reported: {task}")
    
    assessment_prompt = f"""
    As Incident Commander, conduct initial emergency assessment:
    
    Incident: {task}
    Severity Level: {severity_level}
    Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
    
    Assessment tasks:
    1. Evaluate incident severity and scope
    2. Identify immediate threats and hazards
    3. Assess resource requirements
    4. Determine response strategy
    5. Establish command structure
    6. Initiate emergency protocols
    """
    
    initial_assessment = incident_commander.run(assessment_prompt)
    emergency_state["initial_assessment"] = initial_assessment
    
    # Phase 2: Safety Assessment
    safety_officer.run("Conducting safety assessment")
    
    safety_prompt = f"""
    As Safety Officer, assess safety conditions and risks:
    
    Incident: {task}
    Initial Assessment: {initial_assessment}
    
    Safety evaluation:
    1. Environmental hazards and contamination
    2. Structural integrity and stability
    3. Fire and explosion risks
    4. Chemical and biological threats
    5. Evacuation requirements and routes
    6. Personal protective equipment needs
    """
    
    safety_assessment = safety_officer.run(safety_prompt)
    emergency_state["safety_assessments"].append({
        "timestamp": time.time(),
        "assessment": safety_assessment,
        "officer": "SafetyOfficer"
    })
    
    # Phase 3: Medical Response
    medical_officer.run("Activating medical response")
    
    medical_prompt = f"""
    As Medical Officer, assess medical needs and response:
    
    Incident: {task}
    Safety Assessment: {safety_assessment}
    
    Medical evaluation:
    1. Casualty assessment and triage
    2. Medical resource requirements
    3. Hospital capacity and availability
    4. Specialized medical needs
    5. Mass casualty protocols
    6. Medical evacuation procedures
    """
    
    medical_assessment = medical_officer.run(medical_prompt)
    emergency_state["medical_status"] = medical_assessment
    
    # Phase 4: Operations Coordination
    operations_chief.run("Coordinating operational response")
    
    operations_prompt = f"""
    As Operations Chief, coordinate emergency response operations:
    
    Incident: {task}
    Safety: {safety_assessment}
    Medical: {medical_assessment}
    
    Operations coordination:
    1. Deploy response teams and resources
    2. Establish incident command post
    3. Coordinate with external agencies
    4. Implement containment measures
    5. Execute rescue operations
    6. Monitor progress and adjust tactics
    """
    
    operations_plan = operations_chief.run(operations_prompt)
    emergency_state["operations_plan"] = operations_plan
    
    # Phase 5: Communications Management
    communications_officer.run("Managing emergency communications")
    
    comms_prompt = f"""
    As Communications Officer, manage all emergency communications:
    
    Incident: {task}
    Operations: {operations_plan}
    
    Communications tasks:
    1. Notify relevant authorities and agencies
    2. Coordinate with media and public information
    3. Establish communication protocols
    4. Manage internal team communications
    5. Handle public inquiries and updates
    6. Maintain communication logs
    """
    
    communications_plan = communications_officer.run(comms_prompt)
    emergency_state["communications"].append({
        "timestamp": time.time(),
        "plan": communications_plan,
        "officer": "CommunicationsOfficer"
    })
    
    # Phase 6: Logistics Support
    logistics_coordinator.run("Coordinating logistics support")
    
    logistics_prompt = f"""
    As Logistics Coordinator, manage resources and support:
    
    Incident: {task}
    Operations: {operations_plan}
    Communications: {communications_plan}
    
    Logistics coordination:
    1. Resource allocation and deployment
    2. Equipment and supply management
    3. Transportation and logistics
    4. Facility and infrastructure support
    5. Personnel coordination and scheduling
    6. Cost tracking and accountability
    """
    
    logistics_plan = logistics_coordinator.run(logistics_prompt)
    emergency_state["logistics_status"] = logistics_plan
    
    # Phase 7: Continuous Monitoring and Updates
    update_cycles = max_response_time // update_frequency
    
    for cycle in range(1, update_cycles + 1):
        # Status update from each officer
        status_prompt = f"""
        Provide status update for cycle {cycle}:
        
        Current Emergency State: {emergency_state}
        Time Elapsed: {time.time() - emergency_state['start_time']:.1f} seconds
        
        Report:
        1. Current situation status
        2. Progress on assigned tasks
        3. Resource utilization
        4. Challenges and obstacles
        5. Next steps and priorities
        6. Support requirements
        """
        
        # Get updates from all officers
        commander_update = incident_commander.run(status_prompt)
        operations_update = operations_chief.run(status_prompt)
        safety_update = safety_officer.run(status_prompt)
        comms_update = communications_officer.run(status_prompt)
        logistics_update = logistics_coordinator.run(status_prompt)
        medical_update = medical_officer.run(status_prompt)
        
        # Update emergency state
        emergency_state["status_updates"].append({
            "cycle": cycle,
            "timestamp": time.time(),
            "updates": {
                "commander": commander_update,
                "operations": operations_update,
                "safety": safety_update,
                "communications": comms_update,
                "logistics": logistics_update,
                "medical": medical_update
            }
        })
        
        # Simulate incident resolution over time
        resolution_progress = min(cycle / update_cycles, 1.0)
        if resolution_progress >= 0.8:
            emergency_state["resolution_status"] = "resolved"
            break
    
    # Phase 8: Incident Resolution and Debrief
    resolution_prompt = f"""
    As Incident Commander, provide final incident resolution report:
    
    Emergency State: {emergency_state}
    Resolution Status: {emergency_state['resolution_status']}
    
    Final report:
    1. Incident summary and timeline
    2. Response effectiveness assessment
    3. Casualties and damage assessment
    4. Resource utilization summary
    5. Lessons learned and recommendations
    6. Follow-up actions required
    """
    
    final_report = incident_commander.run(resolution_prompt)
    emergency_state["final_report"] = final_report
    
    return {
        "task": task,
        "emergency_state": emergency_state,
        "algorithm_type": "emergency_response",
        "response_time": time.time() - emergency_state["start_time"],
        "resolution_status": emergency_state["resolution_status"]
    }

# Create emergency response team
incident_commander = Agent(
    agent_name="IncidentCommander",
    system_prompt="You are an experienced incident commander focused on leadership and decision-making in emergency situations.",
    model_name="gpt-4o-mini",
    max_loops=1
)

operations_chief = Agent(
    agent_name="OperationsChief",
    system_prompt="You are an operations chief focused on tactical execution and resource coordination.",
    model_name="gpt-4o-mini",
    max_loops=1
)

safety_officer = Agent(
    agent_name="SafetyOfficer",
    system_prompt="You are a safety officer focused on hazard assessment and safety protocols.",
    model_name="gpt-4o-mini",
    max_loops=1
)

communications_officer = Agent(
    agent_name="CommunicationsOfficer",
    system_prompt="You are a communications officer focused on information management and public relations.",
    model_name="gpt-4o-mini",
    max_loops=1
)

logistics_coordinator = Agent(
    agent_name="LogisticsCoordinator",
    system_prompt="You are a logistics coordinator focused on resource management and support operations.",
    model_name="gpt-4o-mini",
    max_loops=1
)

medical_officer = Agent(
    agent_name="MedicalOfficer",
    system_prompt="You are a medical officer focused on casualty care and medical response coordination.",
    model_name="gpt-4o-mini",
    max_loops=1
)

# Create and run emergency response algorithm
emergency_algorithm = SocialAlgorithms(
    name="Emergency-Response-Team",
    description="Emergency response coordination with real-time updates",
    agents=[incident_commander, operations_chief, safety_officer, 
            communications_officer, logistics_coordinator, medical_officer],
    social_algorithm=emergency_response_algorithm,
    verbose=True,
    max_execution_time=600  # 10 minutes
)

# Execute the algorithm
result = emergency_algorithm.run(
    "Chemical spill at industrial facility with potential environmental impact",
    algorithm_args={
        "severity_level": "high",
        "max_response_time": 300,
        "update_frequency": 30
    }
)
```

### Basic Social Algorithm

```python
from swarms import Agent, SocialAlgorithms

# Define a custom social algorithm
def custom_communication_algorithm(agents, task, **kwargs):
    # Agent 1 researches the topic
    research_result = agents[0].run(f"Research: {task}")
    
    # Agent 2 analyzes the research
    analysis = agents[1].run(f"Analyze this research: {research_result}")
    
    # Agent 3 synthesizes the findings
    synthesis = agents[2].run(f"Synthesize: {research_result} + {analysis}")
    
    return {
        "research": research_result,
        "analysis": analysis,
        "synthesis": synthesis
    }

# Create agents
researcher = Agent(agent_name="Researcher", model_name="gpt-4o-mini")
analyst = Agent(agent_name="Analyst", model_name="gpt-4o-mini")
synthesizer = Agent(agent_name="Synthesizer", model_name="gpt-4o-mini")

# Create social algorithm
social_alg = SocialAlgorithms(
    name="Research-Analysis-Synthesis",
    agents=[researcher, analyst, synthesizer],
    social_algorithm=custom_communication_algorithm,
    verbose=True
)

# Run the algorithm
result = social_alg.run("The impact of AI on healthcare")
```

### Competitive Evaluation Algorithm

```python
def competitive_evaluation_algorithm(agents, task, **kwargs):
    """A competitive evaluation algorithm where agents compete and are evaluated."""
    if len(agents) < 3:
        raise ValueError("This algorithm requires at least 3 agents (2 competitors + 1 judge)")
    
    competitors = agents[:-1]
    judge = agents[-1]
    
    # Each competitor works on the task
    competitor_results = {}
    for i, competitor in enumerate(competitors):
        competitor_prompt = f"Solve this task as best as you can: {task}"
        result = competitor.run(competitor_prompt)
        competitor_results[f"competitor_{i+1}_{competitor.agent_name}"] = result
    
    # Judge evaluates all solutions
    evaluation_prompt = f"Evaluate these solutions and rank them:\n\n"
    for name, result in competitor_results.items():
        evaluation_prompt += f"{name}:\n{result}\n\n"
    
    evaluation_prompt += "Provide rankings, scores, and detailed feedback for each solution."
    evaluation = judge.run(evaluation_prompt)
    
    return {
        "competitor_solutions": competitor_results,
        "judge_evaluation": evaluation,
        "task": task,
    }

# Create and run competitive evaluation
social_alg = SocialAlgorithms(
    name="Competitive-Evaluation",
    description="Competitive evaluation where agents compete and are judged",
    agents=[competitor1, competitor2, judge],
    social_algorithm=competitive_evaluation_algorithm,
    verbose=True,
)

result = social_alg.run("Design the most efficient algorithm for sorting large datasets")
```

### Negotiation Algorithm

```python
def negotiation_algorithm(agents, task, **kwargs):
    """A negotiation algorithm where agents engage in back-and-forth communication."""
    negotiating_parties = agents[:-2]  # First 3 agents are negotiating parties
    mediator_agent = agents[-2]        # Second to last is mediator
    legal_agent = agents[-1]           # Last is legal advisor
    
    max_rounds = kwargs.get("max_rounds", 8)
    agreement_threshold = kwargs.get("agreement_threshold", 0.8)
    
    # Initialize negotiation state
    negotiation_history = []
    current_positions = {}
    
    # Phase 1: Initial Position Statements
    for party in negotiating_parties:
        position_prompt = f"""
        As {party.agent_name}, state your initial position for: {task}
        Include your objectives, requirements, and constraints.
        """
        initial_position = party.run(position_prompt)
        current_positions[party.agent_name] = initial_position
    
    # Phase 2: Negotiation Rounds
    for round_num in range(1, max_rounds + 1):
        # Mediator analyzes and provides guidance
        mediation_guidance = mediator_agent.run(f"Analyze positions for round {round_num}")
        
        # Each party responds and makes counter-proposals
        round_responses = {}
        for party in negotiating_parties:
            response = party.run(f"Respond to other positions in round {round_num}")
            round_responses[party.agent_name] = response
            current_positions[party.agent_name] = response
        
        # Legal review
        legal_review = legal_agent.run(f"Review proposals for round {round_num}")
        
        # Record round results
        negotiation_history.append({
            "round": round_num,
            "mediation_guidance": mediation_guidance,
            "responses": round_responses,
            "legal_review": legal_review,
        })
    
    return {
        "task": task,
        "negotiation_history": negotiation_history,
        "current_positions": current_positions,
        "algorithm_type": "negotiation",
    }
```

### Swarm Intelligence Algorithm

```python
def swarm_intelligence_algorithm(agents, task, **kwargs):
    """A swarm intelligence algorithm with emergent behavior through local interactions."""
    explorers = [agent for agent in agents if "Explorer" in agent.agent_name]
    exploiters = [agent for agent in agents if "Exploiter" in agent.agent_name]
    coordinator_agent = next(agent for agent in agents if "Coordinator" in agent.agent_name)
    
    max_iterations = kwargs.get("max_iterations", 8)
    exploration_ratio = kwargs.get("exploration_ratio", 0.6)
    
    # Initialize swarm state
    swarm_knowledge = []
    pheromone_trails = {}
    agent_positions = {}
    
    # Phase 1: Initial Exploration
    for agent in explorers + exploiters:
        exploration_focus = random.choice([
            "technical approach", "user experience", "business model",
            "implementation strategy", "risk mitigation", "innovation"
        ])
        
        discovery = agent.run(f"Explore {exploration_focus} related to: {task}")
        swarm_knowledge.append({
            "agent": agent.agent_name,
            "discovery": discovery,
            "focus": exploration_focus,
            "attractiveness": random.uniform(0.1, 1.0),
        })
        agent_positions[agent.agent_name] = exploration_focus
    
    # Phase 2: Swarm Dynamics
    for iteration in range(1, max_iterations + 1):
        # Calculate pheromone trails
        for discovery in swarm_knowledge:
            solution_key = discovery["focus"]
            pheromone_trails[solution_key] = pheromone_trails.get(solution_key, 0) + discovery["attractiveness"]
        
        # Agents decide whether to explore or exploit
        for agent in explorers + exploiters:
            current_position = agent_positions[agent.agent_name]
            local_pheromone = pheromone_trails.get(current_position, 0.1)
            
            if random.random() < exploration_ratio or local_pheromone < 0.5:
                # Exploration behavior
                action_result = agent.run(f"Continue exploring in: {current_position}")
            else:
                # Exploitation behavior
                best_solution = max(pheromone_trails.items(), key=lambda x: x[1])
                action_result = agent.run(f"Exploit promising area: {best_solution[0]}")
            
            swarm_knowledge.append({
                "agent": agent.agent_name,
                "result": action_result,
                "iteration": iteration,
            })
    
    # Phase 3: Emergent Solution Synthesis
    top_solutions = sorted(pheromone_trails.items(), key=lambda x: x[1], reverse=True)[:3]
    
    final_synthesis = coordinator_agent.run(f"""
    Synthesize the emergent intelligence from the swarm:
    Task: {task}
    Top solutions: {top_solutions}
    Total discoveries: {len(swarm_knowledge)}
    """)
    
    return {
        "task": task,
        "swarm_knowledge": swarm_knowledge,
        "pheromone_trails": pheromone_trails,
        "final_synthesis": final_synthesis,
        "algorithm_type": "swarm_intelligence",
    }
```

## Advanced Features

### Communication Logging

The framework automatically logs all communication between agents when `enable_communication_logging=True`. This includes:

- All `agent.run()` calls
- All `agent.talk_to()` calls
- Timestamps and metadata for each communication

### Timeout Protection

Algorithms are executed with timeout protection to prevent infinite loops or hanging processes. The default timeout is 300 seconds (5 minutes), but this can be customized.

### Error Handling

The framework provides comprehensive error handling:

- `InvalidAlgorithmError` for invalid algorithm definitions
- `AgentNotFoundError` for missing agents
- `TimeoutError` for execution timeouts
- Graceful handling of agent execution failures

### Output Formatting

Results can be formatted according to the specified `output_type`:

- `"dict"`: Dictionary format (default)

- `"list"`: List format

- `"str"`: String format

### Parallel Execution

When `parallel_execution=True`, the framework can execute independent operations in parallel for improved performance.

## Best Practices

1. **Algorithm Design**: Design your social algorithms to be modular and reusable. Consider breaking complex algorithms into smaller, composable functions.

2. **Error Handling**: Always include proper error handling in your custom algorithms. Check for required agents and validate inputs.

3. **Logging**: Use the built-in logging system to track algorithm execution and debug issues.

4. **Timeout Management**: Set appropriate timeouts based on your algorithm's complexity and expected execution time.

5. **Agent Roles**: Clearly define roles for each agent in your algorithm to ensure proper communication patterns.

6. **Testing**: Test your algorithms with different agent configurations and edge cases.

7. **Documentation**: Document your custom algorithms thoroughly, including expected inputs, outputs, and behavior.

## Integration with Other Swarms Components

Social Algorithms integrate seamlessly with other Swarms components:

| Component   | Integration Description                                                     |
|-------------|----------------------------------------------------------------------------|
| **Agents**  | Use any Swarms Agent in your social algorithms                             |
| **Tools**   | Agents can use tools within social algorithms                              |
| **Memory**  | Agents can access long-term memory during algorithm execution              |
| **Workflows**| Social algorithms can be used as steps in larger workflows                |
| **Routers** | Social algorithms can be used with SwarmRouter for dynamic agent selection |

This framework provides a powerful foundation for building complex multi-agent systems with sophisticated communication patterns and emergent behaviors.
