from swarms import Agent
from swarms.structs.majority_voting import MajorityVoting

# System prompts for different coding expertise areas
PYTHON_EXPERT_PROMPT = """You are a senior Python developer with expertise in clean, efficient code.
Focus on:
- Clean, readable code structure
- Pythonic idioms and best practices
- Efficient algorithms and data structures
- Proper error handling and edge cases
- Documentation and code comments"""

GAME_DEV_EXPERT_PROMPT = """You are a game development specialist with focus on game mechanics.
Focus on:
- Smooth game loop implementation
- Intuitive game controls and mechanics
- Game state management
- Performance optimization for games
- User experience and gameplay flow"""

SOFTWARE_ARCHITECT_PROMPT = """You are a software architect who designs robust, maintainable systems.
Focus on:
- Modular, scalable architecture
- Separation of concerns
- Design patterns and principles
- Code maintainability and extensibility
- Best practices for larger codebases"""

CONSENSUS_EVALUATOR_PROMPT = """You are a senior code reviewer and technical lead who evaluates code quality comprehensively.
Your evaluation criteria include:

1. **Code Quality (25%)**:
   - Readability and clarity
   - Code structure and organization
   - Naming conventions
   - Documentation quality

2. **Functionality (25%)**:
   - Correct game mechanics implementation
   - Complete feature set
   - Error handling and edge cases
   - Game stability and robustness

3. **Performance & Efficiency (20%)**:
   - Algorithm efficiency
   - Memory usage optimization
   - Smooth gameplay performance
   - Resource management

4. **Best Practices (15%)**:
   - Python conventions and idioms
   - Design patterns usage
   - Code reusability
   - Maintainability

5. **Innovation & Creativity (15%)**:
   - Creative solutions
   - Unique features
   - Code elegance
   - Problem-solving approach

Provide detailed analysis with scores for each criterion and a final recommendation.
Compare implementations across multiple loops if applicable."""


def create_code_agents():
    """Create multiple coding agents with different expertise"""
    agents = []

    # Python Expert Agent
    python_expert = Agent(
        agent_name="Python-Code-Expert",
        agent_description="Senior Python developer specializing in clean, efficient code",
        system_prompt=PYTHON_EXPERT_PROMPT,
        model_name="gpt-4o",
        max_loops=1,
        max_tokens=4000,
        temperature=0.7,
    )
    agents.append(python_expert)

    # Game Development Expert Agent
    game_expert = Agent(
        agent_name="Game-Dev-Specialist",
        agent_description="Game development expert focusing on mechanics and user experience",
        system_prompt=GAME_DEV_EXPERT_PROMPT,
        model_name="gpt-4o",
        max_loops=1,
        max_tokens=4000,
        temperature=0.7,
    )
    agents.append(game_expert)

    return agents


def create_consensus_agent():
    """Create the consensus agent for code quality evaluation"""
    return Agent(
        agent_name="Code-Quality-Evaluator",
        agent_description="Senior code reviewer evaluating implementations across multiple criteria",
        system_prompt=CONSENSUS_EVALUATOR_PROMPT,
        model_name="gpt-4o",
        max_loops=1,
        max_tokens=5000,
        temperature=0.3,  # Lower temperature for more consistent evaluation
    )


def main():
    """Main function to run the Snake game code quality voting example"""

    # Create agents
    coding_agents = create_code_agents()
    consensus_agent = create_consensus_agent()

    # Create Majority Voting system with multi-loop capability
    snake_game_voting = MajorityVoting(
        name="Snake-Game-Code-Quality-Voting",
        description="Multi-agent system for creating and evaluating Snake game implementations",
        agents=coding_agents,
        consensus_agent=consensus_agent,
        max_loops=1,  # Enable multi-loop refinement
        verbose=True,
    )

    # Define the coding task
    coding_task = """
Create a complete Snake game implementation in Python using the following requirements:

**Core Requirements:**
1. **Game Board**: 20x20 grid with borders
2. **Snake**: Starts at center, grows when eating food
3. **Food**: Randomly placed, appears after being eaten
4. **Controls**: Arrow keys or WASD for movement
5. **Game Over**: When snake hits wall or itself
6. **Score**: Display current score and high score

**Technical Requirements:**
- Use Python standard library only (no external dependencies)
- Clean, well-documented code
- Proper error handling
- Efficient algorithms
- Modular design where possible

**Advanced Features (Bonus):**
- Increasing speed as score grows
- Pause functionality
- Game restart option
- Score persistence
- Smooth animations

**Output Format:**
Provide the complete, runnable Python code with proper structure and documentation.
Make sure the code is immediately executable and includes a main function.
"""

    # Run the multi-loop voting system
    result = snake_game_voting.run(coding_task)

    return result


if __name__ == "__main__":
    main()
