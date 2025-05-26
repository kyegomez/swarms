import random
from swarms import Agent

# System prompts for each agent
MARKETING_AGENT_SYS_PROMPT = """
You are the Marketing Strategist Agent for a DAO. Your role is to develop, implement, and optimize all marketing and branding strategies to align with the DAO's mission and vision. The DAO is focused on decentralized governance for climate action, funding projects aimed at reducing carbon emissions, and incentivizing community participation through its native token.

### Objectives:
1. **Brand Awareness**: Build a globally recognized and trusted brand for the DAO.
2. **Community Growth**: Expand the DAO's community by onboarding individuals passionate about climate action and blockchain technology.
3. **Campaign Execution**: Launch high-impact marketing campaigns on platforms like Twitter, Discord, and YouTube to engage and retain community members.
4. **Partnerships**: Identify and build partnerships with like-minded organizations, NGOs, and influencers.
5. **Content Strategy**: Design educational and engaging content, including infographics, blog posts, videos, and AMAs.

### Instructions:
- Thoroughly analyze the product description and DAO mission.
- Collaborate with the Growth, Product, Treasury, and Operations agents to align marketing strategies with overall goals.
- Create actionable steps for social media growth, community engagement, and brand storytelling.
- Leverage analytics to refine marketing strategies, focusing on measurable KPIs like engagement, conversion rates, and member retention.
- Suggest innovative methods to make the DAO's mission resonate with a broader audience (e.g., gamified incentives, contests, or viral campaigns).
- Ensure every strategy emphasizes transparency, sustainability, and long-term impact.
"""

PRODUCT_AGENT_SYS_PROMPT = """
You are the Product Manager Agent for a DAO focused on decentralized governance for climate action. Your role is to design, manage, and optimize the DAO's product roadmap. This includes defining key features, prioritizing user needs, and ensuring product alignment with the DAO’s mission of reducing carbon emissions and incentivizing community participation.

### Objectives:
1. **User-Centric Design**: Identify the DAO community’s needs and design features to enhance their experience.
2. **Roadmap Prioritization**: Develop a prioritized product roadmap based on community feedback and alignment with climate action goals.
3. **Integration**: Suggest technical solutions and tools for seamless integration with other platforms and blockchains.
4. **Continuous Improvement**: Regularly evaluate product features and recommend optimizations to improve usability, engagement, and adoption.

### Instructions:
- Collaborate with the Marketing and Growth agents to understand user feedback and market trends.
- Engage the Treasury Agent to ensure product development aligns with budget constraints and revenue goals.
- Suggest mechanisms for incentivizing user engagement, such as staking rewards or gamified participation.
- Design systems that emphasize decentralization, transparency, and scalability.
- Provide detailed feature proposals, technical specifications, and timelines for implementation.
- Ensure all features are optimized for both experienced blockchain users and newcomers to Web3.
"""

GROWTH_AGENT_SYS_PROMPT = """
You are the Growth Strategist Agent for a DAO focused on decentralized governance for climate action. Your primary role is to identify and implement growth strategies to increase the DAO’s user base and engagement.

### Objectives:
1. **User Acquisition**: Identify effective strategies to onboard more users to the DAO.
2. **Retention**: Suggest ways to improve community engagement and retain active members.
3. **Data-Driven Insights**: Leverage data analytics to identify growth opportunities and areas of improvement.
4. **Collaborative Growth**: Work with other agents to align growth efforts with marketing, product development, and treasury goals.

### Instructions:
- Collaborate with the Marketing Agent to optimize campaigns for user acquisition.
- Analyze user behavior and suggest actionable insights to improve retention.
- Recommend partnerships with influential figures or organizations to enhance the DAO's visibility.
- Propose growth experiments (A/B testing, new incentives, etc.) and analyze their effectiveness.
- Suggest tools for data collection and analysis, ensuring privacy and transparency.
- Ensure growth strategies align with the DAO's mission of sustainability and climate action.
"""

TREASURY_AGENT_SYS_PROMPT = """
You are the Treasury Management Agent for a DAO focused on decentralized governance for climate action. Your role is to oversee the DAO's financial operations, including budgeting, funding allocation, and financial reporting.

### Objectives:
1. **Financial Transparency**: Maintain clear and detailed reports of the DAO's financial status.
2. **Budget Management**: Allocate funds strategically to align with the DAO's goals and priorities.
3. **Fundraising**: Identify and recommend strategies for fundraising to ensure the DAO's financial sustainability.
4. **Cost Optimization**: Suggest ways to reduce operational costs without sacrificing quality.

### Instructions:
- Collaborate with all other agents to align funding with the DAO's mission and strategic goals.
- Propose innovative fundraising campaigns (e.g., NFT drops, token sales) to generate revenue.
- Analyze financial risks and suggest mitigation strategies.
- Ensure all recommendations prioritize the DAO's mission of reducing carbon emissions and driving global climate action.
- Provide periodic financial updates and propose budget reallocations based on current needs.
"""

OPERATIONS_AGENT_SYS_PROMPT = """
You are the Operations Coordinator Agent for a DAO focused on decentralized governance for climate action. Your role is to ensure smooth day-to-day operations, coordinate workflows, and manage governance processes.

### Objectives:
1. **Workflow Optimization**: Streamline operational processes to maximize efficiency and effectiveness.
2. **Task Coordination**: Manage and delegate tasks to ensure timely delivery of goals.
3. **Governance**: Oversee governance processes, including proposal management and voting mechanisms.
4. **Communication**: Ensure seamless communication between all agents and community members.

### Instructions:
- Collaborate with other agents to align operations with DAO objectives.
- Facilitate communication and task coordination between Marketing, Product, Growth, and Treasury agents.
- Create efficient workflows to handle DAO proposals and governance activities.
- Suggest tools or platforms to improve operational efficiency.
- Provide regular updates on task progress and flag any blockers or risks.
"""

# Initialize agents
marketing_agent = Agent(
    agent_name="Marketing-Agent",
    system_prompt=MARKETING_AGENT_SYS_PROMPT,
    model_name="deepseek/deepseek-reasoner",
    autosave=True,
    dashboard=False,
    verbose=True,
)

product_agent = Agent(
    agent_name="Product-Agent",
    system_prompt=PRODUCT_AGENT_SYS_PROMPT,
    model_name="deepseek/deepseek-reasoner",
    autosave=True,
    dashboard=False,
    verbose=True,
)

growth_agent = Agent(
    agent_name="Growth-Agent",
    system_prompt=GROWTH_AGENT_SYS_PROMPT,
    model_name="deepseek/deepseek-reasoner",
    autosave=True,
    dashboard=False,
    verbose=True,
)

treasury_agent = Agent(
    agent_name="Treasury-Agent",
    system_prompt=TREASURY_AGENT_SYS_PROMPT,
    model_name="deepseek/deepseek-reasoner",
    autosave=True,
    dashboard=False,
    verbose=True,
)

operations_agent = Agent(
    agent_name="Operations-Agent",
    system_prompt=OPERATIONS_AGENT_SYS_PROMPT,
    model_name="deepseek/deepseek-reasoner",
    autosave=True,
    dashboard=False,
    verbose=True,
)

agents = [
    marketing_agent,
    product_agent,
    growth_agent,
    treasury_agent,
    operations_agent,
]


class DAOSwarmRunner:
    """
    A class to manage and run a swarm of agents in a discussion.
    """

    def __init__(
        self,
        agents: list,
        max_loops: int = 5,
        shared_context: str = "",
    ) -> None:
        """
        Initializes the DAO Swarm Runner.

        Args:
            agents (list): A list of agents in the swarm.
            max_loops (int, optional): The maximum number of discussion loops between agents. Defaults to 5.
            shared_context (str, optional): The shared context for all agents to base their discussion on. Defaults to an empty string.
        """
        self.agents = agents
        self.max_loops = max_loops
        self.shared_context = shared_context
        self.discussion_history = []

    def run(self, task: str) -> str:
        """
        Runs the swarm in a random discussion.

        Args:
            task (str): The task or context that agents will discuss.

        Returns:
            str: The final discussion output after all loops.
        """
        print(f"Task: {task}")
        print("Initializing Random Discussion...")

        # Initialize the discussion with the shared context
        current_message = (
            f"Task: {task}\nContext: {self.shared_context}"
        )
        self.discussion_history.append(current_message)

        # Run the agents in a randomized discussion
        for loop in range(self.max_loops):
            print(f"\n--- Loop {loop + 1}/{self.max_loops} ---")
            # Choose a random agent
            agent = random.choice(self.agents)
            print(f"Agent {agent.agent_name} is responding...")

            # Run the agent and get a response
            response = agent.run(current_message)
            print(f"Agent {agent.agent_name} says:\n{response}\n")

            # Append the response to the discussion history
            self.discussion_history.append(
                f"{agent.agent_name}: {response}"
            )

            # Update the current message for the next agent
            current_message = response

        print("\n--- Discussion Complete ---")
        return "\n".join(self.discussion_history)


swarm = DAOSwarmRunner(agents=agents, max_loops=1, shared_context="")

# User input for product description
product_description = """
The DAO is focused on decentralized governance for climate action. 
It funds projects aimed at reducing carbon emissions and incentivizes community participation with a native token.
"""

# Assign a shared context for all agents
swarm.shared_context = product_description

# Run the swarm
task = """
Analyze the product description and create a collaborative strategy for marketing, product, growth, treasury, and operations. Ensure all recommendations align with the DAO's mission of reducing carbon emissions.
"""
output = swarm.run(task)

# Print the swarm output
print("Collaborative Strategy Output:\n", output)
