from swarms import BaseSwarm, AutoSwarm, AutoSwarmRouter, Agent, Anthropic


class MarketingSwarm(BaseSwarm):
    """
    A class representing a marketing swarm.

    Attributes:
        name (str): The name of the marketing swarm.
        market_trend_analyzer (Agent): An agent for analyzing market trends.
        content_idea_generator (Agent): An agent for generating content ideas.
        campaign_optimizer (Agent): An agent for optimizing marketing campaigns.

    Methods:
        run(task: str, *args, **kwargs) -> Any: Runs the marketing swarm for the given task.

    """

    def __init__(self, name="kyegomez/marketingswarm", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

        # Agent for market trend analyzer
        self.market_trend_analyzer = Agent(
            agent_name="Market Trend Analyzer",
            system_prompt="Analyze market trends to identify opportunities for marketing campaigns.",
            llm=Anthropic(),
            meax_loops=1,
            autosave=True,
            dashboard=False,
            streaming_on=True,
            verbose=True,
            stopping_token="<DONE>",
        )

        # Agent for content idea generator
        self.content_idea_generator = Agent(
            agent_name="Content Idea Generator",
            system_prompt="Generate content ideas based on market trends.",
            llm=Anthropic(),
            max_loops=1,
            autosave=True,
            dashboard=False,
            streaming_on=True,
            verbose=True,
            stopping_token="<DONE>",
        )

        # Agent for campaign optimizer
        self.campaign_optimizer = Agent(
            agent_name="Campaign Optimizer",
            system_prompt="Optimize marketing campaigns based on content ideas and market trends.",
            llm=Anthropic(),
            max_loops=1,
            autosave=True,
            dashboard=False,
            streaming_on=True,
            verbose=True,
            stopping_token="<DONE>",
        )

    def run(self, task: str, *args, **kwargs):
        """
        Runs the marketing swarm for the given task.

        Args:
            task (str): The task to be performed by the marketing swarm.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The result of running the marketing swarm.

        """
        # Analyze market trends
        analyzed_trends = self.market_trend_analyzer.run(
            task, *args, **kwargs
        )

        # Generate content ideas based on market trends
        content_ideas = self.content_idea_generator.run(
            task, analyzed_trends, *args, **kwargs
        )

        # Optimize marketing campaigns based on content ideas and market trends
        optimized_campaigns = self.campaign_optimizer.run(
            task, content_ideas, analyzed_trends, *args, **kwargs
        )

        return optimized_campaigns
