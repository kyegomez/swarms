"""
Stagehand Multi-Agent Browser Automation Workflows
=================================================

This example demonstrates advanced multi-agent workflows using Stagehand
for complex browser automation scenarios. It shows how multiple agents
can work together to accomplish sophisticated web tasks.

Use cases:
1. E-commerce price monitoring across multiple sites
2. Competitive analysis and market research
3. Automated testing and validation workflows
4. Data aggregation from multiple sources
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

from swarms import Agent, SequentialWorkflow, ConcurrentWorkflow
from swarms.structs.agent_rearrange import AgentRearrange
from examples.stagehand.stagehand_wrapper_agent import StagehandAgent

load_dotenv()


# Pydantic models for structured data
class ProductInfo(BaseModel):
    """Product information schema."""
    name: str = Field(..., description="Product name")
    price: float = Field(..., description="Product price")
    availability: str = Field(..., description="Availability status")
    url: str = Field(..., description="Product URL")
    screenshot_path: Optional[str] = Field(None, description="Screenshot file path")


class MarketAnalysis(BaseModel):
    """Market analysis report schema."""
    timestamp: datetime = Field(default_factory=datetime.now)
    products: List[ProductInfo] = Field(..., description="List of products analyzed")
    price_range: Dict[str, float] = Field(..., description="Min and max prices")
    recommendations: List[str] = Field(..., description="Analysis recommendations")


# Specialized browser agents
class ProductScraperAgent(StagehandAgent):
    """Specialized agent for scraping product information."""
    
    def __init__(self, site_name: str, *args, **kwargs):
        super().__init__(
            agent_name=f"ProductScraper_{site_name}",
            *args,
            **kwargs
        )
        self.site_name = site_name


class PriceMonitorAgent(StagehandAgent):
    """Specialized agent for monitoring price changes."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            agent_name="PriceMonitorAgent",
            *args,
            **kwargs
        )


# Example 1: E-commerce Price Comparison Workflow
def create_price_comparison_workflow():
    """
    Create a workflow that compares prices across multiple e-commerce sites.
    """
    
    # Create specialized agents for different sites
    amazon_agent = StagehandAgent(
        agent_name="AmazonScraperAgent",
        model_name="gpt-4o-mini",
        env="LOCAL",
    )
    
    ebay_agent = StagehandAgent(
        agent_name="EbayScraperAgent",
        model_name="gpt-4o-mini",
        env="LOCAL",
    )
    
    analysis_agent = Agent(
        agent_name="PriceAnalysisAgent",
        model_name="gpt-4o-mini",
        system_prompt="""You are a price analysis expert. Analyze product prices from multiple sources
        and provide insights on the best deals, price trends, and recommendations.
        Focus on value for money and highlight any significant price differences.""",
    )
    
    # Create concurrent workflow for parallel scraping
    scraping_workflow = ConcurrentWorkflow(
        agents=[amazon_agent, ebay_agent],
        max_loops=1,
        verbose=True,
    )
    
    # Create sequential workflow: scrape -> analyze
    full_workflow = SequentialWorkflow(
        agents=[scraping_workflow, analysis_agent],
        max_loops=1,
        verbose=True,
    )
    
    return full_workflow


# Example 2: Competitive Analysis Workflow
def create_competitive_analysis_workflow():
    """
    Create a workflow for competitive analysis across multiple company websites.
    """
    
    # Agent for extracting company information
    company_researcher = StagehandAgent(
        agent_name="CompanyResearchAgent",
        model_name="gpt-4o-mini",
        env="LOCAL",
    )
    
    # Agent for analyzing social media presence
    social_media_agent = StagehandAgent(
        agent_name="SocialMediaAnalysisAgent",
        model_name="gpt-4o-mini",
        env="LOCAL",
    )
    
    # Agent for compiling competitive analysis report
    report_compiler = Agent(
        agent_name="CompetitiveAnalysisReporter",
        model_name="gpt-4o-mini",
        system_prompt="""You are a competitive analysis expert. Compile comprehensive reports
        based on company information and social media presence data. Identify strengths,
        weaknesses, and market positioning for each company.""",
    )
    
    # Create agent rearrange for flexible routing
    workflow_pattern = "company_researcher -> social_media_agent -> report_compiler"
    
    competitive_workflow = AgentRearrange(
        agents=[company_researcher, social_media_agent, report_compiler],
        flow=workflow_pattern,
        verbose=True,
    )
    
    return competitive_workflow


# Example 3: Automated Testing Workflow
def create_automated_testing_workflow():
    """
    Create a workflow for automated web application testing.
    """
    
    # Agent for UI testing
    ui_tester = StagehandAgent(
        agent_name="UITestingAgent",
        model_name="gpt-4o-mini",
        env="LOCAL",
    )
    
    # Agent for form validation testing
    form_tester = StagehandAgent(
        agent_name="FormValidationAgent",
        model_name="gpt-4o-mini",
        env="LOCAL",
    )
    
    # Agent for accessibility testing
    accessibility_tester = StagehandAgent(
        agent_name="AccessibilityTestingAgent",
        model_name="gpt-4o-mini",
        env="LOCAL",
    )
    
    # Agent for compiling test results
    test_reporter = Agent(
        agent_name="TestReportCompiler",
        model_name="gpt-4o-mini",
        system_prompt="""You are a QA test report specialist. Compile test results from
        UI, form validation, and accessibility testing into a comprehensive report.
        Highlight any failures, warnings, and provide recommendations for fixes.""",
    )
    
    # Concurrent testing followed by report generation
    testing_workflow = ConcurrentWorkflow(
        agents=[ui_tester, form_tester, accessibility_tester],
        max_loops=1,
        verbose=True,
    )
    
    full_test_workflow = SequentialWorkflow(
        agents=[testing_workflow, test_reporter],
        max_loops=1,
        verbose=True,
    )
    
    return full_test_workflow


# Example 4: News Aggregation and Sentiment Analysis
def create_news_aggregation_workflow():
    """
    Create a workflow for news aggregation and sentiment analysis.
    """
    
    # Multiple news scraper agents
    news_scrapers = []
    news_sites = [
        ("TechCrunch", "https://techcrunch.com"),
        ("HackerNews", "https://news.ycombinator.com"),
        ("Reddit", "https://reddit.com/r/technology"),
    ]
    
    for site_name, url in news_sites:
        scraper = StagehandAgent(
            agent_name=f"{site_name}Scraper",
            model_name="gpt-4o-mini",
            env="LOCAL",
        )
        news_scrapers.append(scraper)
    
    # Sentiment analysis agent
    sentiment_analyzer = Agent(
        agent_name="SentimentAnalyzer",
        model_name="gpt-4o-mini",
        system_prompt="""You are a sentiment analysis expert. Analyze news articles and posts
        to determine overall sentiment (positive, negative, neutral) and identify key themes
        and trends in the technology sector.""",
    )
    
    # Trend identification agent
    trend_identifier = Agent(
        agent_name="TrendIdentifier",
        model_name="gpt-4o-mini",
        system_prompt="""You are a trend analysis expert. Based on aggregated news and sentiment
        data, identify emerging trends, hot topics, and potential market movements in the
        technology sector.""",
    )
    
    # Create workflow: parallel scraping -> sentiment analysis -> trend identification
    scraping_workflow = ConcurrentWorkflow(
        agents=news_scrapers,
        max_loops=1,
        verbose=True,
    )
    
    analysis_workflow = SequentialWorkflow(
        agents=[scraping_workflow, sentiment_analyzer, trend_identifier],
        max_loops=1,
        verbose=True,
    )
    
    return analysis_workflow


# Main execution examples
if __name__ == "__main__":
    print("=" * 70)
    print("Stagehand Multi-Agent Workflow Examples")
    print("=" * 70)
    
    # Example 1: Price Comparison
    print("\nExample 1: E-commerce Price Comparison")
    print("-" * 40)
    
    price_workflow = create_price_comparison_workflow()
    
    # Search for a specific product across multiple sites
    price_result = price_workflow.run(
        """Search for 'iPhone 15 Pro Max 256GB' on:
        1. Amazon - extract price, availability, and seller information
        2. eBay - extract price range, number of listings, and average price
        Take screenshots of search results from both sites.
        Compare the prices and provide recommendations on where to buy."""
    )
    print(f"Price Comparison Result:\n{price_result}")
    
    print("\n" + "=" * 70 + "\n")
    
    # Example 2: Competitive Analysis
    print("Example 2: Competitive Analysis")
    print("-" * 40)
    
    competitive_workflow = create_competitive_analysis_workflow()
    
    competitive_result = competitive_workflow.run(
        """Analyze these three AI companies:
        1. OpenAI - visit openai.com and extract mission, products, and recent announcements
        2. Anthropic - visit anthropic.com and extract their AI safety approach and products
        3. DeepMind - visit deepmind.com and extract research focus and achievements
        
        Then check their Twitter/X presence and recent posts.
        Compile a competitive analysis report comparing their market positioning."""
    )
    print(f"Competitive Analysis Result:\n{competitive_result}")
    
    print("\n" + "=" * 70 + "\n")
    
    # Example 3: Automated Testing
    print("Example 3: Automated Web Testing")
    print("-" * 40)
    
    testing_workflow = create_automated_testing_workflow()
    
    test_result = testing_workflow.run(
        """Test the website example.com:
        1. UI Testing: Check if all main navigation links work, images load, and layout is responsive
        2. Form Testing: If there are any forms, test with valid and invalid inputs
        3. Accessibility: Check for alt texts, ARIA labels, and keyboard navigation
        
        Take screenshots of any issues found and compile a comprehensive test report."""
    )
    print(f"Test Results:\n{test_result}")
    
    print("\n" + "=" * 70 + "\n")
    
    # Example 4: News Aggregation
    print("Example 4: Tech News Aggregation and Analysis")
    print("-" * 40)
    
    news_workflow = create_news_aggregation_workflow()
    
    news_result = news_workflow.run(
        """For each news source:
        1. TechCrunch: Extract the top 5 headlines about AI or machine learning
        2. HackerNews: Extract the top 5 posts related to AI/ML with most points
        3. Reddit r/technology: Extract top 5 posts about AI from the past week
        
        Analyze sentiment and identify emerging trends in AI technology."""
    )
    print(f"News Analysis Result:\n{news_result}")
    
    # Cleanup all browser instances
    print("\n" + "=" * 70)
    print("Cleaning up browser instances...")
    
    # Clean up agents
    for agent in price_workflow.agents:
        if isinstance(agent, StagehandAgent):
            agent.cleanup()
        elif hasattr(agent, 'agents'):  # For nested workflows
            for sub_agent in agent.agents:
                if isinstance(sub_agent, StagehandAgent):
                    sub_agent.cleanup()
    
    print("All workflows completed!")
    print("=" * 70)