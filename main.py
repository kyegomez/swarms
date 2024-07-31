import os

import requests

from swarms import Agent, OpenAIChat
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class
model = OpenAIChat(
    api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)


def fetch_financial_news(
    query: str = "Nvidia news", num_articles: int = 5
) -> str:
    """
    Fetches financial news from the Google News API and returns a formatted string of the top news.

    Args:
        api_key (str): Your Google News API key.
        query (str): The query term to search for news. Default is "financial".
        num_articles (int): The number of top articles to fetch. Default is 5.

    Returns:
        str: A formatted string of the top financial news articles.

    Raises:
        ValueError: If the API response is invalid or there are no articles found.
        requests.exceptions.RequestException: If there is an error with the request.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": "ceabc81a7d8f45febfedadb27177f3a3",
        "pageSize": num_articles,
        "sortBy": "relevancy",
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if "articles" not in data or len(data["articles"]) == 0:
            raise ValueError("No articles found or invalid API response.")

        articles = data["articles"]
        formatted_articles = []

        for i, article in enumerate(articles, start=1):
            title = article.get("title", "No Title")
            description = article.get("description", "No Description")
            url = article.get("url", "No URL")
            formatted_articles.append(
                f"{i}. {title}\nDescription: {description}\nRead more: {url}\n"
            )

        return "\n".join(formatted_articles)

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        raise
    except ValueError as e:
        print(f"Value Error: {e}")
        raise


# # Example usage:
# api_key = "ceabc81a7d8f45febfedadb27177f3a3"
# print(fetch_financial_news(api_key))


# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    # system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    llm=model,
    max_loops=2,
    autosave=True,
    # dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    # interactive=True, # Set to False to disable interactive mode
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent.json",
    # tools=[fetch_financial_news],
    # stopping_token="Stop!",
    # interactive=True,
    # docs_folder="docs", # Enter your folder name
    # pdf_path="docs/finance_agent.pdf",
    # sop="Calculate the profit for a company.",
    # sop_list=["Calculate the profit for a company."],
    user_name="swarms_corp",
    # # docs=
    # # docs_folder="docs",
    retry_attempts=3,
    # context_length=1000,
    # tool_schema = dict
    context_length=200000,
    # tool_schema=
    # tools
    # agent_ops_on=True,
    # long_term_memory=ChromaDB(docs_folder="artifacts"),
)


def run_finance_agent(query: str) -> str:
    """
    Runs the financial analysis agent with the given query.

    Args:
        query (str): The user query to run the agent with.

    Returns:
        str: The response from the financial analysis agent.
    """
    query = fetch_financial_news(query)
    print(query)
    response = agent(query)
    return response


# Example usage:
query = "Nvidia news"
response = run_finance_agent(f"Summarize the latest Nvidia financial news {query}")
print(response)
