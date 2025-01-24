# TwitterTool Documentation

## Overview

The TwitterTool is a powerful Python-based interface for interacting with Twitter's API, designed specifically for integration with autonomous agents and AI systems. It provides a streamlined way to perform common Twitter operations while maintaining proper error handling and logging capabilities.

## Installation

Before using the TwitterTool, ensure you have the required dependencies installed:

```bash
pip install tweepy swarms-tools
```

## Basic Configuration

The TwitterTool requires Twitter API credentials for authentication. Here's how to set up the basic configuration:

```python
from swarms_tools.social_media.twitter_tool import TwitterTool

import os

options = {
    "id": "your_unique_id",
    "name": "your_tool_name",
    "description": "Your tool description",
    "credentials": {
        "apiKey": os.getenv("TWITTER_API_KEY"),
        "apiSecretKey": os.getenv("TWITTER_API_SECRET_KEY"),
        "accessToken": os.getenv("TWITTER_ACCESS_TOKEN"),
        "accessTokenSecret": os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
    }
}

twitter_tool = TwitterTool(options)
```

For security, it's recommended to use environment variables for credentials:

```python
import os
from dotenv import load_dotenv

load_dotenv()

options = {
    "id": "twitter_bot",
    "name": "Twitter Bot",
    "credentials": {
        "apiKey": os.getenv("TWITTER_API_KEY"),
        "apiSecretKey": os.getenv("TWITTER_API_SECRET_KEY"),
        "accessToken": os.getenv("TWITTER_ACCESS_TOKEN"),
        "accessTokenSecret": os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
    }
}
```

## Core Functionality

The TwitterTool provides five main functions:

1. **Posting Tweets**: Create new tweets
2. **Replying to Tweets**: Respond to existing tweets
3. **Quoting Tweets**: Share tweets with additional commentary
4. **Liking Tweets**: Engage with other users' content
5. **Fetching Metrics**: Retrieve account statistics

### Basic Usage Examples

```python
# Get a specific function
post_tweet = twitter_tool.get_function('post_tweet')
reply_tweet = twitter_tool.get_function('reply_tweet')
quote_tweet = twitter_tool.get_function('quote_tweet')
like_tweet = twitter_tool.get_function('like_tweet')
get_metrics = twitter_tool.get_function('get_metrics')

# Post a tweet
post_tweet("Hello, Twitter!")

# Reply to a tweet
reply_tweet(tweet_id=123456789, reply="Great point!")

# Quote a tweet
quote_tweet(tweet_id=123456789, quote="Interesting perspective!")

# Like a tweet
like_tweet(tweet_id=123456789)

# Get account metrics
metrics = get_metrics()
print(f"Followers: {metrics['followers']}")
```

## Integration with Agents

The TwitterTool can be particularly powerful when integrated with AI agents. Here are several examples of agent integrations:

### 1. Medical Information Bot

This example shows how to create a medical information bot that shares health facts:

```python
from swarms import Agent
from swarms_models import OpenAIChat

# Initialize the AI model
model = OpenAIChat(
    model_name="gpt-4",
    max_tokens=3000,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create a medical expert agent
medical_expert = Agent(
    agent_name="Medical Expert",
    system_prompt="""
    You are a medical expert sharing evidence-based health information.
    Your tweets should be:
    - Accurate and scientifically sound
    - Easy to understand
    - Engaging and relevant
    - Within Twitter's character limit
    """,
    llm=model
)

# Function to generate and post medical tweets
def post_medical_fact():
    prompt = "Share an interesting medical fact that would be helpful for the general public."
    tweet_text = medical_expert.run(prompt)
    post_tweet = twitter_tool.get_function('post_tweet')
    post_tweet(tweet_text)
```

### 2. News Summarization Bot

This example demonstrates how to create a bot that summarizes news articles:

```python
# Create a news summarization agent
news_agent = Agent(
    agent_name="News Summarizer",
    system_prompt="""
    You are a skilled news editor who excels at creating concise, 
    accurate summaries of news articles while maintaining the key points.
    Your summaries should be:
    - Factual and unbiased
    - Clear and concise
    - Properly attributed
    - Under 280 characters
    """,
    llm=model
)

def summarize_and_tweet(article_url):
    # Generate summary
    prompt = f"Summarize this news article in a tweet-length format: {article_url}"
    summary = news_agent.run(prompt)
    
    # Post the summary
    post_tweet = twitter_tool.get_function('post_tweet')
    post_tweet(f"{summary} Source: {article_url}")
```

### 3. Interactive Q&A Bot

This example shows how to create a bot that responds to user questions:

```python
class TwitterQABot:
    def __init__(self):
        self.twitter_tool = TwitterTool(options)
        self.qa_agent = Agent(
            agent_name="Q&A Expert",
            system_prompt="""
            You are an expert at providing clear, concise answers to questions.
            Your responses should be:
            - Accurate and informative
            - Conversational in tone
            - Limited to 280 characters
            - Include relevant hashtags when appropriate
            """,
            llm=model
        )
    
    def handle_question(self, tweet_id, question):
        # Generate response
        response = self.qa_agent.run(f"Answer this question: {question}")
        
        # Reply to the tweet
        reply_tweet = self.twitter_tool.get_function('reply_tweet')
        reply_tweet(tweet_id=tweet_id, reply=response)

qa_bot = TwitterQABot()
qa_bot.handle_question(123456789, "What causes climate change?")
```

## Best Practices

When using the TwitterTool, especially with agents, consider these best practices:

1. **Rate Limiting**: Implement delays between tweets to comply with Twitter's rate limits:
```python
import time

def post_with_rate_limit(tweet_text, delay=60):
    post_tweet = twitter_tool.get_function('post_tweet')
    post_tweet(tweet_text)
    time.sleep(delay)  # Wait 60 seconds between tweets
```

2. **Content Tracking**: Maintain a record of posted content to avoid duplicates:
```python
posted_tweets = set()

def post_unique_tweet(tweet_text):
    if tweet_text not in posted_tweets:
        post_tweet = twitter_tool.get_function('post_tweet')
        post_tweet(tweet_text)
        posted_tweets.add(tweet_text)
```

3. **Error Handling**: Implement robust error handling for API failures:
```python
def safe_tweet(tweet_text):
    try:
        post_tweet = twitter_tool.get_function('post_tweet')
        post_tweet(tweet_text)
    except Exception as e:
        logging.error(f"Failed to post tweet: {e}")
        # Implement retry logic or fallback behavior
```

4. **Content Validation**: Validate content before posting:
```python
def validate_and_post(tweet_text):
    if len(tweet_text) > 280:
        tweet_text = tweet_text[:277] + "..."
    
    # Check for prohibited content
    prohibited_terms = ["spam", "inappropriate"]
    if any(term in tweet_text.lower() for term in prohibited_terms):
        return False
    
    post_tweet = twitter_tool.get_function('post_tweet')
    post_tweet(tweet_text)
    return True
```

## Advanced Features

### Scheduled Posting

Implement scheduled posting using Python's built-in scheduling capabilities:

```python
from datetime import datetime
import schedule

def scheduled_tweet_job():
    twitter_tool = TwitterTool(options)
    post_tweet = twitter_tool.get_function('post_tweet')
    
    # Generate content using an agent
    content = medical_expert.run("Generate a health tip of the day")
    post_tweet(content)

# Schedule tweets for specific times
schedule.every().day.at("10:00").do(scheduled_tweet_job)
schedule.every().day.at("15:00").do(scheduled_tweet_job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Analytics Integration

Track the performance of your tweets:

```python
class TweetAnalytics:
    def __init__(self, twitter_tool):
        self.twitter_tool = twitter_tool
        self.metrics_history = []
    
    def record_metrics(self):
        get_metrics = self.twitter_tool.get_function('get_metrics')
        current_metrics = get_metrics()
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': current_metrics
        })
    
    def get_growth_rate(self):
        if len(self.metrics_history) < 2:
            return None
        
        latest = self.metrics_history[-1]['metrics']
        previous = self.metrics_history[-2]['metrics']
        
        return {
            'followers_growth': latest['followers'] - previous['followers'],
            'tweets_growth': latest['tweets'] - previous['tweets']
        }
```

## Troubleshooting

Common issues and their solutions:

1. **Authentication Errors**: Double-check your API credentials and ensure they're properly loaded from environment variables.

2. **Rate Limiting**: If you encounter rate limit errors, implement exponential backoff:
```python
import time
from random import uniform

def exponential_backoff(attempt):
    wait_time = min(300, (2 ** attempt) + uniform(0, 1))
    time.sleep(wait_time)

def retry_post(tweet_text, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            post_tweet = twitter_tool.get_function('post_tweet')
            post_tweet(tweet_text)
            return True
        except Exception as e:
            if attempt < max_attempts - 1:
                exponential_backoff(attempt)
            else:
                raise e
```

3. **Content Length Issues**: Implement automatic content truncation:
```python
def truncate_tweet(text, max_length=280):
    if len(text) <= max_length:
        return text
    
    # Try to break at last space before limit
    truncated = text[:max_length-3]
    last_space = truncated.rfind(' ')
    if last_space > 0:
        truncated = truncated[:last_space]
    
    return truncated + "..."
```

Remember to regularly check Twitter's API documentation for any updates or changes to rate limits and functionality. The TwitterTool is designed to be extensible, so you can add new features as needed for your specific use case.