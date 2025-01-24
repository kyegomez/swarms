# TwitterTool Technical Documentation

## Class Overview

The TwitterTool class provides a comprehensive interface for interacting with Twitter's API. It encapsulates common Twitter operations and provides error handling, logging, and integration capabilities with AI agents.

## Installation Requirements

```bash
pip install swarms-tools
```

## Class Definition

### Constructor Parameters

The `options` dictionary accepts the following keys:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|----------|-------------|
| id | str | No | "twitter_plugin" | Unique identifier for the tool instance |
| name | str | No | "Twitter Plugin" | Display name for the tool |
| description | str | No | Default description | Tool description |
| credentials | Dict[str, str] | Yes | None | Twitter API credentials |

The `credentials` dictionary requires:

| Credential | Type | Required | Description |
|------------|------|----------|-------------|
| apiKey | str | Yes | Twitter API key |
| apiSecretKey | str | Yes | Twitter API secret key |
| accessToken | str | Yes | Twitter access token |
| accessTokenSecret | str | Yes | Twitter access token secret |

## Public Methods

### available_functions

```python
@property
def available_functions(self) -> List[str]
```

Returns a list of available function names that can be executed by the tool.

**Returns:**
- List[str]: Names of available functions ['get_metrics', 'reply_tweet', 'post_tweet', 'like_tweet', 'quote_tweet']

**Example:**
```python
twitter_tool = TwitterTool(options)
functions = twitter_tool.available_functions
print(f"Available functions: {functions}")
```

### get_function

```python
def get_function(self, fn_name: str) -> Callable
```

Retrieves a specific function by name for execution.

**Parameters:**
- fn_name (str): Name of the function to retrieve

**Returns:**
- Callable: The requested function

**Raises:**
- ValueError: If the function name is not found

**Example:**
```python
post_tweet = twitter_tool.get_function('post_tweet')
metrics_fn = twitter_tool.get_function('get_metrics')
```

## Protected Methods

### _get_metrics

```python
def _get_metrics(self) -> Dict[str, int]
```

Fetches user metrics including followers, following, and tweet counts.

**Returns:**
- Dict[str, int]: Dictionary containing metrics
  - followers: Number of followers
  - following: Number of accounts following
  - tweets: Total tweet count

**Example:**
```python
metrics = twitter_tool._get_metrics()
print(f"Followers: {metrics['followers']}")
print(f"Following: {metrics['following']}")
print(f"Total tweets: {metrics['tweets']}")
```

### _reply_tweet

```python
def _reply_tweet(self, tweet_id: int, reply: str) -> None
```

Posts a reply to a specific tweet.

**Parameters:**
- tweet_id (int): ID of the tweet to reply to
- reply (str): Text content of the reply

**Example:**
```python
twitter_tool._reply_tweet(
    tweet_id=1234567890,
    reply="Thank you for sharing this insight!"
)
```

### _post_tweet

```python
def _post_tweet(self, tweet: str) -> Dict[str, Any]
```

Creates a new tweet.

**Parameters:**
- tweet (str): Text content of the tweet

**Returns:**
- Dict[str, Any]: Response from Twitter API

**Example:**
```python
twitter_tool._post_tweet(
    tweet="Exploring the fascinating world of AI and machine learning! #AI #ML"
)
```

### _like_tweet

```python
def _like_tweet(self, tweet_id: int) -> None
```

Likes a specific tweet.

**Parameters:**
- tweet_id (int): ID of the tweet to like

**Example:**
```python
twitter_tool._like_tweet(tweet_id=1234567890)
```

### _quote_tweet

```python
def _quote_tweet(self, tweet_id: int, quote: str) -> None
```

Creates a quote tweet.

**Parameters:**
- tweet_id (int): ID of the tweet to quote
- quote (str): Text content to add to the quote

**Example:**
```python
twitter_tool._quote_tweet(
    tweet_id=1234567890,
    quote="This is a fascinating perspective on AI development!"
)
```

## Integration Examples

### Basic Usage with Environment Variables

```python
import os
from dotenv import load_dotenv
from swarms_tools import TwitterTool

load_dotenv()

options = {
    "id": "twitter_bot",
    "name": "AI Twitter Bot",
    "description": "An AI-powered Twitter bot",
    "credentials": {
        "apiKey": os.getenv("TWITTER_API_KEY"),
        "apiSecretKey": os.getenv("TWITTER_API_SECRET_KEY"),
        "accessToken": os.getenv("TWITTER_ACCESS_TOKEN"),
        "accessTokenSecret": os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
    }
}

twitter_tool = TwitterTool(options)
```

### Integration with AI Agent for Content Generation

```python
from swarms import Agent
from swarms_models import OpenAIChat
from swarms_tools import TwitterTool
import os

from dotenv import load_dotenv

load_dotenv()

# Initialize the model
model = OpenAIChat(
    model_name="gpt-4",
    max_tokens=3000,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create content generation agent
content_agent = Agent(
    agent_name="Content Creator",
    system_prompt="""
    You are an expert content creator for Twitter.
    Create engaging, informative tweets that:
    1. Are under 280 characters
    2. Use appropriate hashtags
    3. Maintain a professional tone
    4. Include relevant calls to action
    """,
    llm=model
)

class TwitterContentBot:
    def __init__(self, twitter_tool: TwitterTool, agent: Agent):
        self.twitter_tool = twitter_tool
        self.agent = agent
        self.post_tweet = twitter_tool.get_function('post_tweet')
    
    def generate_and_post(self, topic: str) -> None:
        """
        Generates and posts content about a specific topic.
        
        Args:
            topic (str): The topic to create content about
        """
        prompt = f"Create an engaging tweet about {topic}"
        tweet_content = self.agent.run(prompt)
        self.post_tweet(tweet_content)

# Usage
bot = TwitterContentBot(twitter_tool, content_agent)
bot.generate_and_post("artificial intelligence")
```

### Automated Engagement System

```python
import time
from typing import List, Dict

class TwitterEngagementSystem:
    def __init__(self, twitter_tool: TwitterTool):
        self.twitter_tool = twitter_tool
        self.like_tweet = twitter_tool.get_function('like_tweet')
        self.reply_tweet = twitter_tool.get_function('reply_tweet')
        self.get_metrics = twitter_tool.get_function('get_metrics')
        
        # Track engagement history
        self.engagement_history: List[Dict] = []
    
    def engage_with_tweet(self, tweet_id: int, should_like: bool = True, 
                         reply_text: Optional[str] = None) -> None:
        """
        Engages with a specific tweet through likes and replies.
        
        Args:
            tweet_id (int): The ID of the tweet to engage with
            should_like (bool): Whether to like the tweet
            reply_text (Optional[str]): Text to reply with, if any
        """
        try:
            if should_like:
                self.like_tweet(tweet_id)
            
            if reply_text:
                self.reply_tweet(tweet_id, reply_text)
            
            # Record engagement
            self.engagement_history.append({
                'timestamp': time.time(),
                'tweet_id': tweet_id,
                'actions': {
                    'liked': should_like,
                    'replied': bool(reply_text)
                }
            })
        
        except Exception as e:
            print(f"Failed to engage with tweet {tweet_id}: {e}")

# Usage
engagement_system = TwitterEngagementSystem(twitter_tool)
engagement_system.engage_with_tweet(
    tweet_id=1234567890,
    should_like=True,
    reply_text="Great insights! Thanks for sharing."
)
```

### Scheduled Content System

```python
import schedule
from datetime import datetime, timedelta

class ScheduledTwitterBot:
    def __init__(self, twitter_tool: TwitterTool, agent: Agent):
        self.twitter_tool = twitter_tool
        self.agent = agent
        self.post_tweet = twitter_tool.get_function('post_tweet')
        self.posted_tweets: List[str] = []
    
    def generate_daily_content(self) -> None:
        """
        Generates and posts daily content based on the current date.
        """
        today = datetime.now()
        prompt = f"""
        Create an engaging tweet for {today.strftime('%A, %B %d')}.
        Focus on technology trends and insights.
        """
        
        tweet_content = self.agent.run(prompt)
        
        # Avoid duplicate content
        if tweet_content not in self.posted_tweets:
            self.post_tweet(tweet_content)
            self.posted_tweets.append(tweet_content)
            
            # Keep only last 100 tweets in memory
            if len(self.posted_tweets) > 100:
                self.posted_tweets.pop(0)

# Usage
scheduled_bot = ScheduledTwitterBot(twitter_tool, content_agent)

# Schedule daily posts
schedule.every().day.at("10:00").do(scheduled_bot.generate_daily_content)
schedule.every().day.at("15:00").do(scheduled_bot.generate_daily_content)

# Run the scheduler
while True:
    schedule.run_pending()
    time.sleep(60)
```

## Error Handling

The TwitterTool implements comprehensive error handling through try-except blocks. All methods catch and handle `tweepy.TweepyException` for Twitter API-specific errors. Here's an example of implementing custom error handling:

```python
class TwitterToolWithCustomErrors(TwitterTool):
    def _post_tweet(self, tweet: str) -> Dict[str, Any]:
        """
        Enhanced tweet posting with custom error handling.
        
        Args:
            tweet (str): The tweet content to post
            
        Returns:
            Dict[str, Any]: Response from Twitter API
            
        Raises:
            ValueError: If tweet exceeds character limit
        """
        if len(tweet) > 280:
            raise ValueError("Tweet exceeds 280 character limit")
            
        try:
            return super()._post_tweet(tweet)
        except tweepy.TweepyException as e:
            self.logger.error(f"Twitter API error: {e}")
            raise
```

## Rate Limiting

Twitter's API has rate limits that should be respected. Here's an example implementation of rate limiting:

```python
from time import time, sleep
from collections import deque

class RateLimitedTwitterTool(TwitterTool):
    def __init__(self, options: Dict[str, Any]) -> None:
        super().__init__(options)
        self.tweet_timestamps = deque(maxlen=300)  # Track last 300 tweets
        self.max_tweets_per_15min = 300
        
    def _check_rate_limit(self) -> None:
        """
        Checks if we're within rate limits and waits if necessary.
        """
        now = time()
        
        # Remove timestamps older than 15 minutes
        while self.tweet_timestamps and self.tweet_timestamps[0] < now - 900:
            self.tweet_timestamps.popleft()
        
        if len(self.tweet_timestamps) >= self.max_tweets_per_15min:
            sleep_time = 900 - (now - self.tweet_timestamps[0])
            if sleep_time > 0:
                sleep(sleep_time)
        
        self.tweet_timestamps.append(now)

    def _post_tweet(self, tweet: str) -> Dict[str, Any]:
        self._check_rate_limit()
        return super()._post_tweet(tweet)
```

## Best Practices

1. Always use environment variables for credentials:
```python
from dotenv import load_dotenv
load_dotenv()

options = {
    "credentials": {
        "apiKey": os.getenv("TWITTER_API_KEY"),
        "apiSecretKey": os.getenv("TWITTER_API_SECRET_KEY"),
        "accessToken": os.getenv("TWITTER_ACCESS_TOKEN"),
        "accessTokenSecret": os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
    }
}
```

2. Implement proper logging:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

3. Use type hints consistently:
```python
from typing import Optional, Dict, Any, List

def process_tweet(
    tweet_id: int,
    action: str,
    content: Optional[str] = None
) -> Dict[str, Any]:
    pass
```

4. Handle API rate limits gracefully:
```python
from time import sleep

def retry_with_backoff(func, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return func()
        except tweepy.TweepyException as e:
            if attempt == max_retries - 1:
                raise
            sleep_time = (2 ** attempt) + random.uniform(0, 1)
            sleep(sleep_time)
```

5. Validate input data:
```python
def validate_tweet(tweet: str) -> bool:
    if not tweet or len(tweet) > 280:
        return False
    return True
```

## Testing

Example of a basic test suite for the TwitterTool:

```python
from swarms_tools import TwitterTool
import os
from dotenv import load_dotenv

load_dotenv()

def test_twitter_tool():
    # Test configuration
    options = {
        "id": "test_bot",
        "credentials": {
            "apiKey": os.getenv("TWITTER_API_KEY"),
            "apiSecretKey": os.getenv("TWITTER_API_SECRET_KEY"),
            "accessToken": os.getenv("TWITTER_ACCESS_TOKEN"),
            "accessTokenSecret": os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        }
    }
    
    # Initialize tool
    tool = TwitterTool(options)
    
    # Test available functions
    assert 'post_tweet' in tool.available_functions
    assert 'get_metrics' in tool.available_functions
    
    # Test function retrieval
    post_tweet = tool.get_function('post_tweet')
    assert callable(post_tweet)
    
    # Test metrics
    metrics = tool._get_metrics()
    assert isinstance(metrics, dict)
    assert 'followers' in metrics
    assert 'following' in metrics
    assert 'tweets' in metrics
```

Remember to always test your code thoroughly before deploying it in a production environment. The TwitterTool is designed to be extensible, so you can add new features and customizations as needed for your specific use case.