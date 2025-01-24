# Twitter Tool Documentation

## Overview
The Twitter Tool provides a convenient interface for interacting with Twitter's API through the swarms-tools package. This documentation covers the initialization process and available functions for posting, replying, liking, and quoting tweets, as well as retrieving metrics.

## Installation
```bash
pip install swarms-tools
```

## Authentication
The Twitter Tool requires Twitter API credentials for authentication. These should be stored as environment variables:

```python
TWITTER_ID=your_twitter_id
TWITTER_NAME=your_twitter_name
TWITTER_DESCRIPTION=your_twitter_description
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET_KEY=your_api_secret_key
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
```

## Initialization

### TwitterTool Configuration Options

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | str | Yes | Unique identifier for the Twitter tool instance |
| name | str | Yes | Name of the Twitter tool instance |
| description | str | No | Description of the tool's purpose |
| credentials | dict | Yes | Dictionary containing Twitter API credentials |

### Credentials Dictionary Structure

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| apiKey | str | Yes | Twitter API Key |
| apiSecretKey | str | Yes | Twitter API Secret Key |
| accessToken | str | Yes | Twitter Access Token |
| accessTokenSecret | str | Yes | Twitter Access Token Secret |

## Available Functions

### initialize_twitter_tool()

Creates and returns a new instance of the TwitterTool.

```python
def initialize_twitter_tool() -> TwitterTool:
```

Returns:
- TwitterTool: Initialized Twitter tool instance

### post_tweet()

Posts a new tweet to Twitter.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| tweet | str | Yes | Text content of the tweet to post |

Raises:
- tweepy.TweepyException: If tweet posting fails

### reply_tweet()

Replies to an existing tweet.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| tweet_id | int | Yes | ID of the tweet to reply to |
| reply | str | Yes | Text content of the reply |

Raises:
- tweepy.TweepyException: If reply posting fails

### like_tweet()

Likes a specified tweet.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| tweet_id | int | Yes | ID of the tweet to like |

Raises:
- tweepy.TweepyException: If liking the tweet fails

### quote_tweet()

Creates a quote tweet.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| tweet_id | int | Yes | ID of the tweet to quote |
| quote | str | Yes | Text content to add to the quoted tweet |

Raises:
- tweepy.TweepyException: If quote tweet creation fails

### get_metrics()

Retrieves Twitter metrics.

Returns:
- Dict[str, int]: Dictionary containing various Twitter metrics

Raises:
- tweepy.TweepyException: If metrics retrieval fails

## Usage Examples

### Basic Tweet Posting
```python
from swarms_tools.twitter import initialize_twitter_tool, post_tweet

# Post a simple tweet
post_tweet("Hello, Twitter!")
```

### Interacting with Tweets
```python
# Reply to a tweet
reply_tweet(12345, "Great point!")

# Like a tweet
like_tweet(12345)

# Quote a tweet
quote_tweet(12345, "Adding my thoughts on this!")
```

### Retrieving Metrics
```python
metrics = get_metrics()
print(f"Current metrics: {metrics}")
```

## Error Handling
All functions include built-in error handling and will print error messages if operations fail. It's recommended to implement additional error handling in production environments:

```python
try:
    post_tweet("Hello, Twitter!")
except Exception as e:
    logger.error(f"Tweet posting failed: {e}")
    # Implement appropriate error handling
```


## Production Example

This is an example of how to use the TwitterTool in a production environment using Swarms.

```python

import os
from time import time

from swarm_models import OpenAIChat
from swarms import Agent
from dotenv import load_dotenv

from swarms_tools.social_media.twitter_tool import TwitterTool

load_dotenv()

model_name = "gpt-4o"

model = OpenAIChat(
    model_name=model_name,
    max_tokens=3000,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


medical_coder = Agent(
    agent_name="Medical Coder",
    system_prompt="""
    You are a highly experienced and certified medical coder with extensive knowledge of ICD-10 coding guidelines, clinical documentation standards, and compliance regulations. Your responsibility is to ensure precise, compliant, and well-documented coding for all clinical cases.

    ### Primary Responsibilities:
    1. **Review Clinical Documentation**: Analyze all available clinical records, including specialist inputs, physician notes, lab results, imaging reports, and discharge summaries.
    2. **Assign Accurate ICD-10 Codes**: Identify and assign appropriate codes for primary diagnoses, secondary conditions, symptoms, and complications.
    3. **Ensure Coding Compliance**: Follow the latest ICD-10-CM/PCS coding guidelines, payer-specific requirements, and organizational policies.
    4. **Document Code Justification**: Provide clear, evidence-based rationale for each assigned code.

    ### Detailed Coding Process:
    - **Review Specialist Inputs**: Examine all relevant documentation to capture the full scope of the patient's condition and care provided.
    - **Identify Diagnoses**: Determine the primary and secondary diagnoses, as well as any symptoms or complications, based on the documentation.
    - **Assign ICD-10 Codes**: Select the most accurate and specific ICD-10 codes for each identified diagnosis or condition.
    - **Document Supporting Evidence**: Record the documentation source (e.g., lab report, imaging, or physician note) for each code to justify its assignment.
    - **Address Queries**: Note and flag any inconsistencies, missing information, or areas requiring clarification from providers.

    ### Output Requirements:
    Your response must be clear, structured, and compliant with professional standards. Use the following format:

    1. **Primary Diagnosis Codes**:
        - **ICD-10 Code**: [e.g., E11.9]
        - **Description**: [e.g., Type 2 diabetes mellitus without complications]
        - **Supporting Documentation**: [e.g., Physician's note dated MM/DD/YYYY]
        
    2. **Secondary Diagnosis Codes**:
        - **ICD-10 Code**: [Code]
        - **Description**: [Description]
        - **Order of Clinical Significance**: [Rank or priority]

    3. **Symptom Codes**:
        - **ICD-10 Code**: [Code]
        - **Description**: [Description]

    4. **Complication Codes**:
        - **ICD-10 Code**: [Code]
        - **Description**: [Description]
        - **Relevant Documentation**: [Source of information]

    5. **Coding Notes**:
        - Observations, clarifications, or any potential issues requiring provider input.

    ### Additional Guidelines:
    - Always prioritize specificity and compliance when assigning codes.
    - For ambiguous cases, provide a brief note with reasoning and flag for clarification.
    - Ensure the output format is clean, consistent, and ready for professional use.
    """,
    llm=model,
    max_loops=1,
    dynamic_temperature_enabled=True,
)


# Define your options with the necessary credentials
options = {
    "id": "mcsswarm",
    "name": "mcsswarm",
    "description": "An example Twitter Plugin for testing.",
    "credentials": {
        "apiKey": os.getenv("TWITTER_API_KEY"),
        "apiSecretKey": os.getenv("TWITTER_API_SECRET_KEY"),
        "accessToken": os.getenv("TWITTER_ACCESS_TOKEN"),
        "accessTokenSecret": os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
    },
}

# Initialize the TwitterTool with your options
twitter_plugin = TwitterTool(options)

# # Post a tweet
# post_tweet_fn = twitter_plugin.get_function('post_tweet')
# post_tweet_fn("Hello world!")


# Assuming `twitter_plugin` and `medical_coder` are already initialized
post_tweet = twitter_plugin.get_function("post_tweet")

# Set to track posted tweets and avoid duplicates
posted_tweets = set()


def post_unique_tweet():
    """
    Generate and post a unique tweet. Skip duplicates.
    """
    tweet_prompt = (
        "Share an intriguing, lesser-known fact about a medical disease, and include an innovative, fun, or surprising way to manage or cure it! "
        "Make the response playful, engaging, and inspiring—something that makes people smile while learning. No markdown, just plain text!"
    )

    # Generate a new tweet text
    tweet_text = medical_coder.run(tweet_prompt)

    # Check for duplicates
    if tweet_text in posted_tweets:
        print("Duplicate tweet detected. Skipping...")
        return

    # Post the tweet
    try:
        post_tweet(tweet_text)
        print(f"Posted tweet: {tweet_text}")
        # Add the tweet to the set of posted tweets
        posted_tweets.add(tweet_text)
    except Exception as e:
        print(f"Error posting tweet: {e}")


# Loop to post tweets every 10 seconds
def start_tweet_loop(interval=10):
    """
    Continuously post tweets every `interval` seconds.

    Args:
        interval (int): Time in seconds between tweets.
    """
    print("Starting tweet loop...")
    while True:
        post_unique_tweet()
        time.sleep(interval)


# Start the loop
start_tweet_loop(10)
```


## Best Practices
1. Always store credentials in environment variables
2. Implement rate limiting in production environments
3. Add proper logging for all operations
4. Handle errors gracefully
5. Validate tweet content before posting
6. Monitor API usage limits

## Rate Limits
Be aware of Twitter's API rate limits. Implement appropriate delays between requests in production environments to avoid hitting these limits.

## Dependencies
- tweepy
- python-dotenv
- swarms-tools

## Version Compatibility
- Python 3.7+
- Latest version of swarms-tools package