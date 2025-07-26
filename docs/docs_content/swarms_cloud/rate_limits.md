# Swarms API Rate Limits 

The Swarms API implements a comprehensive rate limiting system that tracks API requests across multiple time windows and enforces various limits to ensure fair usage and system stability.

## Rate Limits Summary

| Rate Limit Type | Free Tier | Premium Tier | Time Window | Description |
|----------------|-----------|--------------|-------------|-------------|
| **Requests per Minute** | 100 | 2,000 | 1 minute | Maximum API calls per minute |
| **Requests per Hour** | 50 | 10,000 | 1 hour | Maximum API calls per hour |
| **Requests per Day** | 1,200 | 100,000 | 24 hours | Maximum API calls per day |
| **Tokens per Agent** | 200,000 | 2,000,000 | Per request | Maximum tokens per agent |
| **Prompt Length** | 200,000 | 200,000 | Per request | Maximum input tokens per request |
| **Batch Size** | 10 | 10 | Per request | Maximum agents in batch requests |
| **IP-based Fallback** | 100 | 100 | 60 seconds | For requests without API keys |

## Detailed Rate Limit Explanations

### 1. **Request Rate Limits**

These limits control how many API calls you can make within specific time windows.

#### **Per-Minute Limit**

| Tier         | Requests per Minute | Reset Interval         | Applies To         |
|--------------|--------------------|------------------------|--------------------|
| Free         | 100                | Every minute (sliding) | All API endpoints  |
| Premium      | 2,000              | Every minute (sliding) | All API endpoints  |

#### **Per-Hour Limit**

- **Free Tier**: 50 requests per hour
- **Premium Tier**: 10,000 requests per hour
- **Reset**: Every hour (sliding window)
- **Applies to**: All API endpoints

#### **Per-Day Limit**

- **Free Tier**: 1,200 requests per day (50 × 24)

- **Premium Tier**: 100,000 requests per day

- **Reset**: Every 24 hours (sliding window)

- **Applies to**: All API endpoints

### 2. **Token Limits**

These limits control the amount of text processing allowed per request.

#### **Tokens per Agent**

- **Free Tier**: 200,000 tokens per agent

- **Premium Tier**: 2,000,000 tokens per agent

- **Applies to**: Individual agent configurations

- **Includes**: System prompts, task descriptions, and agent names

#### **Prompt Length Limit**

- **All Tiers**: 200,000 tokens maximum

- **Applies to**: Combined input text (task + history + system prompts)

- **Error**: Returns 400 error if exceeded

- **Message**: "Prompt is too long. Please provide a prompt that is less than 10000 tokens."

### 3. **Batch Processing Limits**

These limits control concurrent processing capabilities.

#### **Batch Size Limit**

- **All Tiers**: 10 agents maximum per batch

- **Applies to**: `/v1/agent/batch/completions` endpoint

- **Error**: Returns 400 error if exceeded

- **Message**: "ERROR: BATCH SIZE EXCEEDED - You can only run up to 10 batch agents at a time."

## How Rate Limiting Works

### Database-Based Tracking

The system uses a database-based approach for API key requests:

1. **Request Logging**: Every API request is logged to the `swarms_api_logs` table
2. **Time Window Queries**: The system queries for requests in the last minute, hour, and day
3. **Limit Comparison**: Current counts are compared against configured limits
4. **Request Blocking**: Requests are blocked if any limit is exceeded

### Sliding Windows

Rate limits use sliding windows rather than fixed windows:

- **Minute**: Counts requests in the last 60 seconds

- **Hour**: Counts requests in the last 60 minutes  

- **Day**: Counts requests in the last 24 hours

This provides more accurate rate limiting compared to fixed time windows.

## Checking Your Rate Limits

### API Endpoint

Use the `/v1/rate/limits` endpoint to check your current usage:

```bash
curl -H "x-api-key: your-api-key" \
     https://api.swarms.world/v1/rate/limits
```

### Response Format

```json
{
  "success": true,
  "rate_limits": {
    "minute": {
      "count": 5,
      "limit": 100,
      "exceeded": false,
      "remaining": 95,
      "reset_time": "2024-01-15T10:30:00Z"
    },
    "hour": {
      "count": 25,
      "limit": 50,
      "exceeded": false,
      "remaining": 25,
      "reset_time": "2024-01-15T11:00:00Z"
    },
    "day": {
      "count": 150,
      "limit": 1200,
      "exceeded": false,
      "remaining": 1050,
      "reset_time": "2024-01-16T10:00:00Z"
    }
  },
  "limits": {
    "maximum_requests_per_minute": 100,
    "maximum_requests_per_hour": 50,
    "maximum_requests_per_day": 1200,
    "tokens_per_agent": 200000
  },
  "timestamp": "2024-01-15T10:29:30Z"
}
```

## Handling Rate Limit Errors

### Error Response

When rate limits are exceeded, you'll receive a 429 status code:

```json
{
  "detail": "Rate limit exceeded for minute window(s). Upgrade to Premium for increased limits (2,000/min, 10,000/hour, 100,000/day) at https://swarms.world/platform/account for just $99/month."
}
```

### Best Practices

1. **Monitor Usage**: Regularly check your rate limits using the `/v1/rate/limits` endpoint
2. **Implement Retry Logic**: Use exponential backoff when hitting rate limits
3. **Optimize Requests**: Combine multiple operations into single requests when possible
4. **Upgrade When Needed**: Consider upgrading to Premium for higher limits

## Premium Tier Benefits

Upgrade to Premium for significantly higher limits:

- **20x more requests per minute** (2,000 vs 100)

- **200x more requests per hour** (10,000 vs 50)

- **83x more requests per day** (100,000 vs 1,200)

- **10x more tokens per agent** (2M vs 200K)

Visit [Swarms Platform Account](https://swarms.world/platform/account) to upgrade for just $99/month.

## Performance Considerations

- Database queries are optimized to only count request IDs
- Rate limit checks are cached per request
- Fallback mechanisms ensure system reliability
- Minimal impact on request latency
- Persistent tracking across server restarts 