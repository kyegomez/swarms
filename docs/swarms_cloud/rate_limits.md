# Swarms API Rate Limits

The Swarms API implements rate limiting to ensure fair usage and system stability. Here are the current limits:

## Standard Rate Limits

- **General API Requests**: 100 requests per minute
- **Batch Operations**: Maximum 10 requests per batch for agent/swarm batch operations

## Rate Limit Response

When you exceed the rate limit, the API will return a 429 (Too Many Requests) status code with the following message:
```json
{
    "detail": "Rate limit exceeded. Please try again later."
}
```

## Batch Operation Limits

For batch operations (`/v1/agent/batch/completions` and `/v1/swarm/batch/completions`):

- Maximum 10 concurrent requests per batch

- Exceeding this limit will result in a 400 (Bad Request) error

## Increasing Your Rate Limits

Need higher rate limits for your application? You can increase your limits by subscribing to a higher tier plan at [swarms.world/pricing](https://swarms.world/pricing).

Higher tier plans offer:

- Increased rate limits

- Higher batch operation limits

- Priority processing

- Dedicated support

## Best Practices

To make the most of your rate limits:

1. Implement proper error handling for rate limit responses

2. Use batch operations when processing multiple requests

3. Add appropriate retry logic with exponential backoff

4. Monitor your API usage to stay within limits

## Rate Limit Headers

The API does not currently expose rate limit headers. We recommend implementing your own request tracking to stay within the limits.

---

For questions about rate limits or to request a custom plan for higher limits, please contact our support team or visit [swarms.world/pricing](https://swarms.world/pricing).