# Twitter Tools

Twitter, X, and other social-media tools let Swarms agents monitor public conversation, draft posts, classify replies, and publish updates. These tools are powerful because they interact with public identity and audience trust. Keep read-only monitoring separate from write-capable posting, and make every write action explicit.

This guide uses "Twitter" to match the navigation label, but the same patterns apply to X API clients and internal social publishing tools.

## Read-Only Monitoring

Start with read-only tools. They are useful for:

- Tracking mentions of a product, repository, or release.
- Finding questions that need a support reply.
- Monitoring competitor announcements.
- Summarizing community feedback after a launch.
- Building a list of source posts for a human review queue.

```python
from datetime import datetime, timezone
from swarms import Agent


def search_social_posts(query: str, max_results: int = 10) -> dict:
    """Search public social posts and return normalized metadata."""
    return {
        "query": query,
        "searched_at": datetime.now(timezone.utc).isoformat(),
        "posts": [
            {
                "id": "post_123",
                "author": "example_user",
                "url": "https://x.com/example_user/status/123",
                "text": "Example post text",
                "created_at": "2026-01-01T00:00:00Z",
            }
        ][:max_results],
    }


agent = Agent(
    agent_name="Community Monitor",
    system_prompt=(
        "Use social search to find relevant public posts. "
        "Summarize sentiment and include URLs for every quoted or paraphrased post."
    ),
    tools=[search_social_posts],
    max_loops=2,
)
```

The agent can now gather public context without the ability to post.

## Drafting Replies

A drafting tool does not need account credentials. It can take source posts and produce candidate replies for review.

```python
def draft_reply(post_text: str, tone: str = "helpful") -> dict:
    """Draft a social reply without publishing it."""
    return {
        "tone": tone,
        "draft": "Thanks for flagging this. We are checking the behavior and will follow up with details.",
        "requires_review": True,
    }
```

Drafting is safer than direct publishing because it lets you keep brand voice, legal claims, and support promises under review.

## Write-Capable Posting

Only add a publishing tool when the workflow genuinely needs autonomous posting. Make the tool strict: it should require the final post text, destination account, and an idempotency key.

```python
def publish_post(account_id: str, text: str, idempotency_key: str) -> dict:
    """Publish a post to a configured social account."""
    if len(text) > 280:
        raise ValueError("Post text is too long for this account configuration")

    return {
        "ok": True,
        "account_id": account_id,
        "post_url": "https://x.com/example/status/456",
        "idempotency_key": idempotency_key,
    }
```

Recommended safeguards:

- Require exact `account_id` rather than relying on a default.
- Validate post length and media count before sending.
- Return the published URL.
- Keep credentials in environment variables.
- Log request metadata, but never log access tokens.
- Use a dry-run mode during development.

## Environment Variables

Typical variables for an X API integration:

- `TWITTER_BEARER_TOKEN` for read-only API access.
- `TWITTER_API_KEY` and `TWITTER_API_SECRET` for OAuth flows.
- `TWITTER_ACCESS_TOKEN` and `TWITTER_ACCESS_TOKEN_SECRET` for account posting.
- `SOCIAL_POST_DRY_RUN=true` while testing.

When credentials are missing, the tool should raise a clear configuration error. Do not let an agent infer that a post was published.

## Monitoring Agent Prompt

Use a prompt that constrains claims and prevents accidental publishing:

```text
You are a social monitoring agent.
Use read-only tools to collect public posts.
Do not publish, like, repost, follow, or DM.
For each recommended reply, include the source post URL and the exact draft text.
Flag legal, pricing, security, or refund questions for human review.
```

For write-capable agents, add account and policy constraints:

```text
Only publish when the user has provided final approved text in the task.
Do not alter names, prices, promises, or dates.
Return the post URL after publishing.
If publication fails, report the provider error exactly.
```

## Output Contract

Social tools should return structured results:

```json
{
  "ok": true,
  "action": "search",
  "account_id": null,
  "checked_at": "2026-01-01T00:00:00Z",
  "items": [],
  "warnings": []
}
```

For publishing:

```json
{
  "ok": true,
  "action": "publish",
  "account_id": "product-main",
  "post_url": "https://x.com/example/status/456",
  "idempotency_key": "launch-2026-01-01-post-1"
}
```

## Common Failure Modes

- The agent drafts a reply without reading the source post.
- A write-capable tool is exposed to a broad research agent.
- Missing credentials are hidden behind a fake success response.
- Duplicate posts are created because there is no idempotency key.
- The final answer quotes social posts without URLs.

Design tools so these failures are hard to trigger. Small, explicit tools are easier to trust than one large social-media automation function.

