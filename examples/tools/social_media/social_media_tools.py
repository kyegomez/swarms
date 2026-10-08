"""
Social Media Marketing Tools for the Swarms Framework
======================================================

A unified set of tools for automating content posting across major social
media platforms. Each tool exposes a single parameter (`content`) and reads
the credentials it needs from environment variables at call time, so the
same Python process can switch accounts/keys without restart.

All tools return a JSON-encoded string with the shape:

    {"platform": "<name>", "status": "success" | "error", ...}

This makes them safe to plug directly into a Swarms `Agent` as tools and to
parse downstream by an orchestrator agent.

----------------------------------------------------------------------
Required environment variables per platform
----------------------------------------------------------------------

Twitter / X
    TWITTER_API_KEY
    TWITTER_API_SECRET
    TWITTER_ACCESS_TOKEN
    TWITTER_ACCESS_TOKEN_SECRET

LinkedIn
    LINKEDIN_ACCESS_TOKEN
    LINKEDIN_AUTHOR_URN          e.g. "urn:li:person:XXXXX" or "urn:li:organization:XXXXX"

YouTube  (video upload via Data API v3 with OAuth2 refresh token)
    YOUTUBE_CLIENT_ID
    YOUTUBE_CLIENT_SECRET
    YOUTUBE_REFRESH_TOKEN
    YOUTUBE_CATEGORY_ID          (optional, default "22" = People & Blogs)
    YOUTUBE_PRIVACY              (optional, default "public"; one of public/unlisted/private)

Telegram
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID             channel/group/user id (e.g. "@yourchannel" or "-1001234567890")

Discord
    DISCORD_WEBHOOK_URL          full webhook URL from a channel's Integrations settings

Dev.to
    DEVTO_API_KEY
    DEVTO_TITLE                  (optional, falls back to first line of content)
"""

import json
import os
from typing import Any, Dict

import requests


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _env(name: str) -> str:
    """Fetch a required environment variable. Raises if missing."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )
    return value


def _ok(platform: str, data: Dict[str, Any]) -> str:
    return json.dumps(
        {"platform": platform, "status": "success", "data": data}
    )


def _err(platform: str, message: str) -> str:
    return json.dumps(
        {"platform": platform, "status": "error", "error": message}
    )


def _first_line(text: str, max_len: int = 80) -> str:
    line = (
        text.strip().splitlines()[0] if text.strip() else "Untitled"
    )
    return line[:max_len]


# ---------------------------------------------------------------------------
# Twitter / X
# ---------------------------------------------------------------------------


def post_to_twitter(content: str) -> str:
    """
    Publish a tweet to Twitter/X using the v2 API with OAuth1 user context.

    Reads TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, and
    TWITTER_ACCESS_TOKEN_SECRET from the environment. Twitter enforces a
    280-character limit on standard accounts.

    Args:
        content (str): The full text of the tweet.

    Returns:
        str: JSON-encoded status payload.
    """
    try:
        from requests_oauthlib import OAuth1Session

        oauth = OAuth1Session(
            _env("TWITTER_API_KEY"),
            client_secret=_env("TWITTER_API_SECRET"),
            resource_owner_key=_env("TWITTER_ACCESS_TOKEN"),
            resource_owner_secret=_env("TWITTER_ACCESS_TOKEN_SECRET"),
        )
        response = oauth.post(
            "https://api.twitter.com/2/tweets",
            json={"text": content},
            timeout=15,
        )
        response.raise_for_status()
        return _ok("twitter", response.json())
    except Exception as e:
        return _err("twitter", str(e))


# ---------------------------------------------------------------------------
# LinkedIn
# ---------------------------------------------------------------------------


def post_to_linkedin(content: str) -> str:
    """
    Publish a UGC text post to LinkedIn as the configured author URN.

    Reads LINKEDIN_ACCESS_TOKEN and LINKEDIN_AUTHOR_URN from the environment.
    The author URN can be a person ("urn:li:person:...") or an organization
    page ("urn:li:organization:..."); the access token must hold the
    appropriate `w_member_social` or `w_organization_social` scope.

    Args:
        content (str): Post body. LinkedIn allows up to 3,000 characters.

    Returns:
        str: JSON-encoded status payload.
    """
    try:
        token = _env("LINKEDIN_ACCESS_TOKEN")
        author_urn = _env("LINKEDIN_AUTHOR_URN")

        body = {
            "author": author_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": content},
                    "shareMediaCategory": "NONE",
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
            },
        }
        response = requests.post(
            "https://api.linkedin.com/v2/ugcPosts",
            headers={
                "Authorization": f"Bearer {token}",
                "X-Restli-Protocol-Version": "2.0.0",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=15,
        )
        response.raise_for_status()
        return _ok(
            "linkedin",
            {"id": response.headers.get("x-restli-id", "")},
        )
    except Exception as e:
        return _err("linkedin", str(e))


# ---------------------------------------------------------------------------
# YouTube — community post (uses Studio internal API)
# ---------------------------------------------------------------------------


def post_to_youtube(
    title: str,
    description: str,
    tags: list,
    video_path: str,
) -> str:
    """
    Upload a video to YouTube via the Data API v3.

    Reads YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, and YOUTUBE_REFRESH_TOKEN
    from the environment to mint an OAuth2 access token. Optional env vars:
    YOUTUBE_CATEGORY_ID (default "22" = People & Blogs) and YOUTUBE_PRIVACY
    (default "public"; can also be "unlisted" or "private").

    Requires the `google-api-python-client` and `google-auth` packages:
        pip install google-api-python-client google-auth

    Args:
        title (str): Video title (max 100 characters).
        description (str): Video description (max 5,000 characters).
        tags (list): List of tag strings to associate with the video.
        video_path (str): Local filesystem path to the video file to upload.

    Returns:
        str: JSON-encoded status payload including the new video ID on success.
    """
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        creds = Credentials(
            token=None,
            refresh_token=_env("YOUTUBE_REFRESH_TOKEN"),
            client_id=_env("YOUTUBE_CLIENT_ID"),
            client_secret=_env("YOUTUBE_CLIENT_SECRET"),
            token_uri="https://oauth2.googleapis.com/token",
        )
        youtube = build("youtube", "v3", credentials=creds)

        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": list(tags) if tags else [],
                "categoryId": os.getenv("YOUTUBE_CATEGORY_ID", "22"),
            },
            "status": {
                "privacyStatus": os.getenv(
                    "YOUTUBE_PRIVACY", "public"
                ),
                "selfDeclaredMadeForKids": False,
            },
        }
        media = MediaFileUpload(
            video_path, chunksize=-1, resumable=True
        )
        response = (
            youtube.videos()
            .insert(
                part="snippet,status", body=body, media_body=media
            )
            .execute()
        )
        return _ok("youtube", response)
    except Exception as e:
        return _err("youtube", str(e))


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


def post_to_telegram(content: str) -> str:
    """
    Send a message to a Telegram channel, group, or user via the Bot API.

    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from the environment.
    The bot must already be an admin of the target channel/group, or the
    target user must have started a conversation with the bot. Telegram
    enforces a 4096-character limit per message.

    Args:
        content (str): The body of the message. Markdown formatting is
            enabled (parse_mode="Markdown").

    Returns:
        str: JSON-encoded status payload.
    """
    try:
        token = _env("TELEGRAM_BOT_TOKEN")
        chat_id = _env("TELEGRAM_CHAT_ID")
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": content,
                "parse_mode": "Markdown",
                "disable_web_page_preview": False,
            },
            timeout=15,
        )
        response.raise_for_status()
        return _ok("telegram", response.json())
    except Exception as e:
        return _err("telegram", str(e))


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------


def post_to_discord(content: str) -> str:
    """
    Send a message to a Discord channel via an incoming webhook.

    Reads DISCORD_WEBHOOK_URL from the environment. Webhooks are the
    simplest path for one-way posting and require no bot/OAuth flow — just
    create one inside the channel's Integrations settings. Discord enforces
    a 2000-character limit per webhook message.

    Args:
        content (str): The body of the message.

    Returns:
        str: JSON-encoded status payload.
    """
    try:
        webhook_url = _env("DISCORD_WEBHOOK_URL")
        response = requests.post(
            webhook_url,
            json={"content": content},
            params={"wait": "true"},
            timeout=15,
        )
        response.raise_for_status()
        return _ok("discord", response.json())
    except Exception as e:
        return _err("discord", str(e))


# ---------------------------------------------------------------------------
# Dev.to  (Forem)
# ---------------------------------------------------------------------------


def post_to_devto(content: str) -> str:
    """
    Publish an article to Dev.to via the Forem API.

    Reads DEVTO_API_KEY and (optionally) DEVTO_TITLE from the environment.
    If DEVTO_TITLE is not set, the first line of `content` is used as the
    title.

    Args:
        content (str): Markdown body of the article.

    Returns:
        str: JSON-encoded status payload.
    """
    try:
        api_key = _env("DEVTO_API_KEY")
        title = os.getenv("DEVTO_TITLE") or _first_line(content)
        response = requests.post(
            "https://dev.to/api/articles",
            headers={
                "api-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "article": {
                    "title": title,
                    "body_markdown": content,
                    "published": True,
                }
            },
            timeout=20,
        )
        response.raise_for_status()
        return _ok("devto", response.json())
    except Exception as e:
        return _err("devto", str(e))


# ---------------------------------------------------------------------------
# Aggregate fan-out
# ---------------------------------------------------------------------------


# Text-only platforms — share the single-`content` signature and are safe
# to fan out from one input.
TEXT_POSTERS = {
    "twitter": post_to_twitter,
    "linkedin": post_to_linkedin,
    "telegram": post_to_telegram,
    "discord": post_to_discord,
    "devto": post_to_devto,
}

# Full toolkit exposed to the agent. YouTube is included here even though
# it has a different signature (title/description/tags/video_path), since
# the agent calls each tool individually with the right arguments.
ALL_POSTERS = {
    **TEXT_POSTERS,
    "youtube": post_to_youtube,
}


def post_everywhere(content: str) -> str:
    """
    Fan the same `content` out to every text-based platform.

    YouTube is excluded because it requires a video file and structured
    metadata; the agent should call `post_to_youtube` directly. Each
    platform is attempted independently — a failure on one does not abort
    the rest.

    Args:
        content (str): The post content to publish across all platforms.

    Returns:
        str: JSON-encoded mapping of platform -> response payload.
    """
    results: Dict[str, Any] = {}
    for name, fn in TEXT_POSTERS.items():
        try:
            results[name] = json.loads(fn(content))
        except Exception as e:
            results[name] = {
                "platform": name,
                "status": "error",
                "error": str(e),
            }
    return json.dumps(results)


# ---------------------------------------------------------------------------
# Example: a Swarms agent wired up to all the posting tools
# ---------------------------------------------------------------------------

# all_tools = list(ALL_POSTERS.values())

# if __name__ == "__main__":
#     from swarms import Agent

#     agent = Agent(
#         agent_name="Marketing-Distribution-Agent",
#         agent_description=(
#             "Distributes marketing content to all configured social "
#             "media channels using the swarms social_media_tools."
#         ),
#         system_prompt=(
#             "You are a marketing distribution agent. Given a piece of "
#             "content, decide which platforms it should be published to "
#             "and call the corresponding tools. Most tools take a single "
#             "string argument (the post content); `post_to_youtube` is "
#             "different — it takes title, description, tags, and a video "
#             "file path, which you should generate from the source brief. "
#             "Adapt voice and length per platform when appropriate, but "
#             "never invent facts."
#         ),
#         model_name="claude-sonnet-4-5",
#         max_loops=1,
#         tool_call_summary=True,
#         output_type="final",
#         tools=list(ALL_POSTERS.values()),
#     )

#     out = agent.run(
#         "Announce: Swarms 1.0 is live. New multi-agent orchestration, "
#         "MCP support, and a managed cloud API. Post this everywhere."
#     )
#     print(out)
