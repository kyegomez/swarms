"""
Fetch commit titles and descriptions from any GitHub repository for a given time frame.
"""

from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlparse

import requests

# Date string formats accepted for start_date / end_date
_DATE_FORMATS = (
    "%Y-%m-%d",  # 2025-01-15
    "%m/%d/%Y",  # 01/15/2025
    "%b %d, %Y",  # Jan 15, 2025
    "%B %d, %Y",  # January 15, 2025
    "%b %d %Y",  # Jan 15 2025
    "%B %d %Y",  # January 15 2025
    "%d %b %Y",  # 15 Jan 2025
    "%d %B %Y",  # 15 January 2025
)


def _parse_date(value: str | datetime) -> datetime:
    """Parse start_date/end_date from string (YYYY-MM-DD, Jan 15 2025, etc.) or datetime."""
    if isinstance(value, datetime):
        return (
            value.replace(tzinfo=value.tzinfo)
            if value.tzinfo
            else value
        )
    s = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Unrecognized date: {value!r}. Use YYYY-MM-DD or e.g. Jan 15, 2025"
    )


def parse_github_repo_url(repo_url: str) -> str | None:
    """
    Extract owner/repo from a GitHub URL.

    Supports:
    - https://github.com/owner/repo
    - https://github.com/owner/repo/
    - git@github.com:owner/repo.git
    - github.com/owner/repo
    """
    repo_url = repo_url.strip().rstrip("/")
    # Handle git SSH format
    if repo_url.startswith("git@github.com:"):
        path = repo_url.replace("git@github.com:", "")
        path = path.removesuffix(".git")
        parts = path.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    # Handle HTTPS or plain domain
    if "github.com" in repo_url:
        parsed = urlparse(
            repo_url if "://" in repo_url else "https://" + repo_url
        )
        path = parsed.path.strip("/")
        parts = path.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return None


def fetch_github_commits(
    repo_url: str,
    *,
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
    days: int | None = None,
    months: int | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch commit titles and descriptions for a GitHub repo in a date range.

    Args:
        repo_url: GitHub repository URL (e.g. https://github.com/owner/repo)
        start_date: Start of range. String (e.g. "2025-01-15", "Jan 15, 2025") or datetime.
        end_date: End of range. String or datetime. Defaults to now if start_date is set.
        days: Alternative: include commits from the past N days.
        months: Alternative: include commits from the past N months.

    Use either (start_date, end_date) or (days or months). Examples:

        fetch_github_commits(url, start_date="2025-01-15", end_date="2025-02-02")
        fetch_github_commits(url, start_date="January 15, 2025")   # end = now
        fetch_github_commits(url, days=30)

    Returns:
        List of dicts with keys: sha, title, description, date, author_login, url.
    """
    if start_date is not None:
        since_dt = _parse_date(start_date)
        end_dt = (
            _parse_date(end_date)
            if end_date is not None
            else datetime.utcnow()
        )
        if end_dt < since_dt:
            raise ValueError(
                "end_date must be on or after start_date"
            )
    elif days is not None or months is not None:
        since_dt = datetime.utcnow()
        if months:
            since_dt -= timedelta(days=months * 30)
        if days:
            since_dt -= timedelta(days=days)
        end_dt = datetime.utcnow()
    else:
        raise ValueError(
            "Provide start_date (and optional end_date) or days/months"
        )

    repo = parse_github_repo_url(repo_url)
    if not repo:
        raise ValueError(f"Invalid GitHub repo URL: {repo_url}")

    since_iso = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    api_url = f"https://api.github.com/repos/{repo}/commits"
    headers = {"Accept": "application/vnd.github.v3+json"}
    results: list[dict[str, Any]] = []
    page = 1
    per_page = 100

    while True:
        params: dict[str, str | int] = {
            "per_page": per_page,
            "page": page,
            "since": since_iso,
        }
        resp = requests.get(
            api_url, headers=headers, params=params, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        past_start = False
        for c in data:
            commit = c.get("commit") or {}
            author = commit.get("author") or {}
            commit_date_str = author.get("date")
            if commit_date_str and commit_date_str > end_iso:
                continue
            if commit_date_str and commit_date_str < since_iso:
                past_start = True
                continue
            msg = (commit.get("message") or "").strip()
            lines = msg.split("\n")
            title = lines[0] if lines else ""
            description = (
                "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
            )
            author_info = c.get("author") or {}
            results.append(
                {
                    "sha": c.get("sha", "")[:7],
                    "title": title,
                    "description": description,
                    "date": commit_date_str,
                    "author_login": author_info.get("login"),
                    "url": c.get("html_url"),
                }
            )

        if past_start or len(data) < per_page:
            break
        page += 1

    return results


commits = fetch_github_commits(
    "https://github.com/kyegomez/swarms",
    start_date="2025-01-15",
    end_date="2025-02-02",
)

print(commits)
