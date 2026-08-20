"""
Taskmarket tools for Swarms agents.

Taskmarket is an onchain task marketplace where requesters escrow USDC
and workers earn payouts for accepted work (base L2). These plain
functions expose Taskmarket as agent-callable tools, following Swarms'
"functions as tools" convention: each function's signature and docstring
are converted into a function-calling schema automatically.

Reads (list/get/submissions) hit the public Taskmarket REST API and need
no wallet or secrets. The funded requester write (`taskmarket_create_task`)
is gated by explicit user authorization and a hard max-spend cap, and is
delegated to the first-party ``taskmarket`` CLI so wallet signing, X402
payment, legal acceptance and idempotency are handled by official tooling.
"""

import json
import os
import subprocess
import urllib.request

TASKMARKET_API_URL = os.environ.get(
    "TASKMARKET_API_URL", "https://api.taskmarket.dev"
)
TASKMARKET_API_BASE = f"{TASKMARKET_API_URL}/api"
USDC_DECIMALS = 6
BASE_CHAIN_ID = 8453
TASKMARKET_APP_URL = "https://taskmarket.dev"
DEFAULT_MAX_TASK_SPEND_USDC = float(
    os.environ.get("TASKMARKET_MAX_TASK_SPEND_USDC", "25")
)
TASKMARKET_CLI = os.environ.get("TASKMARKET_CLI", "taskmarket")
TASKMARKET_CLI_TIMEOUT_MS = int(
    os.environ.get("TASKMARKET_CLI_TIMEOUT_MS", "180000")
)
TASKMARKET_TASK_ID_RE = __import__("re").compile(r"0x[0-9a-fA-F]{64}")


def _fetch_json(url: str) -> dict:
    """GET a URL and return the parsed JSON body."""
    with urllib.request.urlopen(url, timeout=45) as resp:
        return json.load(resp)


def _from_base_units(base_units) -> float:
    """Convert a base-unit reward string to whole USDC."""
    return float(base_units) / (10**USDC_DECIMALS)


def taskmarket_list_tasks(
    mode: str | None = None,
    max_reward_usdc: float | None = None,
    search: str | None = None,
    cursor: str | None = None,
) -> str:
    """List open (submittable) tasks on the Taskmarket marketplace.

    Use this to discover work an agent could delegate to external
    Taskmarket workers. Read-only; no wallet or secrets required.

    Args:
        mode (Optional[str]): Optional task mode: bounty, claim, pitch,
            benchmark, or auction.
        max_reward_usdc (Optional[float]): Only return tasks with a
            reward at or below this many whole USDC.
        search (Optional[str]): Only return tasks whose description
            contains this substring.
        cursor (Optional[str]): Pagination cursor from a previous call
            to continue paging.

    Returns:
        str: A JSON string with a list of tasks and a pagination cursor.
        Items include id, description, rewardUsdc, mode, status, phase,
        submissionWindowOpen, submissionCount, awardCount, expiryTime.
    """
    try:
        params = []
        if mode:
            params.append(f"mode={mode}")
        if search:
            params.append(f"search={search}")
        if cursor:
            params.append(f"cursor={cursor}")
        query = "&".join(params)
        url = f"{TASKMARKET_API_BASE}/tasks"
        if query:
            url = f"{url}?{query}"
        data = _fetch_json(url)
        tasks = data.get("tasks", [])
        rows = []
        for t in tasks:
            row = {
                "id": t.get("id"),
                "description": str(t.get("description", ""))[:200],
                "rewardUsdc": _from_base_units(t.get("reward", "0")),
                "mode": t.get("mode"),
                "status": t.get("status"),
                "phase": t.get("phase"),
                "submissionWindowOpen": t.get("submissionWindowOpen"),
                "submissionCount": t.get("submissionCount"),
                "awardCount": t.get("awardCount"),
                "expiryTime": t.get("expiryTime"),
            }
            if (
                max_reward_usdc is None
                or row["rewardUsdc"] <= max_reward_usdc
            ):
                rows.append(row)
        return json.dumps(
            {
                "success": True,
                "count": len(rows),
                "tasks": rows,
                "hasMore": data.get("hasMore"),
                "nextCursor": data.get("nextCursor"),
            }
        )
    except Exception as e:  # noqa: BLE001
        return json.dumps(
            {"success": False, "error": f"list failed: {e}"}
        )


def taskmarket_get_task(task_id: str) -> str:
    """Fetch the live status of a single Taskmarket task.

    Use this to track a task the agent posted or is monitoring. Read-only.

    Args:
        task_id (str): The 0x-prefixed Taskmarket task id.

    Returns:
        str: A JSON string with the task's live status: status, phase,
        mode, rewardUsdc, submissionWindowOpen, submissionCount,
        awardCount, expiryTime, visibility, and description.
    """
    try:
        url = f"{TASKMARKET_API_BASE}/tasks/{task_id}"
        t = _fetch_json(url)
        return json.dumps(
            {
                "success": True,
                "task": {
                    "id": t.get("id"),
                    "description": str(t.get("description", ""))[
                        :500
                    ],
                    "rewardUsdc": _from_base_units(
                        t.get("reward", "0")
                    ),
                    "mode": t.get("mode"),
                    "status": t.get("status"),
                    "phase": t.get("phase"),
                    "submissionWindowOpen": t.get(
                        "submissionWindowOpen"
                    ),
                    "submissionCount": t.get("submissionCount"),
                    "awardCount": t.get("awardCount"),
                    "expiryTime": t.get("expiryTime"),
                    "taskVisibility": t.get("taskVisibility"),
                    "submissionVisibility": t.get(
                        "submissionVisibility"
                    ),
                    "platformFeeBps": t.get("platformFeeBps"),
                },
            }
        )
    except Exception as e:  # noqa: BLE001
        return json.dumps(
            {"success": False, "error": f"get failed: {e}"}
        )


def taskmarket_list_submissions(
    task_id: str, max_results: int = 50
) -> str:
    """List the submissions of a Taskmarket task for human review.

    Presents candidates so a human can decide. This tool NEVER accepts
    or rejects work; accept/reject is an explicit human step via the
    first-party CLI.

    Args:
        task_id (str): The 0x-prefixed Taskmarket task id.
        max_results (int): Maximum number of submissions to return.
            (default: 50)

    Returns:
        str: A JSON string of submissions with id, workerAddress,
        submittedAt, rejectedAt, workerAgentId, deliverableHash, and a
        derived status (pending_review or rejected).
    """
    try:
        limit = min(max_results, 200)
        url = f"{TASKMARKET_API_BASE}/tasks/{task_id}/submissions"
        subs = _fetch_json(url)
        if not isinstance(subs, list):
            subs = subs.get("submissions", [])
        rows = []
        for s in subs[:limit]:
            rows.append(
                {
                    "id": s.get("id"),
                    "workerAddress": s.get("workerAddress"),
                    "submittedAt": s.get("submittedAt"),
                    "rejectedAt": s.get("rejectedAt"),
                    "workerAgentId": s.get("workerAgentId"),
                    "deliverableHash": s.get("deliverableHash"),
                    "status": (
                        "rejected"
                        if s.get("rejectedAt")
                        else "pending_review"
                    ),
                }
            )
        return json.dumps(
            {
                "success": True,
                "count": len(rows),
                "total": len(subs),
                "submissions": rows,
            }
        )
    except Exception as e:  # noqa: BLE001
        return json.dumps(
            {
                "success": False,
                "error": f"list submissions failed: {e}",
            }
        )


def taskmarket_create_task(
    description: str,
    reward_usdc: float,
    duration_hours: float,
    authorization: str,
    max_spend_usdc: float | None = None,
    mode: str = "bounty",
    tags: list[str] | None = None,
) -> str:
    """Create and fund a Taskmarket task as a requester.

    This is a funded onchain write on Base L2. It requires explicit,
    fresh user authorization and enforces a hard max-spend cap. The
    funded write is delegated to the first-party ``taskmarket`` CLI so
    no private key ever touches the agent.

    Args:
        description (str): The task description shown to workers.
        reward_usdc (float): Reward in whole USDC (e.g. 5 for 5 USDC).
        duration_hours (float): How long the task stays open, in hours.
        authorization (str): REQUIRED explicit user authorization. Must
            contain the exact phrase "I authorize paying <total> USDC"
            where <total> is the total cost (the reward). The tool
            refuses to create without it.
        max_spend_usdc (Optional[float]): Hard cap on total spend in
            USDC. Defaults to the provider limit.
        mode (str): Task mode (default "bounty").
        tags (Optional[List[str]]): Optional up to 10 tags.

    Returns:
        str: A JSON string with the created task id and link, or a
        refusal reason. On an ambiguous/in-flight result it reports the
        situation and instructs polling rather than resubmitting.
    """
    try:
        total_usdc = reward_usdc
        cap = max_spend_usdc or DEFAULT_MAX_TASK_SPEND_USDC
        if total_usdc > cap:
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        f"refused: reward {total_usdc} USDC exceeds the "
                        f"max-spend cap of {cap} USDC"
                    ),
                }
            )
        expected = f"I authorize paying {total_usdc} USDC"
        if expected not in authorization:
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "refused: no valid explicit authorization; the "
                        f'phrase "{expected}" is required'
                    ),
                }
            )
        cli_args = [
            "task",
            "create",
            "--description",
            description,
            "--reward",
            str(reward_usdc),
            "--duration",
            str(duration_hours),
            "--mode",
            mode,
        ]
        if tags:
            cli_args.extend(["--tags", ",".join(tags)])
        proc = subprocess.run(
            cli_args,
            capture_output=True,
            text=True,
            timeout=TASKMARKET_CLI_TIMEOUT_MS / 1000,
            check=False,
        )
        output = (proc.stdout + "\n" + proc.stderr).strip()
        match = TASKMARKET_TASK_ID_RE.search(output)
        if proc.returncode == 0 and match:
            task_id = match.group(0)
            return json.dumps(
                {
                    "success": True,
                    "taskId": task_id,
                    "taskUrl": f"{TASKMARKET_APP_URL}/tasks/{task_id}",
                    "network": "base:8453",
                    "totalUsdc": total_usdc,
                    "note": (
                        "track live status with taskmarket_get_task; if "
                        "in-flight/ambiguous, poll, do not resubmit"
                    ),
                }
            )
        if proc.returncode == 0:
            return json.dumps(
                {
                    "success": True,
                    "rawOutput": output[:2000],
                    "note": (
                        "CLI reported success; retrieve the task id (do "
                        "not resubmit)"
                    ),
                }
            )
        return json.dumps(
            {
                "success": False,
                "error": f"create failed: {output[:2000]}",
                "note": (
                    "if in-flight/ambiguous, poll the prior task by id; "
                    "do not resubmit"
                ),
            }
        )
    except Exception as e:  # noqa: BLE001
        return json.dumps(
            {"success": False, "error": f"create failed: {e}"}
        )
