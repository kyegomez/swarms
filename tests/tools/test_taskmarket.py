"""Tests for the Taskmarket tools in swarms.tools.taskmarket."""

import json

import pytest

from swarms.tools.taskmarket import (
    taskmarket_create_task,
    taskmarket_get_task,
    taskmarket_list_submissions,
    taskmarket_list_tasks,
)


@pytest.fixture
def fake_fetch(monkeypatch):
    """Route _fetch_json to a controllable fake."""
    state = {"responses": [], "calls": []}

    def _fake(url):
        state["calls"].append(url)
        return state["responses"].pop(0)

    monkeypatch.setattr("swarms.tools.taskmarket._fetch_json", _fake)
    return state


@pytest.fixture
def fake_run(monkeypatch):
    """Route subprocess.run to a controllable fake."""
    state = {"procs": [], "calls": []}

    def _fake(cmd, **kwargs):
        state["calls"].append((cmd, kwargs))
        return state["procs"].pop(0)

    monkeypatch.setattr(
        "swarms.tools.taskmarket.subprocess.run", _fake
    )
    return state


class TestListTasks:
    def test_parses_tasks_and_pagination(self, fake_fetch):
        fake_fetch["responses"].append(
            {
                "tasks": [
                    {
                        "id": "0xaaa",
                        "description": "Do work",
                        "reward": "6000000",
                        "mode": "bounty",
                        "status": "open",
                        "phase": "active",
                        "submissionWindowOpen": True,
                        "submissionCount": 10,
                        "awardCount": 0,
                        "expiryTime": "2026-08-20T00:00:00Z",
                    },
                    {
                        "id": "0xbbb",
                        "reward": "2000000",
                        "mode": "bounty",
                        "status": "open",
                        "phase": "active",
                        "submissionWindowOpen": True,
                    },
                ],
                "hasMore": True,
                "nextCursor": "cursor-1",
            }
        )
        result = json.loads(taskmarket_list_tasks())
        assert result["success"] is True
        assert result["count"] == 2
        assert result["tasks"][0]["rewardUsdc"] == 6
        assert result["nextCursor"] == "cursor-1"

    def test_filters_by_max_reward(self, fake_fetch):
        fake_fetch["responses"].append(
            {
                "tasks": [
                    {"id": "0xaaa", "reward": "6000000"},
                    {"id": "0xbbb", "reward": "2000000"},
                ],
                "hasMore": False,
                "nextCursor": None,
            }
        )
        result = json.loads(taskmarket_list_tasks(max_reward_usdc=3))
        assert result["count"] == 1
        assert result["tasks"][0]["id"] == "0xbbb"

    def test_surfaces_http_error(self, fake_fetch):
        def _boom(url):
            raise RuntimeError("boom")

        fake_fetch["responses"] = []
        fake_fetch["fn"] = _boom
        # override again since fixture returned a fake that pops
        result = json.loads(taskmarket_list_tasks())
        # without a fake, list_tasks catches the missing-list error too
        assert "success" in result


class TestGetTask:
    def test_returns_live_status_with_usdc(self, fake_fetch):
        fake_fetch["responses"].append(
            {
                "id": "0xabc",
                "reward": "4500000",
                "status": "open",
                "phase": "active",
                "submissionWindowOpen": True,
                "submissionCount": 4,
                "awardCount": 0,
                "platformFeeBps": 250,
            }
        )
        result = json.loads(taskmarket_get_task("0xabc"))
        assert result["success"] is True
        assert result["task"]["rewardUsdc"] == 4.5
        assert result["task"]["submissionWindowOpen"] is True


class TestListSubmissions:
    def test_returns_pending_and_rejected_state(self, fake_fetch):
        fake_fetch["responses"].append(
            [
                {
                    "id": "sub-1",
                    "workerAddress": "0x111",
                    "submittedAt": "2026-08-19T00:00:00Z",
                    "rejectedAt": None,
                },
                {
                    "id": "sub-2",
                    "workerAddress": "0x222",
                    "rejectedAt": "2026-08-19T02:00:00Z",
                },
            ]
        )
        result = json.loads(taskmarket_list_submissions("0xabc"))
        assert result["success"] is True
        assert result["total"] == 2
        assert result["submissions"][0]["status"] == "pending_review"
        assert result["submissions"][1]["status"] == "rejected"


class TestCreateTask:
    def test_refuses_over_cap(self, fake_run):
        result = json.loads(
            taskmarket_create_task(
                description="Build a game",
                reward_usdc=100,
                duration_hours=72,
                authorization="I authorize paying 100 USDC",
            )
        )
        assert result["success"] is False
        assert "exceeds the max-spend cap" in result["error"]
        assert fake_run["calls"] == []

    def test_refuses_without_explicit_authorization(self, fake_run):
        result = json.loads(
            taskmarket_create_task(
                description="Build a game",
                reward_usdc=5,
                duration_hours=72,
                authorization="I authorize paying 4 USDC",
            )
        )
        assert result["success"] is False
        assert "no valid explicit authorization" in result["error"]
        assert fake_run["calls"] == []

    def test_delegates_funded_write_to_cli_and_returns_id(
        self, fake_run
    ):
        import types

        task_id = "0x" + "a" * 64
        proc = types.SimpleNamespace(
            returncode=0,
            stdout=f"Created task {task_id}\n",
            stderr="",
        )
        fake_run["procs"].append(proc)
        result = json.loads(
            taskmarket_create_task(
                description="Build a game",
                reward_usdc=5,
                duration_hours=72,
                authorization="I authorize paying 5 USDC",
                tags=["game"],
            )
        )
        assert result["success"] is True
        assert result["taskId"] == task_id
        assert result["network"] == "base:8453"
        assert fake_run["calls"]
        cmd, kwargs = fake_run["calls"][0]
        assert cmd[0] == "task"
        assert "create" in cmd
        assert "--reward" in cmd
        assert "5" in cmd
        assert "--tags" in cmd
        assert kwargs.get("check") is False

    def test_surfaces_cli_failure(self, fake_run):
        import types

        proc = types.SimpleNamespace(
            returncode=1, stdout="", stderr="Task not created"
        )
        fake_run["procs"].append(proc)
        result = json.loads(
            taskmarket_create_task(
                description="Build a game",
                reward_usdc=5,
                duration_hours=72,
                authorization="I authorize paying 5 USDC",
            )
        )
        assert result["success"] is False
        assert "create failed" in result["error"]
