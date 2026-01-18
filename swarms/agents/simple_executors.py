def noop_executor(payload, repo_path):
    # simple executor used by process-based worker pool in tests
    return {"branch": payload.get("branch", "feature/noop"), "commit_message": payload.get("message", "noop")}
