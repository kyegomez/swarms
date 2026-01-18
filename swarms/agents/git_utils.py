from typing import Tuple, Optional


class GitClient:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def fetch(self) -> Tuple[int, str]:
        return 0, ""

    def rebase(self, branch: str) -> Tuple[int, str]:
        return 0, ""

    def create_branch(self, branch: str) -> Tuple[int, str]:
        return 0, ""

    def commit_and_push(self, branch: str, message: str, changes_dir: Optional[str] = None) -> Tuple[bool, dict]:
        # Minimal stub: treat as successful no-op push for prototype/tests
        return True, {"out": "noop"}

    def reset_to_remote_base(self, base_branch: str = "master") -> Tuple[bool, str]:
        return False, "not_implemented"
