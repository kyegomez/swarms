import subprocess
from typing import Tuple, Optional


class GitClient:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def _run(self, args, capture_stderr=True) -> Tuple[int, str, str]:
        p = subprocess.Popen(
            ["git"] + args,
            cwd=self.repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out, err = p.communicate()
        return p.returncode, out, err

    def fetch(self) -> Tuple[int, str]:
        code, out, err = self._run(["fetch", "--all"])
        return code, out + err

    def rebase(self, branch: str) -> Tuple[int, str]:
        # try to rebase branch onto remote
        code, out, err = self._run(["pull", "--rebase", "origin", branch])
        return code, out + err

    def create_branch(self, branch: str) -> Tuple[int, str]:
        code, out, err = self._run(["checkout", "-B", branch])
        return code, out + err

    def commit_and_push(self, branch: str, message: str, changes_dir: Optional[str] = None) -> Tuple[bool, dict]:
        try:
            # ensure on branch
            code, _ = self.create_branch(branch)
            if code != 0:
                return False, {"error": "checkout_failed"}

            if changes_dir:
                # add changes from directory
                rc, out, err = self._run(["add", "-A", changes_dir])
            else:
                rc, out, err = self._run(["add", "-A"])

            if rc != 0:
                return False, {"error": "git_add_failed", "out": out, "err": err}

            rc, out, err = self._run(["commit", "-m", message])
            if rc != 0:
                # nothing to commit is not fatal, treat as success
                if "nothing to commit" in out.lower() + err.lower():
                    pass
                else:
                    return False, {"error": "git_commit_failed", "out": out, "err": err}

            rc, out, err = self._run(["push", "-u", "origin", branch])
            if rc == 0:
                return True, {"out": out}
            else:
                # detect non-fast-forward / conflict
                combined = out + err
                if "non-fast-forward" in combined or "failed to push" in combined or "rejected" in combined:
                    return False, {"conflict": True, "out": combined}
                return False, {"error": "push_failed", "out": combined}
        except Exception as e:
            return False, {"error": "exception", "msg": str(e)}

    def reset_to_remote_base(self, base_branch: str = "master") -> Tuple[bool, str]:
        """Fetch and hard-reset local repo to origin/<base_branch>."""
        try:
            rc, out = self.fetch()
            if rc != 0:
                return False, out
            code, out, err = self._run(["reset", "--hard", f"origin/{base_branch}"])
            combined = out + err
            if code == 0:
                # optional: clean untracked files
                self._run(["clean", "-fd"])
                return True, combined
            return False, combined
        except Exception as e:
            return False, str(e)
