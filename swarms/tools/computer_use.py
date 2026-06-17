"""Security-hardened computer-use toolkit: read_file, list_directory, grep_files, write_file, edit_file, patch_file, delete_file, run_command."""

from __future__ import annotations

import dataclasses, fnmatch, hashlib, hmac, os, re, secrets, shutil, signal, subprocess, sys, time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union


class SecurityError(Exception): pass
class PathPolicyError(SecurityError): pass
class SymlinkPolicyError(SecurityError): pass
class InvalidInputError(SecurityError): pass
class BinaryNotAllowedError(SecurityError): pass
class ArgvPatternDeniedError(SecurityError): pass
class ConfirmationRequired(SecurityError):
    def __init__(self, tool: str, request_id: Optional[str] = None) -> None:
        super().__init__(f"Confirmation required for {tool!r}; pass confirm=True.")
        self.tool = tool
        self.request_id = request_id
        self.tool = tool
        self.request_id = request_id


_GLOB_SAFE = re.compile(r"^[A-Za-z0-9_./?!\*\-]+$")
_PATH_DENY_DEFAULT = frozenset(
    {
        "/etc",
        "/root",
        "/proc",
        "/sys",
        "/var/lib",
        "/boot",
        "/usr/lib",
        "/usr/lib64",
    }
)
@dataclasses.dataclass(frozen=True)
class ReadPolicy:
    """Policy for read-only filesystem tools."""
    require_cwd_under: Optional[str] = None
    deny_paths: frozenset[str] = _PATH_DENY_DEFAULT
    follow_symlinks: bool = False
    max_file_size_bytes: int = 10 * 1024 * 1024
    max_output_bytes: int = 1 * 1024 * 1024
    max_glob_tokens: int = 64  # raised to keep tests honest, not user-facing
    rg_timeout_seconds: float = 10.0
@dataclasses.dataclass(frozen=True)
class WritePolicy:
    """Policy for write/edit/delete operations."""
    require_cwd_under: Optional[str] = None
    require_confirm: bool = True
    mode_default: Literal["fail", "overwrite", "append"] = "fail"
    atomic: bool = True
    max_file_size_bytes: int = 10 * 1024 * 1024
    backup_on_overwrite: bool = True
    backup_dir: str = ".swarm_backups"
    follow_symlinks: Literal["reject", "allow"] = "reject"
    max_content_bytes: int = 1 * 1024 * 1024  # for edit_file old/new

@dataclasses.dataclass(frozen=True)
class ShellPolicy:
    """Policy for shell execution."""
    binary_allowlist: frozenset[str] = frozenset(
        {
            "ls", "cat", "grep", "rg", "find", "head", "tail", "wc", "sort", "sleep",
            "uniq", "tr", "cut", "paste", "sed", "awk", "git", "npm", "node",
            "python", "python3", "pip", "pip3", "pytest", "cargo", "rustc",
            "go", "make", "cmake", "gcc", "clang", "curl", "wget", "jq",
            "tar", "gzip", "gunzip", "zip", "unzip", "ssh", "scp", "rsync",
            "ps", "top", "htop", "df", "du", "free", "uname", "whoami",
            "id", "env", "printenv", "date", "cal", "bc", "diff", "patch",
        }
    )
    argv_substring_denylist: frozenset[str] = frozenset(
        {
            "rm -rf", "mkfs", "dd if=", ">/dev/sd", "chmod -R 777",
            ":(){:|:", "curl ", "| sh", "wget ", "| sh", ">/etc/",
            "/etc/passwd", "/etc/shadow", "/root/.ssh", "ssh-keygen",
            "iptables", "ufw ", "systemctl ", "service ",
        }
    )
    cwd_under: str = "/workspace"
    cwd_extra: frozenset[str] = frozenset({"/tmp", "/home"})
    timeout_default: int = 30
    timeout_max: int = 600
    max_stdout_bytes: int = 1 * 1024 * 1024
    max_stderr_bytes: int = 1 * 1024 * 1024
    max_stdin_bytes: int = 64 * 1024
    max_argv_tokens: int = 256
    max_arg_token_bytes: int = 4096
    kill_process_group_on_timeout: bool = True
    redact_stdin_in_logs: bool = True
    blocked_env_keys: frozenset[str] = frozenset(
        {
            "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID",
            "GITHUB_TOKEN", "GH_TOKEN", "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
            "HUGGINGFACE_TOKEN", "HF_TOKEN", "AZURE_CLIENT_SECRET",
            "STRIPE_SECRET_KEY", "SLACK_TOKEN", "TELEGRAM_BOT_TOKEN",
            "DISCORD_TOKEN", "GITLAB_TOKEN", "BITBUCKET_TOKEN",
            "SENDGRID_API_KEY", "TWILIO_AUTH_TOKEN", "MAILGUN_API_KEY",
            "DATADOG_API_KEY", "NEW_RELIC_LICENSE_KEY",
            "OPENAI_ORGANIZATION", "COHERE_API_KEY",
            "REPLICATE_API_TOKEN",
        }
    )
# Path / input helpers (shared by every tool)
def _reject_nul(value: Any, *, arg_name: str) -> str:
    """Reject strings containing NUL bytes."""
    if not isinstance(value, str):
        raise InvalidInputError(f"{arg_name} must be a string")
    if "\x00" in value:
        raise InvalidInputError(f"{arg_name} must not contain NUL bytes")
    return value
def _validate_glob(glob: str) -> str:
    if glob.startswith("-"):
        raise InvalidInputError(
            f"glob {glob!r} starts with '-' (flag injection risk)"
        )
    if "\x00" in glob:
        raise InvalidInputError("glob must not contain NUL bytes")
    if not _GLOB_SAFE.match(glob):
        raise InvalidInputError(
            f"glob {glob!r} contains disallowed characters; "
            "allowed: alphanumerics, _ . / * ? ! [ ] -"
        )
    return glob

def _canonical(path: Union[str, os.PathLike]) -> Path:
    return Path(os.path.realpath(os.fspath(path)))
def _deny_match(real: Path, deny: Iterable[str]) -> bool:
    for raw in deny:
        if not raw:
            continue
        try:
            denied = Path(os.path.realpath(raw))
        except OSError:
            continue
        try:
            if real == denied or denied in real.parents:
                return True
        except OSError:
            continue
        if "*" in raw or "?" in raw or "[" in raw:
            try:
                if fnmatch.fnmatch(str(real), raw):
                    return True
            except re.error:
                continue
    return False

def _check_realpath(
    real: Path,
    *,
    policy_deny: Iterable[str],
    workspace: Optional[str],
) -> None:
    if _deny_match(real, policy_deny):
        raise PathPolicyError(f"path {real} is on the deny-list")
    if workspace is not None:
        ws = Path(os.path.realpath(workspace))
        try:
            if not (real == ws or ws in real.parents):
                raise PathPolicyError(
                    f"path {real} is outside workspace {ws}"
                )
        except OSError as exc:
            raise PathPolicyError(f"path {real} could not be resolved: {exc}") from exc

def _check_write_path(
    real: Path,
    *,
    policy: WritePolicy,
    original: Optional[Path] = None,
) -> None:
    if policy.follow_symlinks == "reject":
        # Walk the *unresolved* path's components — symlinks at the leaf or
        # any parent must be rejected. We also resolve after the check via
        # ``real`` for the deny-list / workspace comparison.
        leaf = original if original is not None else real
        scan: List[Path] = []
        cur: Optional[Path] = leaf if leaf.is_absolute() else Path.cwd() / leaf
        seen: set = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            scan.append(cur)
            cur = cur.parent
        for part in scan:
            try:
                if part.is_symlink():
                    raise SymlinkPolicyError(
                        f"path {leaf} traverses symlink at {part}"
                    )
            except OSError:
                continue
    _check_realpath(
        real,
        policy_deny=ReadPolicy().deny_paths,
        workspace=policy.require_cwd_under,
    )
def read_file(path: str, encoding: str = "utf-8", offset: int = 0, limit: Optional[int] = None, *, policy: Optional[ReadPolicy] = None, workspace_root: Optional[str] = None) -> str:
    if offset < 0:
        raise InvalidInputError("offset must be >= 0")
    if limit is not None and limit > 10_000_000:
        raise InvalidInputError("limit exceeds 10 MiB cap")
    p = _reject_nul(path, arg_name="path")
    real = _canonical(p)
    _check_realpath(real, policy_deny=(policy or ReadPolicy()).deny_paths, workspace=workspace_root)
    try:
        if real.is_dir():
            return f"Error: {real} is a directory; use list_directory"
    except OSError:
        pass

    try:
        data = real.read_text(encoding=encoding)
    except PermissionError as exc:
        raise PathPolicyError(
            f"permission denied reading {real}"
        ) from exc
    except FileNotFoundError as exc:
        raise PathPolicyError(f"file not found: {real}") from exc

    if offset:
        data = data[offset:]
    if limit is not None:
        data = data[:limit]
    if len(data.encode(encoding, errors="replace")) > eff.max_output_bytes:
        data = data[: eff.max_output_bytes]
        return data + "{...truncated, +N bytes...}"
    return data

def list_directory(
    path: str = ".",
    glob: str = "*",
    include_hidden: bool = False,
    *,
    policy: Optional[ReadPolicy] = None,
    workspace_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List a directory's entries, no file contents."""
    p = _reject_nul(path, arg_name="path")
    glob_v = _validate_glob(glob)
    real = _canonical(p)
    eff = policy or ReadPolicy()
    _check_realpath(
        real, policy_deny=eff.deny_paths, workspace=workspace_root
    )
    if not real.is_dir():
        raise PathPolicyError(f"{real} is not a directory")
    out: List[Dict[str, Any]] = []
    for entry in sorted(real.iterdir()):
        if not include_hidden and entry.name.startswith("."):
            continue
        if not fnmatch.fnmatch(entry.name, glob_v):
            continue
        try:
            lstat = entry.lstat()
        except OSError:
            continue
        kind = "symlink" if entry.is_symlink() else (
            "dir" if entry.is_dir() else "file"
        )
        out.append(
            {
                "name": entry.name,
                "path": str(entry),
                "type": kind,
                "size_bytes": lstat.st_size,
            }
        )
    return out

_RG_PATH_DENY = frozenset({"/proc", "/sys", "/dev"})
def grep_files(
    pattern: str,
    path: str = ".",
    glob: str = "*",
    context_lines: int = 2,
    max_matches: int = 200,
    *,
    policy: Optional[ReadPolicy] = None,
    workspace_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Grep a directory tree for ``pattern`` (ripgrep if available)."""
    if not pattern:
        raise InvalidInputError("pattern is required")
    if "\x00" in pattern:
        raise InvalidInputError("pattern must not contain NUL bytes")
    if context_lines < 0:
        raise InvalidInputError("context_lines must be >= 0")
    if max_matches <= 0:
        raise InvalidInputError("max_matches must be > 0")
    glob_v = _validate_glob(glob)
    real = _canonical(_reject_nul(path, arg_name="path"))
    eff = policy or ReadPolicy()
    if _deny_match(real, _RG_PATH_DENY):
        raise PathPolicyError(
            f"{real} is on the ripgrep deny-list {_RG_PATH_DENY}"
        )
    _check_realpath(
        real, policy_deny=eff.deny_paths, workspace=workspace_root
    )
    rg_path = shutil.which("rg")
    if rg_path is not None:
        return _grep_with_rg(
            rg_path,
            pattern=pattern,
            root=real,
            glob_v=glob_v,
            context_lines=context_lines,
            max_matches=max_matches,
            timeout=eff.rg_timeout_seconds,
            max_output=eff.max_output_bytes,
        )
    return _grep_python(
        pattern=pattern,
        root=real,
        glob_v=glob_v,
        context_lines=context_lines,
        max_matches=max_matches,
        follow=eff.follow_symlinks,
    )
def _grep_with_rg(
    rg_path: str,
    *,
    pattern: str,
    root: Path,
    glob_v: str,
    context_lines: int,
    max_matches: int,
    timeout: float,
    max_output: int,
) -> List[Dict[str, Any]]:
    argv = [
        rg_path,
        "--no-heading",
        "--line-number",
        "--no-messages",
        f"--glob={glob_v}",
        f"--max-count={max_matches}",
        "-C",
        str(context_lines),
        "--",
        pattern,
        str(root),
    ]
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise InvalidInputError(
            f"ripgrep timed out after {timeout}s"
        ) from exc

    stdout_b = proc.stdout.encode("utf-8", errors="replace")[:max_output]
    out: List[Dict[str, Any]] = []
    for line in stdout_b.decode("utf-8", errors="replace").splitlines():
        if len(out) >= max_matches:
            break
        out.append({"raw": line})
    if not out and proc.returncode not in (0, 1):
        raise InvalidInputError(
            f"ripgrep failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    return out

def _grep_python(
    *,
    pattern: str,
    root: Path,
    glob_v: str,
    context_lines: int,
    max_matches: int,
    follow: bool,
) -> List[Dict[str, Any]]:
    rx = re.compile(pattern)
    matches: List[Dict[str, Any]] = []
    for path in root.rglob(glob_v):
        if not path.is_file():
            continue
        if path.is_symlink() and not follow:
            continue
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        for idx, line in enumerate(lines):
            if not rx.search(line):
                continue
            ctx_before = lines[max(0, idx - context_lines): idx]
            ctx_after = lines[idx + 1: idx + 1 + context_lines]
            matches.append(
                {
                    "path": str(path),
                    "line": idx + 1,
                    "match": line.rstrip("\n"),
                    "context_before": [l.rstrip("\n") for l in ctx_before],
                    "context_after": [l.rstrip("\n") for l in ctx_after],
                }
            )
            if len(matches) >= max_matches:
                return matches
    return matches

# Theme 2 — Write/edit filesystem tools

def _confirm_or_raise(policy: WritePolicy, *, tool: str) -> None:
    if policy.require_confirm:
        raise ConfirmationRequired(tool=tool, request_id=secrets.token_hex(8))
def _atomic_write(
    target: Path,
    content: Union[str, bytes],
    *,
    encoding: str = "utf-8",
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(
        target.suffix + f".tmp.{os.getpid()}.{int(time.time() * 1_000_000)}"
    )
    mode = "wb" if isinstance(content, bytes) else "w"
    enc = None if isinstance(content, bytes) else encoding
    with open(tmp, mode, encoding=enc) as fh:
        if isinstance(content, str):
            fh.write(content)
        else:
            fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)
def write_file(
    path: str,
    content: str,
    mode: Literal["fail", "overwrite", "append"] = "fail",
    *,
    confirm: bool = False,
    policy: Optional[WritePolicy] = None,
    workspace_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Write ``content`` to ``path`` atomically.

    ``mode='fail'`` refuses if the file exists. ``mode='overwrite'`` requires
    ``confirm=True`` (or a policy with ``require_confirm=False``).
    """
    p = _reject_nul(path, arg_name="path")
    content = _reject_nul(content, arg_name="content")
    if mode not in ("fail", "overwrite", "append"):
        raise InvalidInputError(f"unknown mode {mode!r}")
    eff = policy or WritePolicy()
    real = _canonical(p)
    leaf = Path(p)
    _check_write_path(real, policy=eff, original=leaf)
    if eff.require_confirm and mode != "fail" and not confirm:
        raise ConfirmationRequired(tool="write_file")
    if mode == "overwrite" and not confirm and eff.require_confirm:
        raise ConfirmationRequired(tool="write_file")
    exists = real.exists()
    if mode == "fail" and exists:
        return {"path": str(real), "written": False, "reason": "exists"}
    if eff.backup_on_overwrite and mode == "overwrite" and exists:
        _backup(real, backup_dir=eff.backup_dir)
    if mode == "append":
        with open(real, "a", encoding="utf-8") as fh:
            fh.write(content)
        return {"path": str(real), "appended": True, "bytes": len(content)}

    _atomic_write(real, content)
    return {
        "path": str(real),
        "written": True,
        "bytes": len(content.encode("utf-8")),
    }

def _backup(real: Path, *, backup_dir: str) -> None:
    bak_root = real.parent / backup_dir
    bak_root.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1_000_000)
    bak = bak_root / f"{real.name}.{stamp}.bak"
    shutil.copy2(real, bak)
def edit_file(
    path: str,
    old: str,
    new: str,
    expected_replacements: Optional[int] = 1,
    *,
    confirm: bool = False,
    policy: Optional[WritePolicy] = None,
    workspace_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Replace ``old`` with ``new`` in ``path``, asserting match count."""
    p = _reject_nul(path, arg_name="path")
    old = _reject_nul(old, arg_name="old")
    new = _reject_nul(new, arg_name="new")
    eff = policy or WritePolicy()
    if len(old.encode("utf-8")) > eff.max_content_bytes:
        raise InvalidInputError("'old' exceeds content size cap")
    if len(new.encode("utf-8")) > eff.max_content_bytes:
        raise InvalidInputError("'new' exceeds content size cap")
    real = _canonical(p)
    leaf = Path(p)
    _check_write_path(real, policy=eff, original=leaf)
    if eff.require_confirm and not confirm:
        raise ConfirmationRequired(tool="edit_file")
    text = real.read_text(encoding="utf-8")
    count = text.count(old)
    target = expected_replacements if expected_replacements is not None else 1
    if count != target:
        return {
            "path": str(real),
            "replaced": 0,
            "expected": target,
            "actual": count,
            "error": "match count mismatch",
        }
    new_text = text.replace(old, new, target)
    if eff.backup_on_overwrite:
        _backup(real, backup_dir=eff.backup_dir)
    _atomic_write(real, new_text)
    return {"path": str(real), "replaced": target, "bytes": len(new_text.encode("utf-8"))}

def patch_file(path, old, new, *, confirm=False, policy=None, workspace_root=None):
    return edit_file(path, old=old, new=new, expected_replacements=1, confirm=confirm, policy=policy, workspace_root=workspace_root)
def delete_file(
    path: str,
    recursive: bool = False,
    *,
    confirm: bool = False,
    policy: Optional[WritePolicy] = None,
    workspace_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Delete a file or recursive directory (snapshot first)."""
    p = _reject_nul(path, arg_name="path")
    eff = policy or WritePolicy()
    real = _canonical(p)
    leaf = Path(p)
    _check_write_path(real, policy=eff, original=leaf)
    if eff.require_confirm and not confirm:
        raise ConfirmationRequired(tool="delete_file")
    if real.is_dir() and not recursive:
        raise SecurityError(
            f"{real} is a directory; pass recursive=True to remove"
        )
    if real.is_dir():
        bak = real.parent / eff.backup_dir / f"rm-{secrets.token_hex(6)}"
        shutil.copytree(real, bak)
        shutil.rmtree(real)
        return {"path": str(real), "deleted": True, "snapshot": str(bak)}
    if not real.exists():
        return {"path": str(real), "deleted": False, "reason": "missing"}
    real.unlink()
    return {"path": str(real), "deleted": True}


def run_command(
    argv: Sequence[str],
    cwd: str = "/workspace",
    env: Optional[Mapping[str, str]] = None,
    timeout: int = 30,
    stdin: Optional[str] = None,
    *,
    policy: Optional[ShellPolicy] = None,
) -> Dict[str, Any]:
    """Run a subprocess with a hard allow-list and substrine deny-list.

    Always ``shell=False``. The call is refused before :func:`subprocess.run`
    if any check fails.
    """
    eff = policy or ShellPolicy()
    argv = tuple(argv)
    if not argv:
        raise InvalidInputError("argv must contain at least one element")
    if len(argv) > eff.max_argv_tokens:
        raise InvalidInputError(
            f"argv has {len(argv)} tokens; cap is {eff.max_argv_tokens}"
        )
    cleaned: List[str] = []
    for tok in argv:
        tok_s = str(tok)
        if "\x00" in tok_s:
            raise InvalidInputError("argv tokens must not contain NUL bytes")
        if len(tok_s.encode("utf-8")) > eff.max_arg_token_bytes:
            raise InvalidInputError(
                f"argv token exceeds {eff.max_arg_token_bytes} bytes"
            )
        cleaned.append(tok_s)
    binary = cleaned[0]
    if binary not in eff.binary_allowlist:
        raise BinaryNotAllowedError(
            f"binary {binary!r} is not on the allow-list"
        )
    flat = " ".join(cleaned)
    for bad in eff.argv_substring_denylist:
        if bad in flat:
            raise ArgvPatternDeniedError(
                f"argv contains forbidden substring {bad!r}"
            )
    real_cwd = _canonical(cwd)
    in_workspace = (real_cwd == Path(os.path.realpath(eff.cwd_under))
                    or Path(os.path.realpath(eff.cwd_under)) in real_cwd.parents)
    in_extras = any(real_cwd == Path(p) or Path(p) in real_cwd.parents
                    for p in eff.cwd_extra)
    if not (in_workspace or in_extras):
        raise PathPolicyError(
            f"cwd {real_cwd} is outside {eff.cwd_under} and not in {eff.cwd_extra}"
        )
    if timeout < 1 or timeout > eff.timeout_max:
        raise InvalidInputError(
            f"timeout must be in [1, {eff.timeout_max}]"
        )
    env_final = _filter_env(env, eff)
    stdin_bytes: Optional[bytes] = None
    if stdin is not None:
        if "\x00" in stdin:
            raise InvalidInputError("stdin must not contain NUL bytes")
        stdin_bytes = stdin.encode("utf-8")
        if len(stdin_bytes) > eff.max_stdin_bytes:
            raise InvalidInputError(
                f"stdin exceeds {eff.max_stdin_bytes} bytes"
            )
    stdout_truncated = stderr_truncated = False
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cleaned,
            cwd=str(real_cwd),
            env=env_final,
            input=stdin_bytes,
            shell=False,
            check=False,
            timeout=timeout,
            capture_output=True,
            text=True,
            start_new_session=eff.kill_process_group_on_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        if eff.kill_process_group_on_timeout:
            proc = getattr(exc, "process", None)
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        duration_ms = int((time.monotonic() - started) * 1000)
        return {
            "argv": list(cleaned),
            "returncode": -1,
            "stdout": (exc.stdout or "")[: eff.max_stdout_bytes]
            if isinstance(exc.stdout, str)
            else "",
            "stderr": (exc.stderr or "")[: eff.max_stderr_bytes]
            if isinstance(exc.stderr, str)
            else "",
            "duration_ms": duration_ms,
            "truncated": {"stdout": False, "stderr": False},
            "error": f"timeout after {timeout}s",
        }

    duration_ms = int((time.monotonic() - started) * 1000)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if len(stdout.encode("utf-8")) > eff.max_stdout_bytes:
        stdout = stdout[: eff.max_stdout_bytes] + (
            "{... truncated at 1 MiB, see redirect files ...}"
        )
        stdout_truncated = True
    if len(stderr.encode("utf-8")) > eff.max_stderr_bytes:
        stderr = stderr[: eff.max_stderr_bytes] + (
            "{... truncated at 1 MiB, see redirect files ...}"
        )
        stderr_truncated = True

    return {
        "argv": list(cleaned),
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "duration_ms": duration_ms,
        "truncated": {"stdout": stdout_truncated, "stderr": stderr_truncated},
    }

def _filter_env(
    provided: Optional[Mapping[str, str]],
    policy: ShellPolicy,
) -> Dict[str, str]:
    if provided is None:
        return {k: v for k, v in os.environ.items() if k not in policy.blocked_env_keys}
    return {k: v for k, v in provided.items() if k not in policy.blocked_env_keys}

def create_computer_use_tools(
    workspace_root: str = None,
    write_policy: WritePolicy = None,
    shell_policy: ShellPolicy = None,
):
    """Create all computer-use tools pre-configured for a workspace."""
    import re, shlex
    workspace_root = workspace_root or os.environ.get("COMPUTER_USE_WORKSPACE",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "/examples/tools" in os.getcwd() else os.getcwd())
    write_policy = write_policy or WritePolicy(require_confirm=False, follow_symlinks="reject")
    shell_policy = shell_policy or ShellPolicy(cwd_extra=frozenset({"/tmp", workspace_root}))
    def _run_cmd(cmd: str):
        """Run a shell command."""
        import re, shlex
        argv = shlex.split(re.sub(r'^\s*cd\s+\S+\s*&&\s*', '', cmd.strip()))
        return run_command(argv=argv, cwd=workspace_root, policy=shell_policy)
    def _read(path: str):
        """Read a file's contents."""
        return read_file(path, workspace_root=workspace_root)
    def _list_dir(path: str = "."):
        """List directory contents."""
        return list_directory(path, workspace_root=workspace_root)
    def _grep(pattern: str, path: str = "."):
        """Search for pattern in files."""
        return grep_files(pattern, path, workspace_root=workspace_root)
    def _write(path: str, content: str):
        """Write content to a file (overwrite mode)."""
        return write_file(path, content, mode="overwrite", confirm=True, policy=write_policy, workspace_root=workspace_root)
    def _edit(path: str, old: str, new: str):
        """Edit a file by replacing old text with new text."""
        return edit_file(path, old=old, new=new, confirm=True, policy=write_policy, workspace_root=workspace_root)
    def _patch(path: str, old: str, new: str):
        """Patch a file (single replacement)."""
        return patch_file(path, old=old, new=new, confirm=True, policy=write_policy, workspace_root=workspace_root)
    def _delete(path: str):
        """Delete a file."""
        return delete_file(path, confirm=True, policy=write_policy, workspace_root=workspace_root)
    return {"read_file": _read, "list_directory": _list_dir, "grep_files": _grep,
            "write_file": _write, "edit_file": _edit, "patch_file": _patch, "delete_file": _delete, "run_command": _run_cmd}
