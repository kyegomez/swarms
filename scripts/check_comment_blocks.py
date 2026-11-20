"""Fail when the Python lines a change adds hold a comment block.

Usage: python scripts/check_comment_blocks.py [BASE]
"""

import re
import subprocess
import sys
import tokenize
from typing import Dict, List, Set

DIVIDER = re.compile(r"#\s*[-=*#~_]{4,}")
PRAGMA = re.compile(r"#(!|\s*(noqa|type:|fmt:))")
HUNK = re.compile(r"@@ -\S+ \+(\d+)(?:,(\d+))?")


def added_lines(base: str) -> Dict[str, Set[int]]:
    """Map each Python file changed since base to its added lines.

    Args:
        base: The branch or commit the change starts from.

    Returns:
        The added line numbers, keyed by path.
    """
    diff = subprocess.check_output(
        ["git", "diff", "-U0", "--merge-base", base, "--", "*.py"],
        text=True,
    )

    added: Dict[str, Set[int]] = {}
    path, in_hunk = None, False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            path, in_hunk = None, False
        elif line.startswith("+++ b/") and not in_hunk:
            path = line[6:]
            added[path] = set()
        elif line.startswith("@@ ") and path:
            in_hunk = True
            first, count = map(int, HUNK.match(line).groups("1"))
            added[path].update(range(first, first + count))

    return added


def comment_blocks(path: str, added: Set[int]) -> List[str]:
    """List the comment blocks and dividers among the added lines.

    Args:
        path: The Python file to read.
        added: The line numbers the change adds to it.

    Returns:
        One path:line message for each block or divider.
    """
    with tokenize.open(path) as source:
        comments = {
            tok.start[0]: tok.string
            for tok in tokenize.generate_tokens(source.readline)
            if tok.type == tokenize.COMMENT
            and tok.start[0] in added
            and not tok.line[: tok.start[1]].strip()
            and not PRAGMA.match(tok.string)
        }

    findings = []
    for line, text in comments.items():
        if line - 1 in comments:
            continue
        if line + 1 in comments:
            findings.append(f"{path}:{line}: comment block")
        elif DIVIDER.match(text):
            findings.append(f"{path}:{line}: divider comment")

    return findings


def main() -> int:
    """Print each finding and return 1 when there is one."""
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/master"
    findings = []
    for path, lines in added_lines(base).items():
        findings += comment_blocks(path, lines)

    for finding in findings:
        print(f"{finding}; keep each comment to one line")

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
