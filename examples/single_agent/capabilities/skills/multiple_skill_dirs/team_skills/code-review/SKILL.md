---
name: code-review
description: Review Python code for bugs, missing tests and security issues using the team's shared standards
---

# Team Code Review

Check every change against the team's standards:

1. **Correctness**: does the code do what the PR says? Look for off-by-one errors, unhandled `None` and swallowed exceptions.
2. **Tests**: every bug fix comes with a test that fails without the fix.
3. **Security**: no SQL built from strings, no `eval`, no secrets in code or logs.

Report findings as a list, most severe first, each with the file and line.
