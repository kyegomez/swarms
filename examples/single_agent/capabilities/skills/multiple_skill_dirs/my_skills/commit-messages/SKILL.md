---
name: commit-messages
description: Write git commit messages and pull request titles in the WARP format used by the swarms repo
---

# Commit Messages

Use `[TYPE][Function/FileName][Short Description]`:

- **TYPE** in capitals: `FEAT`, `FIX`, `DOCS`, `REFACTOR`, `TEST` or `CHORE`.
- **Function/FileName**: the function, class, module or file the change is about.
- **Short Description**: one imperative line.

Examples:

```
[FIX][Agent._run][Raise AgentLLMError after retry exhaustion]
[FEAT][SkillsManager][Load skills from several directories at construction]
```
