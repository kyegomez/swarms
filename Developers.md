Developers

Install pre-commit (https://pre-commit.com/)

```bash
pip install pre-commit
```

Check that it's installed

```bash
pre-commit --versioni
```

This repository already has a pre-commit configuration. To install the hooks, run:

```bash
pre-commit install
```

Now when you make a git commit, the black code formatter and ruff linter will run.
