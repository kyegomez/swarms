# Swarms Chief of Staff

Lightweight orchestration service that keeps tasks, does research, proposes actions, and spawns sub-agents using the `swarms` package.

Install in editable mode for local development:

```bash
python -m pip install -e .
```

Run CLI:

```bash
swarms-chief --help
python -m swarms_chief_of_staff.cli --help
```

## GitHub repository & publishing

Create a remote repository using GitHub CLI (recommended):

```bash
# from the package root
./create_github_repo.sh my-username swarms-chief-of-staff
```

The script requires `gh` to be installed and authenticated (`gh auth login`). It creates
the repo and pushes the current tree. Alternatively, create a repo on GitHub and push the
code manually.

## CI

A CI workflow is included at `.github/workflows/ci.yml` to run tests on pushes and PRs.

## Publishing

To publish the package to PyPI or a private registry, add credentials and use `build` and `twine`:

```bash
python -m pip install build twine
python -m build
twine upload dist/*
```

## Local development

Install in editable mode for local development:

```bash
python -m pip install -e .
```

Run the CLI (uses local `swarms` package in this workspace):

```bash
PYTHONPATH=src:/workspaces/swarms swarms-chief list-tasks
```
