# uv project notes

Guide: (https://www.loopwerk.io/articles/2024/migrate-poetry-to-uv/)

## install uv

Follow the [instructions](https://docs.astral.sh/uv/getting-started/installation/)

## set up a virtual environment

This assumes you have uv installed.

```bash
uv venv --python 3.12.0
source .venv/bin/activate
```

or you can run uv without setting up a virtual environment

```bash
# use --sytem to run the tool ruff not in a virtual environment
uv run ruff check --system
```

# pyproject.toml differences

In Poetry, the project is described in the [tool.poetry] section of the pyproject.toml file. In uv, the project is described in the [project] section of the pyproject.toml file.

To convert, we use ```pdm``` to get us most of the way there followin [1]

```bash
uvx pdm import pyproject.toml
```

There were several differences in the pyproject.toml file that needed to be addressed manually.
The authors field needed to be in the form:

```toml
authors = [ { name = "John Doe", email = "email"}]
license = {text = "MIT License"}
```

There weren't a lot of good docs on these formatting issues, and the serde error messages were only helpful if you had some idea of what the goal was already. I ended up referring to [1] and [2] for examples.

[1](https://packaging.python.org/en/latest/flow/#the-configuration-file)
[2](https://reinforcedknowledge.com/a-comprehensive-guide-to-python-project-management-and-packaging-concepts-illustrated-with-uv-part-i/)