# Environment Setup Guide for Swarms Contributors

Welcome to the Swarms development environment setup guide! This comprehensive guide will walk you through setting up your development environment from scratch, whether you're a first-time contributor or an experienced developer.

!!! success "🚀 One-Click Setup (Recommended)"
    **New!** Use our automated setup script that handles everything:
    ```bash
    git clone https://github.com/kyegomez/swarms.git
    cd swarms
    chmod +x scripts/setup.sh
    ./scripts/setup.sh
    ```
    This script automatically installs Poetry, creates a virtual environment, installs all dependencies, sets up pre-commit hooks, and more!

!!! info "Manual Setup"
    **Alternative**: For manual control, install Python 3.10+, Git, and Poetry, then run:
    ```bash
    git clone https://github.com/kyegomez/swarms.git
    cd swarms
    poetry install --with dev
    ```

---

## :material-list-status: Prerequisites

Before setting up your development environment, ensure you have the following installed:

### System Requirements

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.10+ | Core runtime |
| **Git** | 2.30+ | Version control |
| **Poetry** | 1.4+ | Dependency management (recommended) |
| **Node.js** | 16+ | Documentation tools (optional) |

### Operating System Support

=== "macOS"
    
    ```bash
    # Install Homebrew if not already installed
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Install prerequisites
    brew install python@3.10 git poetry node
    ```

=== "Ubuntu/Debian"
    
    ```bash
    # Update package list
    sudo apt update
    
    # Install Python 3.10 and pip
    sudo apt install python3.10 python3.10-venv python3-pip git curl
    
    # Install Poetry
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add Poetry to PATH
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    ```

=== "Windows"
    
    1. **Install Python 3.10+** from [python.org](https://python.org/downloads/)
    2. **Install Git** from [git-scm.com](https://git-scm.com/download/win)
    3. **Install Poetry** using PowerShell:
    ```powershell
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
    ```

---

## :material-auto-fix: Automated Setup (Recommended)

We provide a comprehensive setup script that automates the entire development environment setup process. This is the **recommended approach** for new contributors.

### What the Setup Script Does

The `scripts/setup.sh` script automatically handles:

- ✅ **Python Version Check**: Verifies Python 3.10+ is installed
- ✅ **Poetry Installation**: Installs Poetry if not present
- ✅ **Virtual Environment**: Creates and configures a project-specific virtual environment
- ✅ **Dependencies**: Installs all main, development, lint, and test dependencies
- ✅ **Pre-commit Hooks**: Sets up and installs pre-commit hooks for code quality
- ✅ **Environment Template**: Creates a `.env` file template with common variables
- ✅ **Verification**: Runs initial setup verification checks
- ✅ **Helpful Output**: Provides colored output and next steps

### Running the Automated Setup

```bash
# Clone the repository
git clone https://github.com/kyegomez/swarms.git
cd swarms

# Make the script executable and run it
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Script Features

=== "🎯 Smart Detection"
    The script intelligently detects your system state:
    - Checks if Poetry is already installed
    - Verifies Python version compatibility
    - Detects existing virtual environments
    - Checks for Git repository status

=== "🔧 Comprehensive Setup"
    Installs everything you need:
    ```bash
    # All dependency groups
    poetry install --with dev,lint,test
    
    # Pre-commit hooks
    pre-commit install
    pre-commit install --hook-type commit-msg
    
    # Initial verification run
    pre-commit run --all-files
    ```

=== "📋 Environment Template"
    Creates a starter `.env` file:
    ```bash
    # Generated .env template
    OPENAI_API_KEY=your_openai_api_key_here
    ANTHROPIC_API_KEY=your_anthropic_key_here
    LOG_LEVEL=INFO
    DEVELOPMENT=true
    ```

=== "💡 Helpful Guidance"
    Provides next steps and useful commands:
    - How to activate the virtual environment
    - Essential Poetry commands
    - Testing and development workflow
    - Troubleshooting tips

### When to Use Manual Setup

Use the manual setup approach if you:
- Want full control over each step
- Have specific system requirements
- Are troubleshooting installation issues
- Prefer to understand each component

---

## :material-git: Repository Setup

### Step 1: Fork and Clone

1. **Fork the repository** on GitHub: [github.com/kyegomez/swarms](https://github.com/kyegomez/swarms)

2. **Clone your fork**:
```bash
git clone https://github.com/YOUR_USERNAME/swarms.git
cd swarms
```

3. **Add upstream remote**:
```bash
git remote add upstream https://github.com/kyegomez/swarms.git
```

4. **Verify remotes**:
```bash
git remote -v
# origin    https://github.com/YOUR_USERNAME/swarms.git (fetch)
# origin    https://github.com/YOUR_USERNAME/swarms.git (push)
# upstream  https://github.com/kyegomez/swarms.git (fetch)
# upstream  https://github.com/kyegomez/swarms.git (push)
```

---

## :material-package-variant: Dependency Management

Choose your preferred method for managing dependencies:

=== "Poetry (Recommended)"
    
    Poetry provides superior dependency resolution and virtual environment management.

    ### Installation
    
    ```bash
    # Navigate to project directory
    cd swarms
    
    # Install all dependencies including development tools
    poetry install --with dev,lint,test
    
    # Activate the virtual environment
    poetry shell
    ```

    ### Useful Poetry Commands
    
    ```bash
    # Add a new dependency
    poetry add package_name
    
    # Add a development dependency
    poetry add --group dev package_name
    
    # Update dependencies
    poetry update
    
    # Show dependency tree
    poetry show --tree
    
    # Run commands in the virtual environment
    poetry run python your_script.py
    ```

=== "pip + venv"
    
    Traditional pip-based setup with virtual environments.

    ### Installation
    
    ```bash
    # Navigate to project directory
    cd swarms
    
    # Create virtual environment
    python -m venv venv
    
    # Activate virtual environment
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install core dependencies
    pip install -r requirements.txt
    
    # Install documentation dependencies (optional)
    pip install -r docs/requirements.txt
    ```

---

## :material-tools: Development Tools Setup

### Code Quality Tools

Swarms uses several tools to maintain code quality:

=== "Formatting"
    
    **Black** - Code formatter
    ```bash
    # Format code
    poetry run black swarms/
    # or with pip:
    black swarms/
    
    # Check formatting without making changes
    black swarms/ --check --diff
    ```

=== "Linting"
    
    **Ruff** - Fast Python linter
    ```bash
    # Run linter
    poetry run ruff check swarms/
    # or with pip:
    ruff check swarms/
    
    # Auto-fix issues
    ruff check swarms/ --fix
    ```

=== "Type Checking"
    
    **MyPy** - Static type checker
    ```bash
    # Run type checking
    poetry run mypy swarms/
    # or with pip:
    mypy swarms/
    ```

### Pre-commit Hooks (Optional but Recommended)

Set up pre-commit hooks to automatically run quality checks:

```bash
# Install pre-commit
poetry add --group dev pre-commit
# or with pip:
pip install pre-commit

# Install git hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

The project uses the latest ruff-pre-commit configuration with separate hooks for linting and formatting:

- **ruff-check**: Runs the linter with automatic fixes (`--fix` flag)
- **ruff-format**: Runs the formatter for code styling
- **types_or: [python, pyi]**: Excludes Jupyter notebooks from processing

This configuration ensures consistent code quality and style across the project while avoiding conflicts with Jupyter notebook files.

---

## :material-test-tube: Testing Setup

### Running Tests

```bash
# Run all tests
poetry run pytest
# or with pip:
pytest

# Run tests with coverage
poetry run pytest --cov=swarms tests/

# Run specific test file
poetry run pytest tests/test_specific_file.py

# Run tests matching a pattern
poetry run pytest -k "test_agent"
```

### Test Structure

The project uses pytest with the following structure:
```
tests/
├── agents/          # Agent-related tests
├── structs/         # Multi-agent structure tests
├── tools/           # Tool tests
├── utils/           # Utility tests
└── conftest.py      # Test configuration
```

### Writing Tests

```python
# Example test file: tests/test_example.py
import pytest
from swarms import Agent

def test_agent_creation():
    """Test that an agent can be created successfully."""
    agent = Agent(
        agent_name="test_agent",
        system_prompt="You are a helpful assistant"
    )
    assert agent.agent_name == "test_agent"

@pytest.mark.parametrize("input_val,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
])
def test_uppercase(input_val, expected):
    """Example parametrized test."""
    assert input_val.upper() == expected
```

---

## :material-book-open-page-variant: Documentation Setup

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Navigate to docs directory
cd docs

# Serve documentation locally
mkdocs serve
# Documentation will be available at http://127.0.0.1:8000
```

### Documentation Structure

```
docs/
├── index.md              # Homepage
├── mkdocs.yml           # MkDocs configuration
├── swarms/              # Core documentation
├── examples/            # Examples and tutorials
├── contributors/        # Contributor guides
└── assets/              # Images and static files
```

### Writing Documentation

Use Markdown with MkDocs extensions:

```markdown
# Page Title

!!! tip "Pro Tip"
    Use admonitions to highlight important information.

=== "Python"
    ```python
    from swarms import Agent
    agent = Agent()
    ```

=== "CLI"
    ```bash
    swarms create-agent --name myagent
    ```
```

---

## :material-application-variable: Environment Variables

Create a `.env` file for local development:

```bash
# Copy example environment file
cp .env.example .env  # if it exists

# Or create your own .env file
touch .env
```

Common environment variables:
```bash
# .env file
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Development settings
DEBUG=true
LOG_LEVEL=INFO

# Optional: Database settings
DATABASE_URL=sqlite:///swarms.db
```

---

## :material-check-circle: Verification Steps

!!! tip "Automated Verification"
    If you used the automated setup script (`./scripts/setup.sh`), most verification steps are handled automatically. The script runs verification checks and reports any issues.

For manual setups, verify your setup is working correctly:

### 1. Basic Import Test
```bash
poetry run python -c "from swarms import Agent; print('✅ Import successful')"
```

### 2. Run a Simple Agent
```python
# test_setup.py
from swarms import Agent

agent = Agent(
    agent_name="setup_test",
    system_prompt="You are a helpful assistant for testing setup.",
    max_loops=1
)

response = agent.run("Say hello!")
print(f"✅ Agent response: {response}")
```

### 3. Code Quality Check
```bash
# Run all quality checks
poetry run black swarms/ --check
poetry run ruff check swarms/
poetry run pytest tests/ -x
```

### 4. Documentation Build
```bash
cd docs
mkdocs build
echo "✅ Documentation built successfully"
```

---

## :material-rocket-launch: Development Workflow

### Creating a Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout master
git rebase upstream/master

# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes...
# Add and commit
git add .
git commit -m "feat: add your feature description"

# Push to your fork
git push origin feature/your-feature-name
```

### Daily Development Commands

```bash
# Start development session
cd swarms
poetry shell  # or source venv/bin/activate

# Pull latest changes
git fetch upstream
git rebase upstream/master

# Run tests during development
poetry run pytest tests/ -v

# Format and lint before committing
poetry run black swarms/
poetry run ruff check swarms/ --fix

# Run a quick smoke test
poetry run python -c "from swarms import Agent; print('✅ All good')"
```

---

## :material-bug: Troubleshooting

!!! tip "First Step: Try the Automated Setup"
    If you're experiencing setup issues, try running our automated setup script first:
    ```bash
    chmod +x scripts/setup.sh
    ./scripts/setup.sh
    ```
    This script handles most common setup problems automatically and provides helpful error messages.

### Common Issues and Solutions

=== "Poetry Issues"
    
    **Problem**: Poetry command not found
    ```bash
    # Solution: Add Poetry to PATH
    export PATH="$HOME/.local/bin:$PATH"
    # Add to your shell profile (.bashrc, .zshrc, etc.)
    ```
    
    **Problem**: Poetry install fails
    ```bash
    # Solution: Clear cache and reinstall
    poetry cache clear --all pypi
    poetry install --with dev
    ```

=== "Python Version Issues"
    
    **Problem**: Wrong Python version
    ```bash
    # Check Python version
    python --version
    
    # Use pyenv to manage Python versions
    curl https://pyenv.run | bash
    pyenv install 3.10.12
    pyenv local 3.10.12
    ```

=== "Import Errors"
    
    **Problem**: Cannot import swarms modules
    ```bash
    # Ensure you're in the virtual environment
    poetry shell
    # or
    source venv/bin/activate
    
    # Install in development mode
    poetry install --with dev
    # or
    pip install -e .
    ```

=== "Test Failures"
    
    **Problem**: Tests fail due to missing dependencies
    ```bash
    # Install test dependencies
    poetry install --with test
    # or
    pip install pytest pytest-cov pytest-mock
    ```

### Getting Help

If you encounter issues:

1. **Check the FAQ** in the main documentation
2. **Search existing issues** on GitHub
3. **Ask in the Discord community**: [discord.gg/jM3Z6M9uMq](https://discord.gg/EamjgSaEQf)
4. **Create a GitHub issue** with:
   - Your operating system
   - Python version
   - Error messages
   - Steps to reproduce

---

## :material-next-step: Next Steps

Now that your environment is set up:

1. **Read the Contributing Guide**: [contributors/main.md](main.md)
2. **Explore the Codebase**: Start with `swarms/structs/agent.py`
3. **Run Examples**: Check out `examples/` directory
4. **Pick an Issue**: Look for `good-first-issue` labels on GitHub
5. **Join the Community**: Discord, Twitter, and GitHub discussions

!!! success "You're Ready!"
    Your Swarms development environment is now set up! You're ready to contribute to the most important technology for multi-agent collaboration.

---

## :material-bookmark-outline: Quick Reference

### Essential Commands

```bash
# Setup (choose one)
./scripts/setup.sh                   # Automated setup (recommended)
poetry install --with dev            # Manual dependency install

# Daily workflow
poetry shell                          # Activate environment
poetry run pytest                    # Run tests
poetry run black swarms/             # Format code
poetry run ruff check swarms/        # Lint code

# Git workflow
git fetch upstream                    # Get latest changes
git rebase upstream/master           # Update your branch
git checkout -b feature/name         # Create feature branch
git push origin feature/name         # Push your changes

# Documentation
cd docs && mkdocs serve              # Serve docs locally
mkdocs build                         # Build docs
```

### Project Structure

```
swarms/
├── swarms/              # Core package
│   ├── agents/         # Agent implementations
│   ├── structs/        # Multi-agent structures
│   ├── tools/          # Agent tools
│   └── utils/          # Utilities
├── examples/           # Usage examples
├── tests/              # Test suite
├── docs/               # Documentation
├── pyproject.toml      # Poetry configuration
└── requirements.txt    # Pip dependencies
```

Happy coding! 🚀 