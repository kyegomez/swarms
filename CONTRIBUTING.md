# Contributing to Swarms 🛠️

Thank you for your interest in contributing to Swarms! This guide will help you get started with your contribution.

## Essential Steps for Contributing

### 1. Fork and Clone the Repository

1. Fork the Swarms repository to your GitHub account by clicking the "Fork" button in the top-right corner of the repository page.
2. Clone your forked repository to your local machine:
   ```
   git clone https://github.com/kyegomez/swarms.git
   ```

### 2. Make Changes

1. Create a new branch for your changes:
   ```
   git checkout -b your-branch-name
   ```
2. Make your desired changes to the codebase.

### 3. Commit and Push Changes

1. Stage your changes:
   ```
   git add .
   ```
2. Commit your changes:
   ```
   git commit -m "Your descriptive commit message"
   ```
3. Push your changes to your fork:
   ```
   git push -u origin your-branch-name
   ```

### 4. Create a Pull Request

1. Go to the original Swarms repository on GitHub.
2. Click on "Pull Requests" and then "New Pull Request".
3. Select your fork and branch as the compare branch.
4. Click "Create Pull Request".
5. Fill in the pull request description and submit.

## Important Considerations

- Ensure your code follows the project's coding standards.
- Include tests for new features or bug fixes.
- Update documentation as necessary.
- Make sure your branch is up-to-date with the main repository before submitting a pull request.

## Additional Information

### Code Quality

We use pre-commit hooks to maintain code quality. To set up pre-commit:

1. Install pre-commit:
   ```
   pip install pre-commit
   ```
2. Set up the git hook scripts:
   ```
   pre-commit install
   ```

### Documentation

- We use mkdocs for documentation. To serve the docs locally:
  ```
  mkdocs serve
  ```

### Testing

- Run tests using pytest:
  ```
  pytest
  ```

For more detailed information on code quality, documentation, and testing, please refer to the full contributing guidelines in the repository.

### Integration Testing

- To run the integration tests, you should first start the test-server.

```bash
$ docker-compose -f docker-compose-test.yaml up -d
```

If you do not want to use docker, you can navigate to `test-server` and follow the instruction there. It simply runs a FastAPI simple Rest API.

- Make sure your virtual environment is active. Execute the test (currently, there is only one test using it)

```bash
python -m pytest tests/agents/test_openai.py -o log_cli=true
```

#### Extending integration tests

Make sure the testing server is running. If you want to extend tests, first you need to select your agent. Probably, this agent has an API exposed so an SDK or a client can interact with it. Most agents are configurable, including the API URL. Modify that URL to `http://localhost:8000` and run the workflow you want to test. The testing server is configured to log requests that it does not handle. So, take that log (it includes the path and method), get a valid response and extend the server. To get a valid response, you can either go to the SDK docs or by running it locally with debug and finding your own valid response. Try to follow the examples available.
