# Deployment Solutions Examples

This directory contains practical examples of different deployment strategies for Swarms agents and multi-agent systems.

## Examples Overview

### FastAPI + Uvicorn

- **File**: `fastapi_agent_api_example.py`

- **Description**: Complete FastAPI application that exposes Swarms agents as REST APIs

- **Use Case**: Creating HTTP endpoints for your agents

- **Requirements**: `requirements.txt`


### Cron Jobs

- **Directory**: `cron_job_examples/`

- **Description**: Various examples of running agents on schedules

- **Use Case**: Automated, periodic task execution

- **Examples**: Crypto tracking, stock monitoring, data processing

## Quick Start

### FastAPI Example

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API server**:
   ```bash
   python fastapi_agent_api_example.py
   ```

3. **Access the API**:
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

4. **Test with curl**:
   ```bash
   curl -X POST "http://localhost:8000/agent/run" \
        -H "Content-Type: application/json" \
        -d '{"task": "What are the best top 3 ETFs for gold coverage?"}'
   ```

### Cron Job Examples

Navigate to the `cron_job_examples/` directory for various scheduling examples:

| File Name                      | Description                              |
|------------------------------- |------------------------------------------|
| `cron_job_example.py`           | Basic cron job setup                     |
| `crypto_concurrent_cron_example.py` | Concurrent crypto monitoring         |
| `solana_price_tracker.py`       | Solana price tracking                    |
| `figma_stock_example.py`        | Stock monitoring with Figma integration  |

## Testing

Run the test script to verify your setup:

```bash
python test_fastapi_example.py
```

## Documentation

For detailed guides and documentation, see:

- [Deployment Solutions Overview](../../docs/deployment_solutions/overview.md)

- [FastAPI Agent API Guide](../../docs/deployment_solutions/fastapi_agent_api.md)

## Requirements

- Python 3.8+

- Swarms framework

- FastAPI and Uvicorn (for API examples)

- Required API keys for your chosen models

## Support

| Issue Encountered                        | Troubleshooting Step                                              |
|------------------------------------------|------------------------------------------------------------------|
| Requirements not working                 | Check the requirements are installed correctly                    |
| API authentication problems              | Verify your API keys are set                                      |
| Setup or usage confusion                 | Check the documentation for detailed setup instructions           |
| Unexpected errors or failures            | Review the test script output for debugging information           |
