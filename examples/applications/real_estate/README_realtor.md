# Realtor Agent Example

This example demonstrates how to create an AI-powered rental property specialist using the Swarms framework and the Realtor API.

## Quick Start

1. Install dependencies:
```bash
pip install swarms
```

2. Get your Realtor API key:
- Visit [Realtor Search API](https://rapidapi.com/ntd119/api/realtor-search/)
- Sign up for RapidAPI
- Subscribe to the API
- Copy your API key

3. Update the API key in `realtor_agent.py`:
```python
headers = {
    "x-rapidapi-key": "YOUR_API_KEY_HERE",
    "x-rapidapi-host": "realtor-search.p.rapidapi.com",
}
```

4. Run the example:
```python
from realtor_agent import agent

# Search single location
response = agent.run(
    "What are the best properties in Menlo Park for rent under $3,000?"
    f"Data: {get_realtor_data_from_one_source('Menlo Park, CA')}"
)
print(response)
```

## Features

- Property search across multiple locations
- Detailed property analysis
- Location assessment
- Financial analysis
- Tenant matching recommendations

For full documentation, see [docs/examples/realtor_agent.md](../docs/examples/realtor_agent.md). 