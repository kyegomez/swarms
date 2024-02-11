# Example

import os

from dotenv import load_dotenv

from swarms import Agent, OpenAIChat
from swarms.structs.company import Company

load_dotenv()

llm = OpenAIChat(
    openai_api_key=os.getenv("OPENAI_API_KEY"), max_tokens=4000
)

ceo = Agent(llm=llm, ai_name="CEO")
dev = Agent(llm=llm, ai_name="Developer")
va = Agent(llm=llm, ai_name="VA")

# Create a company
company = Company(
    org_chart=[[dev, va]],
    shared_instructions="Do your best",
    ceo=ceo,
)

# Add agents to the company
hr = Agent(llm=llm, name="HR")
company.add(hr)

# Get an agent from the company
hr = company.get("CEO")

# Remove an agent from the company
company.remove(hr)

# Run the company
company.run()
