from swarms.structs.agent import Agent
from swarms.utils.litellm_wrapper import LiteLLM
import os

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    llm = LiteLLM(
        model_name="gpt-4o-mini",
        openai_api_key=api_key,
        temperature=0.1,
        max_tokens=256,
        system_prompt="You are an XML output agent."
    )
    agent = Agent(
        llm=llm,
        agent_name="XMLAgent",
        agent_description="Demo agent for XML output",
        max_loops=1,
        output_type="xml",
    )
    result = agent.run("What is the capital of France?")
    print("Agent XML output:")
    print(result)

if __name__ == "__main__":
    main()
