import cProfile
import time

from swarms.prompts.paper_idea_agent import (
    PAPER_IDEA_AGENT_SYSTEM_PROMPT,
)
from swarms import Agent
from swarms.utils.any_to_str import any_to_str

print("All imports completed...")


tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_paper_idea",
            "description": "Generate a structured academic paper idea with all required components.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Concise identifier for the paper idea",
                    },
                    "title": {
                        "type": "string",
                        "description": "Academic paper title",
                    },
                    "short_hypothesis": {
                        "type": "string",
                        "description": "Core hypothesis in 1-2 sentences",
                    },
                    "related_work": {
                        "type": "string",
                        "description": "Key papers and how this differs from existing work",
                    },
                    "abstract": {
                        "type": "string",
                        "description": "Complete paper abstract",
                    },
                    "experiments": {
                        "type": "string",
                        "description": "Detailed experimental plan",
                    },
                    "risk_factors": {
                        "type": "string",
                        "description": "Known challenges and constraints",
                    },
                },
                "required": [
                    "name",
                    "title",
                    "short_hypothesis",
                    "related_work",
                    "abstract",
                    "experiments",
                    "risk_factors",
                ],
            },
        },
    }
]


# agent = Agent(
#     agent_name="Paper Idea Agent",
#     agent_role="You are an experienced AI researcher tasked with proposing high-impact research ideas.",
#     system_prompt=PAPER_IDEA_AGENT_SYSTEM_PROMPT,
#     tools_list_dictionary=tools,
#     max_loops=1,
#     model_name="gpt-4o-mini",
#     output_type="final",
# )
def generate_paper_idea():
    print("Starting generate_paper_idea function...")
    try:
        print("Creating agent...")
        agent = Agent(
            agent_name="Paper Idea Agent",
            agent_role="You are an experienced AI researcher tasked with proposing high-impact research ideas.",
            system_prompt=PAPER_IDEA_AGENT_SYSTEM_PROMPT,
            tools_list_dictionary=tools,
            max_loops=1,
            model_name="gpt-4o-mini",
            output_type="final",
        )

        print("Agent created, starting run...")
        start_time = time.time()
        out = agent.run(
            "Generate a paper idea for collaborative foundation transformer models"
        )
        end_time = time.time()

        print(f"Execution time: {end_time - start_time:.2f} seconds")
        print("Output:", any_to_str(out))
        return out
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


print("Defining main block...")
if __name__ == "__main__":
    print("Entering main block...")

    # Basic timing first
    print("\nRunning basic timing...")
    generate_paper_idea()

    # Then with profiler
    print("\nRunning with profiler...")
    profiler = cProfile.Profile()
    profiler.enable()
    generate_paper_idea()
    profiler.disable()
    profiler.print_stats(sort="cumulative")

print("Script completed.")
