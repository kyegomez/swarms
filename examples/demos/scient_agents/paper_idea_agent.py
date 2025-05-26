from swarms.prompts.paper_idea_agent import (
    PAPER_IDEA_AGENT_SYSTEM_PROMPT,
)
from swarms import Agent
from swarms.utils.any_to_str import any_to_str

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

agent = Agent(
    agent_name="Paper Idea Agent",
    agent_role="You are an experienced AI researcher tasked with proposing high-impact research ideas.",
    system_prompt=PAPER_IDEA_AGENT_SYSTEM_PROMPT,
    tools_list_dictionary=tools,
    max_loops=1,
    model_name="gpt-4o-mini",
    output_type="final",
)

out = agent.run(
    "Generate a paper idea for collaborative foundation transformer models"
)
print(any_to_str(out))
