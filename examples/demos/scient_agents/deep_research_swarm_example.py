from swarms.structs.deep_research_swarm import DeepResearchSwarm


model = DeepResearchSwarm(
    research_model_name="groq/deepseek-r1-distill-qwen-32b"
)

model.run(
    "What are the latest research papers on extending telomeres in humans? Give 1 queries for the search not too many`"
)
