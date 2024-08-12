from swarms import GPT4VisionAPI, Agent

llm = GPT4VisionAPI()

agent = Agent(
    max_loops="auto",
    llm=llm,
)

agent.run(
    task="Describe this image in a few sentences: ",
    img="https://unsplash.com/photos/0pIC5ByPpZY",
)
