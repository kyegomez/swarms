from swarms import Flow, Fuyu

llm = Fuyu()

flow = Flow(max_loops="auto", llm=llm)

flow.run(
    task="Describe this image in a few sentences: ",
    img="https://unsplash.com/photos/0pIC5ByPpZY",
)
