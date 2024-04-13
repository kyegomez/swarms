from swarms import Agent, OpenAIChat

# Initialize the workflow
agent = Agent(
    llm=OpenAIChat(),
    max_loops="auto",
    agent_name="Amazon Product Scraper",
    system_prompt=(
        "Create the code in python to scrape amazon product reviews"
        " and return csv given a product url"
    ),
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    interactive=True,
)

# Run the workflow on a task
agent(
    "Create the code to scrape this amazon url and rturn a csv of"
    " reviews:"
    " https://www.amazon.com/Creative-Act-Way-Being/dp/0593652886/ref=sr_1_1?dib=eyJ2IjoiMSJ9.JVdL3JSDmBVH_jv4eM6YE4npUpG6jO6-ai6lgmax-Ya4nH3oPk8cxkmzKsx9yAMX-Eo4A1ErqipCeY-FhTqMc7hhNTqCoAvNd65rvXH1GnYv7WlfSDYTjMkB_vVrH-iitBXAY6uASm73ff2hPWzqhF3ldGkYr8fA5FtmoYMSOnarvCU11YpoSp3EqdK526XOxkRJqeFlZAoAkXOmYHe9B5sY8-zQlVgkIV3U-7rUQdY.UXen28vr2K-Tbbz9aB7vNLLurAiR2ZSblFOVNjXYaf8&dib_tag=se&hvadid=652633987879&hvdev=c&hvlocphy=9061268&hvnetw=g&hvqmt=e&hvrand=413884426001746223&hvtargid=kwd-1977743614989&hydadcr=8513_13545021&keywords=the+creative+act+rick+rubin+book&qid=1710541252&sr=8-1"
)
