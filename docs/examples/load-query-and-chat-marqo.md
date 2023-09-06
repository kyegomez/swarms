```python
from swarms import utils
from swarms.drivers import MarqoVectorStoreDriver
from swarms.engines import VectorQueryEngine
from swarms.loaders import WebLoader
from swarms.structures import Agent
from swarms.tools import KnowledgeBaseClient
import openai
from marqo import Client

# Set the OpenAI API key
openai.api_key_path = "../openai_api_key.txt"

# Define the namespace
namespace = "kyegomez"

# Initialize the vector store driver
vector_store = MarqoVectorStoreDriver(
    api_key=openai.api_key_path,
    url="http://localhost:8882",
    index="chat2",
    mq=Client(api_key="foobar", url="http://localhost:8882")
)

# Get a list of all indexes
#indexes = vector_store.get_indexes()
#print(indexes)

# Initialize the query engine
query_engine = VectorQueryEngine(vector_store_driver=vector_store)

# Initialize the knowledge base tool
kb_tool = KnowledgeBaseClient(
    description="Contains information about the Swarms Framework from www.swarms.ai",
    query_engine=query_engine,
    namespace=namespace
)

# Load artifacts from the web
artifacts = WebLoader(max_tokens=200).load("https://www.swarms.ai")

# Upsert the artifacts into the vector store
vector_store.upsert_text_artifacts({namespace: artifacts,})

# Initialize the agent
agent = Agent(tools=[kb_tool])

# Start the chat
utils.Chat(agent).start()

```