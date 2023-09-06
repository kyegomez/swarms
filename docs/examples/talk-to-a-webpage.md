This example demonstrates how to vectorize a webpage and setup a Swarms agent with rules and the `KnowledgeBase` tool to use it during conversations.

```python
from swarms.engines import VectorQueryEngine
from swarms.loaders import WebLoader
from swarms.rules import Ruleset, Rule
from swarms.structures import Agent
from swarms.tools import KnowledgeBaseClient
from swarms.utils import Chat


namespace = "physics-wiki"

engine = VectorQueryEngine()

artifacts = WebLoader().load(
    "https://en.wikipedia.org/wiki/Physics"
)

engine.vector_store_driver.upsert_text_artifacts(
    {namespace: artifacts}
)


kb_client = KnowledgeBaseClient(
    description="Contains information about physics. "
                "Use it to answer any physics-related questions.",
    query_engine=engine,
    namespace=namespace
)

agent = Agent(
    rulesets=[
        Ruleset(
            name="Physics Tutor",
            rules=[
                Rule(
                    "Always introduce yourself as a physics tutor"
                ),
                Rule(
                    "Be truthful. Only discuss physics."
                )
            ]
        )
    ],
    tools=[kb_client]
)

Chat(agent).start()
```