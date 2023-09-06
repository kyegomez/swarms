```python
from swarms.artifacts import BaseArtifact
from swarms.drivers import LocalVectorStoreDriver
from swarms.loaders import WebLoader


vector_store = LocalVectorStoreDriver()

[
    vector_store.upsert_text_artifact(a, namespace="swarms")
    for a in WebLoader(max_tokens=100).load("https://www.swarms.ai")
]

results = vector_store.query(
    "creativity",
    count=3,
    namespace="swarms"
)

values = [BaseArtifact.from_json(r.meta["artifact"]).value for r in results]

print("\n\n".join(values))
```