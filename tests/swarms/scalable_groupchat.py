from unittest.mock import patch
from swarms.swarms.scalable_groupchat import ScalableGroupChat


def test_scalablegroupchat_initialization():
    scalablegroupchat = ScalableGroupChat(
        worker_count=5, collection_name="swarm", api_key="api_key"
    )
    assert isinstance(scalablegroupchat, ScalableGroupChat)
    assert len(scalablegroupchat.workers) == 5
    assert scalablegroupchat.collection_name == "swarm"
    assert scalablegroupchat.api_key == "api_key"


@patch("chromadb.utils.embedding_functions.OpenAIEmbeddingFunction")
def test_scalablegroupchat_embed(mock_openaiembeddingfunction):
    scalablegroupchat = ScalableGroupChat(
        worker_count=5, collection_name="swarm", api_key="api_key"
    )
    scalablegroupchat.embed("input", "model_name")
    assert mock_openaiembeddingfunction.call_count == 1


@patch("swarms.swarms.scalable_groupchat.ScalableGroupChat.collection.query")
def test_scalablegroupchat_retrieve_results(mock_query):
    scalablegroupchat = ScalableGroupChat(
        worker_count=5, collection_name="swarm", api_key="api_key"
    )
    scalablegroupchat.retrieve_results(1)
    assert mock_query.call_count == 1


@patch("swarms.swarms.scalable_groupchat.ScalableGroupChat.collection.add")
def test_scalablegroupchat_update_vector_db(mock_add):
    scalablegroupchat = ScalableGroupChat(
        worker_count=5, collection_name="swarm", api_key="api_key"
    )
    scalablegroupchat.update_vector_db({"vector": "vector", "task_id": "task_id"})
    assert mock_add.call_count == 1


@patch("swarms.swarms.scalable_groupchat.ScalableGroupChat.collection.add")
def test_scalablegroupchat_append_to_db(mock_add):
    scalablegroupchat = ScalableGroupChat(
        worker_count=5, collection_name="swarm", api_key="api_key"
    )
    scalablegroupchat.append_to_db("result")
    assert mock_add.call_count == 1


@patch("swarms.swarms.scalable_groupchat.ScalableGroupChat.collection.add")
@patch("swarms.swarms.scalable_groupchat.ScalableGroupChat.embed")
@patch("swarms.swarms.scalable_groupchat.ScalableGroupChat.run")
def test_scalablegroupchat_chat(mock_run, mock_embed, mock_add):
    scalablegroupchat = ScalableGroupChat(
        worker_count=5, collection_name="swarm", api_key="api_key"
    )
    scalablegroupchat.chat(sender_id=1, receiver_id=2, message="Hello, Agent 2!")
    assert mock_embed.call_count == 1
    assert mock_add.call_count == 1
    assert mock_run.call_count == 1
