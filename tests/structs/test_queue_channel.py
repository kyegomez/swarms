import multiprocessing
import threading

from swarms.structs.conversation import Conversation
from swarms.structs.queue_channel import QueueChannel


class TestQueueChannel:
    def test_posts_are_both_history_and_queued(self):
        channel = QueueChannel()
        channel.post("user", "Report on EVs.")
        channel.post("Researcher", "Growth was 31%.")

        assert channel.messages() == [
            {"agent_name": "user", "content": "Report on EVs."},
            {
                "agent_name": "Researcher",
                "content": "Growth was 31%.",
            },
        ]
        assert channel.wait(timeout=0.1) == {
            "agent_name": "user",
            "content": "Report on EVs.",
        }
        assert channel.wait(timeout=0.1)["agent_name"] == "Researcher"

    def test_wait_returns_none_on_timeout(self):
        assert QueueChannel().wait(timeout=0.05) is None

    def test_a_consumer_blocks_until_the_producer_posts(self):
        channel = QueueChannel()
        received = []
        started = threading.Event()

        def consume():
            started.set()
            for message in channel:
                received.append(message["agent_name"])

        consumer = threading.Thread(target=consume)
        consumer.start()
        started.wait(1)

        channel.post("Researcher", "one")
        channel.post("Writer", "two")
        channel.close()
        consumer.join(2)

        assert not consumer.is_alive()
        assert received == ["Researcher", "Writer"]

    def test_clear_empties_history_and_queue(self):
        channel = QueueChannel()
        channel.post("user", "hi")
        channel.clear()
        assert channel.messages() == []
        assert channel.wait(timeout=0.05) is None

    def test_accepts_a_multiprocessing_queue(self):
        channel = QueueChannel(multiprocessing.Queue())
        channel.post("user", "hi")
        assert channel.wait(timeout=1) == {
            "agent_name": "user",
            "content": "hi",
        }

    def test_a_conversation_mirrors_into_the_queue(self):
        channel = QueueChannel()
        conv = Conversation(channel=channel)
        conv.add("User", "Report on EVs.")
        conv.add("Writer", "Done.")
        assert [m["agent_name"] for m in channel.messages()] == [
            "User",
            "Writer",
        ]
        assert channel.wait(timeout=0.1)["agent_name"] == "User"
        assert channel.wait(timeout=0.1)["agent_name"] == "Writer"
