import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

from swarms.structs.conversation import Conversation
from swarms.structs.sqlite_channel import SqliteChannel


class TestSqliteChannel:
    def test_round_trip_keeps_order_and_roles(self, tmp_path):
        channel = SqliteChannel(tmp_path / "runs.db")
        channel.post("user", "Report on EVs.")
        channel.post("Market Analyst", "Growth was 31%.")

        assert channel.messages() == [
            {"agent_name": "user", "content": "Report on EVs."},
            {
                "agent_name": "Market Analyst",
                "content": "Growth was 31%.",
            },
        ]

    def test_a_new_file_is_an_empty_channel(self, tmp_path):
        assert SqliteChannel(tmp_path / "runs.db").messages() == []

    def test_conversations_in_one_file_are_kept_apart(self, tmp_path):
        path = tmp_path / "runs.db"
        q3 = SqliteChannel(path, name="q3")
        q4 = SqliteChannel(path, name="q4")
        q3.post("user", "third quarter")
        q4.post("user", "fourth quarter")

        assert [m["content"] for m in q3.messages()] == [
            "third quarter"
        ]
        assert [m["content"] for m in q4.messages()] == [
            "fourth quarter"
        ]

    def test_messages_survive_reopening_the_file(self, tmp_path):
        path = tmp_path / "runs.db"
        first = SqliteChannel(path)
        first.post("Writer", "Done.")
        first.close()

        assert SqliteChannel(path).messages() == [
            {"agent_name": "Writer", "content": "Done."}
        ]

    def test_another_connection_sees_posts_at_once(self, tmp_path):
        path = tmp_path / "runs.db"
        SqliteChannel(path).post("user", "hi")

        rows = (
            sqlite3.connect(path)
            .execute(
                "SELECT agent_name, content, created_at FROM messages"
            )
            .fetchall()
        )
        assert rows[0][:2] == ("user", "hi")
        assert rows[0][2]

    def test_concurrent_posts_are_all_recorded(self, tmp_path):
        channel = SqliteChannel(tmp_path / "runs.db")
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(
                pool.map(
                    lambda i: channel.post(
                        f"Worker-{i}", f"answer {i}"
                    ),
                    range(40),
                )
            )

        assert len(channel.messages()) == 40
        assert threading.active_count() >= 1

    def test_clear_empties_only_this_conversation(self, tmp_path):
        path = tmp_path / "runs.db"
        a = SqliteChannel(path, name="a")
        b = SqliteChannel(path, name="b")
        a.post("user", "one")
        b.post("user", "two")
        a.clear()

        assert a.messages() == []
        assert len(b.messages()) == 1

    def test_in_memory_database_works_for_one_process(self):
        channel = SqliteChannel(":memory:")
        channel.post("user", "hi")
        assert channel.messages() == [
            {"agent_name": "user", "content": "hi"}
        ]

    def test_fetch_messages_carry_ids_and_timestamps_in_order(
        self, tmp_path
    ):
        channel = SqliteChannel(tmp_path / "runs.db")
        channel.post("user", "one")
        channel.post("Writer", "two")

        rows = channel.fetch_messages()
        assert [
            (r["id"], r["agent_name"], r["content"]) for r in rows
        ] == [
            (1, "user", "one"),
            (2, "Writer", "two"),
        ]
        assert all(r["created_at"] for r in rows)

    def test_fetch_messages_filter_by_speaker_and_keep_the_latest_n(
        self, tmp_path
    ):
        channel = SqliteChannel(tmp_path / "runs.db")
        for i in range(5):
            channel.post("Writer" if i % 2 else "user", f"m{i}")

        assert [
            r["content"]
            for r in channel.fetch_messages(agent_name="Writer")
        ] == [
            "m1",
            "m3",
        ]
        # The latest two, still oldest first.
        assert [
            r["content"] for r in channel.fetch_messages(limit=2)
        ] == [
            "m3",
            "m4",
        ]

    def test_list_conversations_lists_the_names_in_a_file(
        self, tmp_path
    ):
        path = tmp_path / "runs.db"
        assert SqliteChannel.list_conversations(path) == []
        SqliteChannel(path, name="q3").post("user", "a")
        SqliteChannel(path, name="q4").post("user", "b")
        SqliteChannel(path, name="q3").post("user", "c")

        assert SqliteChannel.list_conversations(path) == ["q3", "q4"]

    def test_a_conversation_mirrors_into_the_table(self, tmp_path):
        channel = SqliteChannel(tmp_path / "runs.db")
        conv = Conversation(channel=channel)
        conv.add("User", "Report on EVs.")
        conv.add("System", "roster")
        conv.add("Writer", "Done.")

        assert [m["agent_name"] for m in channel.messages()] == [
            "User",
            "Writer",
        ]
