import pytest

from swarms.structs.comms_channel import CommsChannel
from swarms.structs.file_channel import FileChannel


class TestFileChannel:
    def test_round_trip_keeps_order_and_roles(self, tmp_path):
        channel = FileChannel(tmp_path / "run.md")
        channel.post("user", "Write a report.")
        channel.post("Market Analyst", "Growth was 31%.")

        assert channel.messages() == [
            {"agent_name": "user", "content": "Write a report."},
            {
                "agent_name": "Market Analyst",
                "content": "Growth was 31%.",
            },
        ]

    def test_only_markdown_and_text_files_are_accepted(
        self, tmp_path
    ):
        FileChannel(tmp_path / "run.MD")
        FileChannel(tmp_path / "run.txt")
        with pytest.raises(ValueError):
            FileChannel(tmp_path / "run.json")

    def test_a_missing_file_is_an_empty_channel(self, tmp_path):
        channel = FileChannel(tmp_path / "nothing" / "run.md")
        assert channel.messages() == []

    def test_post_creates_the_parent_directory(self, tmp_path):
        channel = FileChannel(tmp_path / "runs" / "q3" / "run.txt")
        channel.post("user", "hi")
        assert channel.path.read_text().strip() == "## @user\n\nhi"

    def test_headings_inside_a_message_survive(self, tmp_path):
        channel = FileChannel(tmp_path / "run.md")
        body = (
            "# Report\n\n## Supply\n\n- cells\n\n### Detail\n\ntext"
        )
        channel.post("Writer", body)
        assert channel.messages() == [
            {"agent_name": "Writer", "content": body}
        ]

    def test_a_hand_written_file_is_read_the_same_way(self, tmp_path):
        path = tmp_path / "run.md"
        path.write_text(
            "## @user\nFirst.\n\n## @Editor   \n\nSecond.\n",
            encoding="utf-8",
        )
        assert FileChannel(path).messages() == [
            {"agent_name": "user", "content": "First."},
            {"agent_name": "Editor", "content": "Second."},
        ]

    def test_clear_removes_the_file(self, tmp_path):
        channel = FileChannel(tmp_path / "run.md")
        channel.post("user", "hi")
        channel.clear()
        assert not channel.path.exists()
        assert channel.messages() == []
        channel.clear()  # Clearing an empty channel is not an error.


class TestCommsChannel:
    def test_subclasses_must_implement_post_and_messages(self):
        class Half(CommsChannel):
            def post(self, agent_name, content):
                pass

        with pytest.raises(TypeError):
            Half()

    def test_clear_is_optional(self):
        class Memory(CommsChannel):
            def __init__(self):
                self.rows = []

            def post(self, agent_name, content):
                self.rows.append(
                    {"agent_name": agent_name, "content": content}
                )

            def messages(self):
                return list(self.rows)

        channel = Memory()
        channel.post("user", "hi")
        assert channel.messages() == [
            {"agent_name": "user", "content": "hi"}
        ]
        with pytest.raises(NotImplementedError):
            channel.clear()
