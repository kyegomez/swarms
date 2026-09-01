import json
import concurrent.futures
import shutil
from datetime import datetime
from pathlib import Path

from loguru import logger

from swarms.structs.conversation import Conversation
from swarms.utils.litellm_tokenizer import count_tokens


def setup_temp_conversations_dir():
    """Create a temporary directory for conversation cache files."""
    temp_dir = Path("temp_test_conversations")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    logger.info(f"Created temporary test directory: {temp_dir}")
    return temp_dir


def create_test_conversation(temp_dir):
    """Create a basic conversation for testing."""
    conv = Conversation(
        name="test_conversation", conversations_dir=str(temp_dir)
    )
    conv.add("user", "Hello, world!")
    conv.add("assistant", "Hello, user!")
    logger.info("Created test conversation with basic messages")
    return conv


def test_add_message():
    logger.info("Running test_add_message")
    conv = Conversation()
    conv.add("user", "Hello, world!")
    try:
        assert len(conv.conversation_history) == 1
        assert conv.conversation_history[0]["role"] == "user"
        assert (
            conv.conversation_history[0]["content"] == "Hello, world!"
        )
        logger.success("test_add_message passed")
        return True
    except AssertionError as e:
        logger.error(f"test_add_message failed: {str(e)}")
        return False


def test_add_message_with_time():
    logger.info("Running test_add_message_with_time")
    conv = Conversation(time_enabled=False)
    conv.add("user", "Hello, world!")
    try:
        assert len(conv.conversation_history) == 1
        assert conv.conversation_history[0]["role"] == "user"
        assert (
            conv.conversation_history[0]["content"] == "Hello, world!"
        )
        assert "timestamp" in conv.conversation_history[0]
        logger.success("test_add_message_with_time passed")
        return True
    except AssertionError as e:
        logger.error(f"test_add_message_with_time failed: {str(e)}")
        return False


def test_delete_message():
    logger.info("Running test_delete_message")
    conv = Conversation()
    conv.add("user", "Hello, world!")
    conv.delete(0)
    try:
        assert len(conv.conversation_history) == 0
        logger.success("test_delete_message passed")
        return True
    except AssertionError as e:
        logger.error(f"test_delete_message failed: {str(e)}")
        return False


def test_delete_message_out_of_bounds():
    logger.info("Running test_delete_message_out_of_bounds")
    conv = Conversation()
    conv.add("user", "Hello, world!")
    try:
        conv.delete(1)
        logger.error(
            "test_delete_message_out_of_bounds failed: Expected IndexError"
        )
        return False
    except IndexError:
        logger.success("test_delete_message_out_of_bounds passed")
        return True


def test_update_message():
    logger.info("Running test_update_message")
    conv = Conversation()
    conv.add("user", "Hello, world!")
    conv.update(0, "assistant", "Hello, user!")
    try:
        assert len(conv.conversation_history) == 1
        assert conv.conversation_history[0]["role"] == "assistant"
        assert (
            conv.conversation_history[0]["content"] == "Hello, user!"
        )
        logger.success("test_update_message passed")
        return True
    except AssertionError as e:
        logger.error(f"test_update_message failed: {str(e)}")
        return False


def test_update_message_out_of_bounds():
    logger.info("Running test_update_message_out_of_bounds")
    conv = Conversation()
    conv.add("user", "Hello, world!")
    try:
        conv.update(1, "assistant", "Hello, user!")
        logger.error(
            "test_update_message_out_of_bounds failed: Expected IndexError"
        )
        return False
    except IndexError:
        logger.success("test_update_message_out_of_bounds passed")
        return True


def test_return_history_as_string():
    logger.info("Running test_return_history_as_string")
    conv = Conversation()
    conv.add("user", "Hello, world!")
    conv.add("assistant", "Hello, user!")
    result = conv.return_history_as_string()
    expected = "user: Hello, world!\n\nassistant: Hello, user!"
    try:
        assert result == expected
        logger.success("test_return_history_as_string passed")
        return True
    except AssertionError as e:
        logger.error(
            f"test_return_history_as_string failed: {str(e)}"
        )
        return False


def test_search():
    logger.info("Running test_search")
    conv = Conversation()
    conv.add("user", "Hello, world!")
    conv.add("assistant", "Hello, user!")
    results = conv.search("Hello")
    try:
        assert len(results) == 2
        assert results[0]["content"] == "Hello, world!"
        assert results[1]["content"] == "Hello, user!"
        logger.success("test_search passed")
        return True
    except AssertionError as e:
        logger.error(f"test_search failed: {str(e)}")
        return False


def test_conversation_cache_creation():
    logger.info("Running test_conversation_cache_creation")
    temp_dir = setup_temp_conversations_dir()
    try:
        conv = Conversation(
            name="cache_test", conversations_dir=str(temp_dir)
        )
        conv.add("user", "Test message")
        cache_file = temp_dir / "cache_test.json"
        result = cache_file.exists()
        if result:
            logger.success("test_conversation_cache_creation passed")
        else:
            logger.error(
                "test_conversation_cache_creation failed: Cache file not created"
            )
        return result
    finally:
        shutil.rmtree(temp_dir)


def test_conversation_cache_loading():
    logger.info("Running test_conversation_cache_loading")
    temp_dir = setup_temp_conversations_dir()
    try:
        conv1 = Conversation(
            name="load_test", conversations_dir=str(temp_dir)
        )
        conv1.add("user", "Test message")

        conv2 = Conversation.load_conversation(
            name="load_test", conversations_dir=str(temp_dir)
        )
        result = (
            len(conv2.conversation_history) == 1
            and conv2.conversation_history[0]["content"]
            == "Test message"
        )
        if result:
            logger.success("test_conversation_cache_loading passed")
        else:
            logger.error(
                "test_conversation_cache_loading failed: Loaded conversation mismatch"
            )
        return result
    finally:
        shutil.rmtree(temp_dir)


def test_add_multiple_messages():
    logger.info("Running test_add_multiple_messages")
    conv = Conversation()
    roles = ["user", "assistant", "system"]
    contents = ["Hello", "Hi there", "System message"]
    conv.add_multiple_messages(roles, contents)
    try:
        assert len(conv.conversation_history) == 3
        assert conv.conversation_history[0]["role"] == "user"
        assert conv.conversation_history[1]["role"] == "assistant"
        assert conv.conversation_history[2]["role"] == "system"
        logger.success("test_add_multiple_messages passed")
        return True
    except AssertionError as e:
        logger.error(f"test_add_multiple_messages failed: {str(e)}")
        return False


def test_query():
    logger.info("Running test_query")
    conv = Conversation()
    conv.add("user", "Test message")
    try:
        result = conv.query(0)
        assert result["role"] == "user"
        assert result["content"] == "Test message"
        logger.success("test_query passed")
        return True
    except AssertionError as e:
        logger.error(f"test_query failed: {str(e)}")
        return False


def test_display_conversation():
    logger.info("Running test_display_conversation")
    conv = Conversation()
    conv.add("user", "Hello")
    conv.add("assistant", "Hi")
    try:
        conv.display_conversation()
        logger.success("test_display_conversation passed")
        return True
    except Exception as e:
        logger.error(f"test_display_conversation failed: {str(e)}")
        return False


def test_count_messages_by_role():
    logger.info("Running test_count_messages_by_role")
    conv = Conversation()
    conv.add("user", "Hello")
    conv.add("assistant", "Hi")
    conv.add("system", "System message")
    try:
        counts = conv.count_messages_by_role()
        assert counts["user"] == 1
        assert counts["assistant"] == 1
        assert counts["system"] == 1
        logger.success("test_count_messages_by_role passed")
        return True
    except AssertionError as e:
        logger.error(f"test_count_messages_by_role failed: {str(e)}")
        return False


def test_get_str():
    logger.info("Running test_get_str")
    conv = Conversation()
    conv.add("user", "Hello")
    try:
        result = conv.get_str()
        assert "user: Hello" in result
        logger.success("test_get_str passed")
        return True
    except AssertionError as e:
        logger.error(f"test_get_str failed: {str(e)}")
        return False


def test_to_json():
    logger.info("Running test_to_json")
    conv = Conversation()
    conv.add("user", "Hello")
    try:
        result = conv.to_json()
        assert isinstance(result, str)
        assert "Hello" in result
        logger.success("test_to_json passed")
        return True
    except AssertionError as e:
        logger.error(f"test_to_json failed: {str(e)}")
        return False


def test_to_dict():
    logger.info("Running test_to_dict")
    conv = Conversation()
    conv.add("user", "Hello")
    try:
        result = conv.to_dict()
        assert isinstance(result, list)
        assert result[0]["content"] == "Hello"
        logger.success("test_to_dict passed")
        return True
    except AssertionError as e:
        logger.error(f"test_to_dict failed: {str(e)}")
        return False


def test_to_yaml():
    logger.info("Running test_to_yaml")
    conv = Conversation()
    conv.add("user", "Hello")
    try:
        result = conv.to_yaml()
        assert isinstance(result, str)
        assert "Hello" in result
        logger.success("test_to_yaml passed")
        return True
    except AssertionError as e:
        logger.error(f"test_to_yaml failed: {str(e)}")
        return False


def test_get_last_message_as_string():
    logger.info("Running test_get_last_message_as_string")
    conv = Conversation()
    conv.add("user", "First")
    conv.add("assistant", "Last")
    try:
        result = conv.get_last_message_as_string()
        assert result == "assistant: Last"
        logger.success("test_get_last_message_as_string passed")
        return True
    except AssertionError as e:
        logger.error(
            f"test_get_last_message_as_string failed: {str(e)}"
        )
        return False


def test_return_messages_as_list():
    logger.info("Running test_return_messages_as_list")
    conv = Conversation()
    conv.add("user", "Hello")
    conv.add("assistant", "Hi")
    try:
        result = conv.return_messages_as_list()
        assert len(result) == 2
        assert result[0] == "user: Hello"
        assert result[1] == "assistant: Hi"
        logger.success("test_return_messages_as_list passed")
        return True
    except AssertionError as e:
        logger.error(f"test_return_messages_as_list failed: {str(e)}")
        return False


def test_return_messages_as_dictionary():
    logger.info("Running test_return_messages_as_dictionary")
    conv = Conversation()
    conv.add("user", "Hello")
    try:
        result = conv.return_messages_as_dictionary()
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        logger.success("test_return_messages_as_dictionary passed")
        return True
    except AssertionError as e:
        logger.error(
            f"test_return_messages_as_dictionary failed: {str(e)}"
        )
        return False


def test_add_tool_output_to_agent():
    logger.info("Running test_add_tool_output_to_agent")
    conv = Conversation()
    tool_output = {"name": "test_tool", "output": "test result"}
    try:
        conv.add_tool_output_to_agent("tool", tool_output)
        assert len(conv.conversation_history) == 1
        assert conv.conversation_history[0]["role"] == "tool"
        assert conv.conversation_history[0]["content"] == tool_output
        logger.success("test_add_tool_output_to_agent passed")
        return True
    except AssertionError as e:
        logger.error(
            f"test_add_tool_output_to_agent failed: {str(e)}"
        )
        return False


def test_get_final_message():
    logger.info("Running test_get_final_message")
    conv = Conversation()
    conv.add("user", "First")
    conv.add("assistant", "Last")
    try:
        result = conv.get_final_message()
        assert result == "assistant: Last"
        logger.success("test_get_final_message passed")
        return True
    except AssertionError as e:
        logger.error(f"test_get_final_message failed: {str(e)}")
        return False


def test_get_final_message_content():
    logger.info("Running test_get_final_message_content")
    conv = Conversation()
    conv.add("user", "First")
    conv.add("assistant", "Last")
    try:
        result = conv.get_final_message_content()
        assert result == "Last"
        logger.success("test_get_final_message_content passed")
        return True
    except AssertionError as e:
        logger.error(
            f"test_get_final_message_content failed: {str(e)}"
        )
        return False


def test_return_all_except_first():
    logger.info("Running test_return_all_except_first")
    conv = Conversation()
    conv.add("system", "System")
    conv.add("user", "Hello")
    conv.add("assistant", "Hi")
    try:
        result = conv.return_all_except_first()
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        logger.success("test_return_all_except_first passed")
        return True
    except AssertionError as e:
        logger.error(f"test_return_all_except_first failed: {str(e)}")
        return False


def test_return_all_except_first_string():
    logger.info("Running test_return_all_except_first_string")
    conv = Conversation()
    conv.add("system", "System")
    conv.add("user", "Hello")
    conv.add("assistant", "Hi")
    try:
        result = conv.return_all_except_first_string()
        assert "Hello" in result
        assert "Hi" in result
        assert "System" not in result
        logger.success("test_return_all_except_first_string passed")
        return True
    except AssertionError as e:
        logger.error(
            f"test_return_all_except_first_string failed: {str(e)}"
        )
        return False


def test_batch_add():
    logger.info("Running test_batch_add")
    conv = Conversation()
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    try:
        conv.batch_add(messages)
        assert len(conv.conversation_history) == 2
        assert conv.conversation_history[0]["role"] == "user"
        assert conv.conversation_history[1]["role"] == "assistant"
        logger.success("test_batch_add passed")
        return True
    except AssertionError as e:
        logger.error(f"test_batch_add failed: {str(e)}")
        return False


def test_get_cache_stats():
    logger.info("Running test_get_cache_stats")
    conv = Conversation(cache_enabled=True)
    conv.add("user", "Hello")
    try:
        stats = conv.get_cache_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "cached_tokens" in stats
        assert "total_tokens" in stats
        assert "hit_rate" in stats
        logger.success("test_get_cache_stats passed")
        return True
    except AssertionError as e:
        logger.error(f"test_get_cache_stats failed: {str(e)}")
        return False


def test_list_cached_conversations():
    logger.info("Running test_list_cached_conversations")
    temp_dir = setup_temp_conversations_dir()
    try:
        conv = Conversation(
            name="test_list", conversations_dir=str(temp_dir)
        )
        conv.add("user", "Test message")

        conversations = Conversation.list_cached_conversations(
            str(temp_dir)
        )
        try:
            assert "test_list" in conversations
            logger.success("test_list_cached_conversations passed")
            return True
        except AssertionError as e:
            logger.error(
                f"test_list_cached_conversations failed: {str(e)}"
            )
            return False
    finally:
        shutil.rmtree(temp_dir)


def test_clear():
    logger.info("Running test_clear")
    conv = Conversation()
    conv.add("user", "Hello")
    conv.add("assistant", "Hi")
    try:
        conv.clear()
        assert len(conv.conversation_history) == 0
        logger.success("test_clear passed")
        return True
    except AssertionError as e:
        logger.error(f"test_clear failed: {str(e)}")
        return False


def test_save_and_load_json():
    logger.info("Running test_save_and_load_json")
    temp_dir = setup_temp_conversations_dir()
    file_path = temp_dir / "test_save.json"

    try:
        conv = Conversation()
        conv.add("user", "Hello")
        conv.save_as_json(str(file_path))

        conv2 = Conversation()
        conv2.load_from_json(str(file_path))

        try:
            assert len(conv2.conversation_history) == 1
            assert conv2.conversation_history[0]["content"] == "Hello"
            logger.success("test_save_and_load_json passed")
            return True
        except AssertionError as e:
            logger.error(f"test_save_and_load_json failed: {str(e)}")
            return False
    finally:
        shutil.rmtree(temp_dir)


# ── memory_md_path initialization ──


def test_memory_md_creates_parent_dir(tmp_path):
    """Parent dir is created when it doesn't exist."""
    md_path = tmp_path / "nested" / "deep" / "MEMORY.md"
    Conversation(
        memory_md_path=str(md_path),
        conversations_dir=str(tmp_path / "convs"),
    )
    assert md_path.parent.is_dir()
    assert md_path.is_file()


def test_memory_md_seeds_file_with_header(tmp_path):
    """New file is seeded with Agent Memory header and Interaction Log."""
    md_path = tmp_path / "MEMORY.md"
    Conversation(
        memory_md_path=str(md_path),
        conversations_dir=str(tmp_path / "convs"),
    )
    content = md_path.read_text()
    assert "# Agent Memory" in content
    assert "## Interaction Log" in content


def test_memory_md_no_overwrite_existing(tmp_path):
    """Existing file is NOT overwritten on construction."""
    md_path = tmp_path / "MEMORY.md"
    original = "# Existing\n\nKeep this.\n"
    md_path.write_text(original)
    Conversation(
        memory_md_path=str(md_path),
        conversations_dir=str(tmp_path / "convs"),
    )
    assert md_path.read_text() == original


def test_memory_md_none_no_file_io(tmp_path):
    """No file I/O when memory_md_path is None (default)."""
    conv = Conversation(
        conversations_dir=str(tmp_path / "convs"),
    )
    assert conv.memory_md_path is None
    conv.add("user", "Hello")
    assert not list(tmp_path.glob("**/MEMORY.md"))


# ── Preload ──


def test_preload_no_preamble_for_header_only(tmp_path):
    """Header-only file produces no preamble in history."""
    conv = Conversation(
        memory_md_path=str(tmp_path / "MEMORY.md"),
        conversations_dir=str(tmp_path / "convs"),
    )
    preambles = [
        m
        for m in conv.conversation_history
        if "Persistent Memory" in str(m.get("content", ""))
    ]
    assert len(preambles) == 0


def test_preload_injects_system_preamble(tmp_path):
    """Prior interactions produce exactly one System preamble
    containing the full file contents."""
    md_path = tmp_path / "MEMORY.md"
    file_text = (
        "# Agent Memory\n\n## Interaction Log\n\n"
        "### user — 2025-01-01T00:00:00\n\n"
        "Hello from past\n\n---\n\n"
    )
    md_path.write_text(file_text)
    conv = Conversation(
        memory_md_path=str(md_path),
        conversations_dir=str(tmp_path / "convs"),
    )
    preambles = [
        m
        for m in conv.conversation_history
        if "Persistent Memory" in str(m.get("content", ""))
    ]
    assert len(preambles) == 1
    assert preambles[0]["role"] == "System"
    # Full file contents must be embedded in the preamble
    assert file_text in preambles[0]["content"]


def test_preload_does_not_write_to_disk(tmp_path):
    """Preload appends to history, never writes back to MEMORY.md."""
    md_path = tmp_path / "MEMORY.md"
    content = (
        "# Agent Memory\n\n## Interaction Log\n\n"
        "### user — 2025-01-01T00:00:00\n\n"
        "Hello\n\n---\n\n"
    )
    md_path.write_text(content)
    Conversation(
        memory_md_path=str(md_path),
        conversations_dir=str(tmp_path / "convs"),
    )
    assert md_path.read_text() == content


# ── Write-through ──


def test_add_appends_to_memory_md(tmp_path):
    """Each add() writes a ### {role} — <iso-timestamp> block."""
    md_path = tmp_path / "MEMORY.md"
    conv = Conversation(
        memory_md_path=str(md_path),
        conversations_dir=str(tmp_path / "convs"),
    )
    conv.add("user", "Test message")
    content = md_path.read_text()
    assert "### user — " in content
    assert "Test message" in content
    assert "---" in content


def test_concurrent_add_thread_safety(tmp_path):
    """Concurrent add() calls serialize writes via the lock."""
    md_path = tmp_path / "MEMORY.md"
    conv = Conversation(
        memory_md_path=str(md_path),
        conversations_dir=str(tmp_path / "convs"),
    )

    def add_msg(i):
        conv.add("user", f"Message {i}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(add_msg, range(50)))

    content = md_path.read_text()
    for i in range(50):
        assert f"Message {i}" in content


def test_suppress_memory_md_during_setup(tmp_path):
    """system_prompt / rules / custom_rules NOT written to MEMORY.md."""
    md_path = tmp_path / "MEMORY.md"
    Conversation(
        system_prompt="You are helpful.",
        rules="Be nice.",
        custom_rules_prompt="Custom rule.",
        memory_md_path=str(md_path),
        conversations_dir=str(tmp_path / "convs"),
    )
    content = md_path.read_text()
    assert "### " not in content
    assert "You are helpful." not in content
    assert "Be nice." not in content
    assert "Custom rule." not in content


def test_empty_content_skipped(tmp_path):
    """Empty or whitespace-only content is not written to MEMORY.md."""
    md_path = tmp_path / "MEMORY.md"
    conv = Conversation(
        memory_md_path=str(md_path),
        conversations_dir=str(tmp_path / "convs"),
    )
    header = md_path.read_text()
    conv.add("user", "")
    conv.add("user", "   ")
    conv.add("user", None)
    assert md_path.read_text() == header


# ── compact ──


def test_compact_preserves_static_context(tmp_path):
    """After compact: system_prompt, rules, custom_rules, summary."""
    conv = Conversation(
        system_prompt="System prompt",
        rules="Rules text",
        custom_rules_prompt="Custom rules",
        memory_md_path=str(tmp_path / "MEMORY.md"),
        conversations_dir=str(tmp_path / "convs"),
    )
    conv.add("user", "Hello")
    conv.add("assistant", "Hi")
    conv.compact("Summary of conversation")

    h = conv.conversation_history
    assert len(h) == 4
    assert h[0]["role"] == "System"
    assert h[0]["content"] == "System prompt"
    assert h[1]["role"] == "User"
    assert h[1]["content"] == "Rules text"
    assert h[2]["role"] == "User"
    assert h[2]["content"] == "Custom rules"
    assert h[3]["role"] == "System"
    assert h[3]["content"] == "Summary of conversation"


def test_compact_removes_raw_turns(tmp_path):
    """Raw turn-by-turn messages are gone after compaction."""
    conv = Conversation(
        conversations_dir=str(tmp_path / "convs"),
    )
    conv.add("user", "Turn 1")
    conv.add("assistant", "Response 1")
    conv.compact("Summary")
    contents = [m["content"] for m in conv.conversation_history]
    assert "Turn 1" not in contents
    assert "Response 1" not in contents
    assert "Summary" in contents


def test_compact_archives_memory_md(tmp_path):
    """Prior MEMORY.md is copied to archive/ before wipe."""
    md_path = tmp_path / "MEMORY.md"
    conv = Conversation(
        memory_md_path=str(md_path),
        conversations_dir=str(tmp_path / "convs"),
    )
    conv.add("user", "Archived interaction")
    conv.compact("Summary")

    archives = list((tmp_path / "archive").glob("history_*.md"))
    assert len(archives) == 1
    assert "Archived interaction" in archives[0].read_text()


def test_compact_resets_memory_md(tmp_path):
    """After compaction MEMORY.md has fresh header + summary only."""
    md_path = tmp_path / "MEMORY.md"
    conv = Conversation(
        memory_md_path=str(md_path),
        conversations_dir=str(tmp_path / "convs"),
    )
    conv.add("user", "Old message")
    conv.compact("Fresh summary")

    content = md_path.read_text()
    assert "# Agent Memory" in content
    assert "## Interaction Log" in content
    assert "Old message" not in content
    assert "Fresh summary" in content


def test_compact_skips_archive_when_no_interactions(tmp_path):
    """No archive when MEMORY.md has no ### blocks."""
    conv = Conversation(
        memory_md_path=str(tmp_path / "MEMORY.md"),
        conversations_dir=str(tmp_path / "convs"),
    )
    conv.compact("Summary")
    archive_dir = tmp_path / "archive"
    if archive_dir.exists():
        assert len(list(archive_dir.glob("history_*.md"))) == 0


def test_archive_filename_format(tmp_path):
    """Archive filename uses %Y-%m-%d_%H-%M-%S format."""
    md_path = tmp_path / "MEMORY.md"
    conv = Conversation(
        memory_md_path=str(md_path),
        conversations_dir=str(tmp_path / "convs"),
    )
    conv.add("user", "Interaction")
    conv.compact("Summary")

    archives = list((tmp_path / "archive").glob("history_*.md"))
    assert len(archives) == 1
    stamp = archives[0].stem.replace("history_", "")
    datetime.strptime(stamp, "%Y-%m-%d_%H-%M-%S")


# ── Timestamp in prompt string ──


def test_timestamp_format_in_history_string(tmp_path):
    """Messages with timestamp render as [<ts>] Role: content."""
    conv = Conversation(
        time_enabled=True,
        dynamic_context_window=False,
        save_filepath=str(tmp_path / "timestamp_1.json"),
        conversations_dir=str(tmp_path / "convs"),
    )
    conv.add("user", "Hello")
    result = conv.return_history_as_string()
    assert result.startswith("[")
    assert "] user: Hello" in result


def test_no_timestamp_fallback(tmp_path):
    """Messages without timestamp render as Role: content."""
    conv = Conversation(
        time_enabled=False,
        dynamic_context_window=False,
        save_filepath=str(tmp_path / "timestamp_2.json"),
        conversations_dir=str(tmp_path / "convs"),
    )
    conv.add("user", "Hello")
    result = conv.return_history_as_string()
    assert result == "user: Hello"


def test_time_enabled_end_to_end(tmp_path):
    """Two messages with time_enabled contain ISO-8601 timestamps."""
    conv = Conversation(
        time_enabled=True,
        dynamic_context_window=False,
        save_filepath=str(tmp_path / "timestamp_3.json"),
        conversations_dir=str(tmp_path / "convs"),
    )
    conv.add("user", "First")
    conv.add("assistant", "Second")
    result = conv.return_history_as_string()
    lines = result.split("\n\n")
    assert len(lines) == 2
    for line in lines:
        assert line.startswith("[")
        ts = line.split("]")[0][1:]
        # Must parse as valid ISO-8601
        datetime.fromisoformat(ts)


def _chunking_conversation(tmp_path, **kwargs):
    """A Conversation that cannot inherit another test's save file.

    Conversation defaults to ``name="conversation-test"``, and since #1925 a
    bare ``Conversation()`` loads
    ``conversations/conversation_conversation-test.json`` when it exists. That
    file is written by an earlier test in this module and survives between
    runs, so these tests would start with a history they never added --
    silently, and only when run as part of the file.
    """
    return Conversation(conversations_dir=str(tmp_path), **kwargs)


def _overflowing_conversation(
    tmp_path, context_length=512, messages=60
):
    """A conversation whose history is well past ``context_length``."""
    conv = _chunking_conversation(
        tmp_path, context_length=context_length, time_enabled=True
    )
    for index in range(messages):
        conv.add("user", f"message {index} " + "word " * 100)
    return conv


def test_dynamic_chunking_stays_within_context_length(tmp_path):
    """The trimmed history must fit the window it was trimmed for.

    Swept across window and message sizes because the per-message budget
    has to account for the separators the join adds back; mis-attributing
    even one of them shows up as an overflow at some ratio.
    """
    for context_length in (64, 200, 512, 2048):
        for words in (5, 40, 100):
            conv = _chunking_conversation(
                tmp_path,
                context_length=context_length,
                time_enabled=True,
            )
            for index in range(40):
                conv.add("user", f"m{index} " + "word " * words)

            result = conv.return_history_as_string()

            assert (
                count_tokens(result) <= context_length
            ), f"overflow at context_length={context_length}, words={words}"


def test_dynamic_chunking_keeps_whole_messages(tmp_path):
    """Trimming happens on message boundaries, not mid-message."""
    conv = _overflowing_conversation(tmp_path, messages=60)
    result = conv.return_history_as_string()

    # Every complete message starts with its [timestamp] under time_enabled.
    for line in result.split("\n\n"):
        assert line.startswith("[")

    assert "message 59 " in result


def test_dynamic_chunking_work_is_bounded_by_the_window(
    monkeypatch, tmp_path
):
    """Trimming must not re-tokenize the whole transcript on every read."""
    import swarms.structs.conversation as conversation_module

    tokenized = []
    original = conversation_module.count_tokens

    def recording(text, *args, **kwargs):
        tokenized.append(len(text))
        return original(text, *args, **kwargs)

    monkeypatch.setattr(
        conversation_module, "count_tokens", recording
    )

    _overflowing_conversation(
        tmp_path, messages=30
    ).return_history_as_string()
    small = sum(tokenized)

    tokenized.clear()
    _overflowing_conversation(
        tmp_path, messages=240
    ).return_history_as_string()
    large = sum(tokenized)

    # 8x the transcript, same window: tokenized characters should stay flat.
    assert large < small * 2


def test_dynamic_chunking_leaves_a_short_history_alone(tmp_path):
    """Under the limit, the full history is returned verbatim."""
    conv = _chunking_conversation(
        tmp_path, context_length=8192, time_enabled=True
    )
    conv.add("user", "hello there")
    conv.add("assistant", "general kenobi")

    assert (
        conv.return_history_as_string()
        == conv._return_history_as_string_worker()
    )


def test_dynamic_chunking_handles_a_single_oversized_message(
    tmp_path,
):
    """One message bigger than the window is trimmed, not dropped."""
    conv = _chunking_conversation(
        tmp_path, context_length=40, time_enabled=False
    )
    conv.add("user", "word " * 4000)
    result = conv.return_history_as_string()

    assert result
    assert count_tokens(result) <= conv.context_length


def test_dynamic_chunking_handles_an_empty_history(tmp_path):
    """No messages must not raise on the newest-message fallback."""
    conv = _chunking_conversation(tmp_path, context_length=40)

    assert conv.return_history_as_string() == ""


# ── Message Metadata Tests (Issue #1926) ──


def test_add_message_with_metadata(tmp_path):
    """Test that Conversation.add preserves message metadata."""
    conv = Conversation(
        name="test_add_metadata",
        save_filepath=str(tmp_path / "metadata_1.json"),
        conversations_dir=str(tmp_path / "convs"),
    )
    metadata = {"trace_id": "abc123", "cost": 0.01}
    msg = conv.add("user", "Hello with metadata", metadata=metadata)
    assert len(conv.conversation_history) == 1
    assert conv.conversation_history[0]["metadata"] == metadata
    assert msg["metadata"] == metadata
    return True


def test_add_message_with_metadata_and_category(tmp_path):
    """Test that metadata and category coexist on stored message."""
    conv = Conversation(
        name="test_add_metadata_cat",
        save_filepath=str(tmp_path / "metadata_2.json"),
        conversations_dir=str(tmp_path / "convs"),
    )
    metadata = {"trace_id": "xyz789"}
    msg = conv.add(
        "user",
        "Hello",
        category="input",
        metadata=metadata,
    )
    assert conv.conversation_history[0]["category"] == "input"
    assert conv.conversation_history[0]["metadata"] == metadata
    assert msg["category"] == "input"
    assert msg["metadata"] == metadata
    return True


def test_add_message_metadata_serialization(tmp_path):
    """Test that metadata is preserved across to_dict, to_json, and to_yaml."""
    conv = Conversation(
        name="test_add_metadata_ser",
        save_filepath=str(tmp_path / "metadata_3.json"),
        conversations_dir=str(tmp_path / "convs"),
    )
    metadata = {"source": "unit_test", "run_id": 42}
    conv.add("user", "Test metadata serialization", metadata=metadata)

    # to_dict
    dict_res = conv.to_dict()
    assert dict_res[0]["metadata"] == metadata

    # to_json
    json_res = conv.to_json()
    assert '"metadata":' in json_res
    assert '"source": "unit_test"' in json_res

    # to_yaml
    yaml_res = conv.to_yaml()
    assert "metadata:" in yaml_res
    assert "source: unit_test" in yaml_res
    return True


def test_add_message_empty_and_none_metadata(tmp_path):
    """Test that empty or None metadata does not pollute message dict."""
    conv = Conversation(
        name="test_add_metadata_empty",
        save_filepath=str(tmp_path / "metadata_4.json"),
        conversations_dir=str(tmp_path / "convs"),
    )
    msg_none = conv.add("user", "None metadata", metadata=None)
    assert "metadata" not in conv.conversation_history[0]
    assert "metadata" not in msg_none

    msg_empty = conv.add("user", "Empty metadata", metadata={})
    assert "metadata" not in conv.conversation_history[1]
    assert "metadata" not in msg_empty
    return True


def run_all_tests():
    """Run all test functions and return results."""
    logger.info("Starting test suite execution")
    test_results = []
    test_functions = [
        test_add_message,
        test_add_message_with_time,
        test_add_message_with_metadata,
        test_add_message_with_metadata_and_category,
        test_add_message_metadata_serialization,
        test_add_message_empty_and_none_metadata,
        test_delete_message,
        test_delete_message_out_of_bounds,
        test_update_message,
        test_update_message_out_of_bounds,
        test_return_history_as_string,
        test_search,
        test_conversation_cache_creation,
        test_conversation_cache_loading,
        test_add_multiple_messages,
        test_query,
        test_display_conversation,
        test_count_messages_by_role,
        test_get_str,
        test_to_json,
        test_to_dict,
        test_to_yaml,
        test_get_last_message_as_string,
        test_return_messages_as_list,
        test_return_messages_as_dictionary,
        test_add_tool_output_to_agent,
        test_get_final_message,
        test_get_final_message_content,
        test_return_all_except_first,
        test_return_all_except_first_string,
        test_batch_add,
        test_get_cache_stats,
        test_list_cached_conversations,
        test_clear,
        test_save_and_load_json,
    ]

    for test_func in test_functions:
        start_time = datetime.now()
        try:
            result = test_func()
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            test_results.append(
                {
                    "name": test_func.__name__,
                    "result": "PASS" if result else "FAIL",
                    "duration": duration,
                }
            )
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            test_results.append(
                {
                    "name": test_func.__name__,
                    "result": "ERROR",
                    "error": str(e),
                    "duration": duration,
                }
            )
            logger.error(
                f"Test {test_func.__name__} failed with error: {str(e)}"
            )

    return test_results


def generate_markdown_report(results):
    """Generate a markdown report from test results."""
    logger.info("Generating test report")

    # Summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["result"] == "PASS")
    failed_tests = sum(1 for r in results if r["result"] == "FAIL")
    error_tests = sum(1 for r in results if r["result"] == "ERROR")

    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"Errors: {error_tests}")

    report = "# Test Results Report\n\n"
    report += f"Test Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    report += "## Summary\n\n"
    report += f"- Total Tests: {total_tests}\n"
    report += f"- Passed: {passed_tests}\n"
    report += f"- Failed: {failed_tests}\n"
    report += f"- Errors: {error_tests}\n\n"

    # Detailed Results
    report += "## Detailed Results\n\n"
    report += "| Test Name | Result | Duration (s) | Error |\n"
    report += "|-----------|---------|--------------|-------|\n"

    for result in results:
        name = result["name"]
        test_result = result["result"]
        duration = f"{result['duration']:.4f}"
        error = result.get("error", "")
        report += (
            f"| {name} | {test_result} | {duration} | {error} |\n"
        )

    return report


# ============================================================================
# Context isolation: an unnamed conversation must not resume someone else's
# ============================================================================


class TestDefaultConversationDoesNotResumeFromDisk:
    """
    Regression for cross-contamination between unrelated conversations.

    ``name`` defaults to "conversation-test", so every anonymous
    ``Conversation()`` resolved to the same file under the conversations
    directory and silently loaded whatever a previous, unrelated run had left
    there. Every swarm in the process started with someone else's messages and
    re-sent them on every agent call.
    """

    def test_anonymous_conversation_starts_empty(self, tmp_path):
        stale = tmp_path / "conversation_conversation-test.json"
        stale.write_text(
            json.dumps(
                [
                    {
                        "role": "user",
                        "content": "left over from another run",
                    }
                ]
            )
        )

        conversation = Conversation(conversations_dir=str(tmp_path))

        assert (
            conversation.conversation_history == []
        ), "an unnamed conversation resumed from an unrelated file"

    def test_two_anonymous_conversations_do_not_share_state(
        self, tmp_path
    ):
        first = Conversation(
            conversations_dir=str(tmp_path), autosave=True
        )
        first.add("user", "private to the first conversation")
        first.save_as_json(
            str(tmp_path / "conversation_conversation-test.json")
        )

        second = Conversation(conversations_dir=str(tmp_path))

        contents = [m["content"] for m in second.conversation_history]
        assert "private to the first conversation" not in contents

    def test_a_named_conversation_still_resumes(self, tmp_path):
        """Resume-by-name is deliberate and must keep working."""
        first = Conversation(
            name="my-project", conversations_dir=str(tmp_path)
        )
        first.add("user", "remember this")
        first.save_as_json(
            str(tmp_path / "conversation_my-project.json")
        )

        second = Conversation(
            name="my-project", conversations_dir=str(tmp_path)
        )

        contents = [m["content"] for m in second.conversation_history]
        assert "remember this" in contents

    def test_an_explicit_filepath_still_resumes(self, tmp_path):
        path = tmp_path / "explicit.json"
        first = Conversation(save_filepath=str(path))
        first.add("user", "explicit save")
        first.save_as_json(str(path))

        second = Conversation(save_filepath=str(path))

        contents = [m["content"] for m in second.conversation_history]
        assert "explicit save" in contents


# ============================================================================
# MESSAGE ACCESSORS: STRINGS VS REAL MESSAGE DICTS
# ============================================================================


def _two_turn_conversation():
    conv = Conversation()
    conv.add("user", "Hello, world!")
    conv.add("assistant", "Hello, user!")
    return conv


def test_return_messages_as_strings_formats_role_colon_content():
    """The flattener returns renderings, not messages."""
    conv = _two_turn_conversation()

    strings = conv.return_messages_as_strings()

    assert isinstance(strings, list)
    assert all(isinstance(s, str) for s in strings)
    assert strings == [
        "user: Hello, world!",
        "assistant: Hello, user!",
    ]


def test_return_messages_as_list_returns_message_dicts():
    """The name promises a message list, so it returns dicts, not strings."""
    conversation = Conversation()
    conversation.add("user", "Hello, world!")
    conversation.add("assistant", "Hello, user!")

    messages = conversation.return_messages_as_list()

    assert all(isinstance(message, dict) for message in messages)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert [m["content"] for m in messages] == [
        "Hello, world!",
        "Hello, user!",
    ]
    # The string rendering lives under its own name.
    assert conversation.return_messages_as_strings() == [
        "user: Hello, world!",
        "assistant: Hello, user!",
    ]


def test_return_messages_as_dictionary_preserves_roles():
    """The real message-list accessor: dicts with exactly role and content.

    Asserting the type and the role keeps a future rename from silently
    swapping this with the "role: content" flattener.
    """
    conv = _two_turn_conversation()

    messages = conv.return_messages_as_dictionary()

    assert isinstance(messages, list)
    for message in messages:
        assert isinstance(message, dict)
        assert set(message.keys()) == {"role", "content"}

    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello, world!"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hello, user!"


if __name__ == "__main__":
    logger.info("Starting test execution")
    results = run_all_tests()
    report = generate_markdown_report(results)

    # Save report to file
    with open("test_results.md", "w") as f:
        f.write(report)

    logger.success(
        "Test execution completed. Results saved to test_results.md"
    )
