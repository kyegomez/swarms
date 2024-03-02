import threading

from swarms.memory.short_term_memory import ShortTermMemory


def test_init():
    memory = ShortTermMemory()
    assert memory.short_term_memory == []
    assert memory.medium_term_memory == []


def test_add():
    memory = ShortTermMemory()
    memory.add("user", "Hello, world!")
    assert memory.short_term_memory == [
        {"role": "user", "message": "Hello, world!"}
    ]


def test_get_short_term():
    memory = ShortTermMemory()
    memory.add("user", "Hello, world!")
    assert memory.get_short_term() == [
        {"role": "user", "message": "Hello, world!"}
    ]


def test_get_medium_term():
    memory = ShortTermMemory()
    memory.add("user", "Hello, world!")
    memory.move_to_medium_term(0)
    assert memory.get_medium_term() == [
        {"role": "user", "message": "Hello, world!"}
    ]


def test_clear_medium_term():
    memory = ShortTermMemory()
    memory.add("user", "Hello, world!")
    memory.move_to_medium_term(0)
    memory.clear_medium_term()
    assert memory.get_medium_term() == []


def test_get_short_term_memory_str():
    memory = ShortTermMemory()
    memory.add("user", "Hello, world!")
    assert (
        memory.get_short_term_memory_str()
        == "[{'role': 'user', 'message': 'Hello, world!'}]"
    )


def test_update_short_term():
    memory = ShortTermMemory()
    memory.add("user", "Hello, world!")
    memory.update_short_term(0, "user", "Goodbye, world!")
    assert memory.get_short_term() == [
        {"role": "user", "message": "Goodbye, world!"}
    ]


def test_clear():
    memory = ShortTermMemory()
    memory.add("user", "Hello, world!")
    memory.clear()
    assert memory.get_short_term() == []


def test_search_memory():
    memory = ShortTermMemory()
    memory.add("user", "Hello, world!")
    assert memory.search_memory("Hello") == {
        "short_term": [
            (0, {"role": "user", "message": "Hello, world!"})
        ],
        "medium_term": [],
    }


def test_return_shortmemory_as_str():
    memory = ShortTermMemory()
    memory.add("user", "Hello, world!")
    assert (
        memory.return_shortmemory_as_str()
        == "[{'role': 'user', 'message': 'Hello, world!'}]"
    )


def test_move_to_medium_term():
    memory = ShortTermMemory()
    memory.add("user", "Hello, world!")
    memory.move_to_medium_term(0)
    assert memory.get_medium_term() == [
        {"role": "user", "message": "Hello, world!"}
    ]
    assert memory.get_short_term() == []


def test_return_medium_memory_as_str():
    memory = ShortTermMemory()
    memory.add("user", "Hello, world!")
    memory.move_to_medium_term(0)
    assert (
        memory.return_medium_memory_as_str()
        == "[{'role': 'user', 'message': 'Hello, world!'}]"
    )


def test_thread_safety():
    memory = ShortTermMemory()

    def add_messages():
        for _ in range(1000):
            memory.add("user", "Hello, world!")

    threads = [
        threading.Thread(target=add_messages) for _ in range(10)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(memory.get_short_term()) == 10000


def test_save_and_load():
    memory1 = ShortTermMemory()
    memory1.add("user", "Hello, world!")
    memory1.save_to_file("memory.json")
    memory2 = ShortTermMemory()
    memory2.load_from_file("memory.json")
    assert memory1.get_short_term() == memory2.get_short_term()
    assert memory1.get_medium_term() == memory2.get_medium_term()
