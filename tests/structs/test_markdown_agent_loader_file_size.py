import os
import tempfile
from swarms.structs.agent_loader import AgentLoader, load_agents_from_markdown


def test_markdown_agent_loader_skips_all_oversized_files_sequential():
    """Verify that consecutive oversized files are all skipped and caller's list is not mutated."""
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    agent_names = ["BigAgent1", "BigAgent2", "SmallAgent"]
    
    for name in agent_names:
        p = os.path.join(temp_dir, f"{name}.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"---\nname: {name}\n---\nYou are {name}.\n")
            if name.startswith("Big"):
                f.write("x" * 4096)
        file_paths.append(p)

    caller_list = list(file_paths)
    loader = AgentLoader()
    agents = loader.load_agents_from_markdown(
        caller_list,
        concurrent=False,
        max_file_size_mb=0.001,
    )

    # Both BigAgent1 and BigAgent2 must be skipped
    loaded_names = [a.agent_name for a in agents]
    assert loaded_names == ["SmallAgent"]
    
    # Caller's original list must not be mutated
    assert caller_list == file_paths


def test_markdown_agent_loader_skips_all_oversized_files_concurrent():
    """Verify concurrent loading also skips all consecutive oversized files without list mutation."""
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    agent_names = ["BigAgent1", "BigAgent2", "SmallAgent"]
    
    for name in agent_names:
        p = os.path.join(temp_dir, f"{name}.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"---\nname: {name}\n---\nYou are {name}.\n")
            if name.startswith("Big"):
                f.write("x" * 4096)
        file_paths.append(p)

    caller_list = list(file_paths)
    loader = AgentLoader()
    agents = loader.load_agents_from_markdown(
        caller_list,
        concurrent=True,
        max_file_size_mb=0.001,
    )

    loaded_names = [a.agent_name for a in agents]
    assert loaded_names == ["SmallAgent"]
    assert caller_list == file_paths


def test_module_level_load_agents_from_markdown_skips_all_oversized():
    """Verify module-level load_agents_from_markdown function also handles consecutive oversized files."""
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    agent_names = ["BigAgent1", "BigAgent2", "SmallAgent"]
    
    for name in agent_names:
        p = os.path.join(temp_dir, f"{name}.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"---\nname: {name}\n---\nYou are {name}.\n")
            if name.startswith("Big"):
                f.write("x" * 4096)
        file_paths.append(p)

    caller_list = list(file_paths)
    agents = load_agents_from_markdown(
        caller_list,
        concurrent=False,
        max_file_size_mb=0.001,
    )

    loaded_names = [a.agent_name for a in agents]
    assert loaded_names == ["SmallAgent"]
    assert caller_list == file_paths
