import os

from swarms.env import load_swarms_env, resolve_swarms_env_path


def test_resolve_swarms_env_path_prefers_cwd_tree(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    (home_dir / ".env").write_text("OPENAI_API_KEY=home-key\n")

    project_dir = tmp_path / "project"
    nested_dir = project_dir / "nested"
    nested_dir.mkdir(parents=True)
    project_env = project_dir / ".env"
    project_env.write_text("OPENAI_API_KEY=project-key\n")

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.chdir(nested_dir)

    assert resolve_swarms_env_path() == project_env


def test_load_swarms_env_uses_home_fallback(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    (home_dir / ".env").write_text("OPENAI_API_KEY=home-key\n")

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.chdir(project_dir)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert load_swarms_env() is True
    assert os.getenv("OPENAI_API_KEY") == "home-key"
