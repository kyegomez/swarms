"""
Functional tests for autoswarm file writer.

These tests actually execute the generated Python code and verify that
real Agent and SwarmRouter objects are created with the correct parameters.
"""

import ast
import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

from swarms.agents.auto_generate_swarm_config import (
    _agent_var_name,
    _format_value,
    _slugify,
    write_autoswarm_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = {
    "agents": [
        {
            "agent_name": "Researcher",
            "system_prompt": "You are a research specialist.",
            "model_name": "gpt-4.1",
            "max_loops": 1,
            "temperature": 0.5,
        },
        {
            "agent_name": "Analyst",
            "system_prompt": "You are a data analyst.",
            "model_name": "gpt-4.1",
            "max_loops": 2,
            "temperature": 0.3,
        },
        {
            "agent_name": "Writer",
            "system_prompt": "You are a technical writer.",
            "model_name": "gpt-4.1",
            "max_loops": 1,
        },
    ],
    "swarm_architecture": {
        "name": "Research-Pipeline",
        "description": "End-to-end research pipeline",
        "swarm_type": "SequentialWorkflow",
        "max_loops": 1,
    },
}

TASK = "build a research pipeline"


def _generate_file(config=None, task=None, output_path=None):
    """Write autoswarm file to a temp location and return (path, source)."""
    config = config if config is not None else SAMPLE_CONFIG
    task = task or TASK
    with tempfile.TemporaryDirectory() as tmpdir:
        path = output_path or os.path.join(tmpdir, "test_output.py")
        result_path = write_autoswarm_file(
            config=config, task=task, output_path=path
        )
        with open(result_path) as f:
            source = f.read()
    return result_path, source


def _exec_generated_code(source: str) -> dict:
    """Execute generated source and return the resulting namespace.

    The generated code imports Agent and SwarmRouter from swarms.
    We exec it in a namespace with the real classes so we can inspect
    the constructed objects afterward.
    """
    from swarms.structs.agent import Agent
    from swarms.structs.swarm_router import SwarmRouter

    namespace = {
        "__builtins__": __builtins__,
    }

    # Inject the real modules into a fake import system so `from swarms import Agent` works
    # We do this by exec'ing the source after stripping import lines, then injecting manually.
    lines = source.splitlines()
    code_lines = []
    for line in lines:
        if line.startswith("from swarms"):
            continue  # skip imports, we inject below
        code_lines.append(line)

    namespace["Agent"] = Agent
    namespace["SwarmRouter"] = SwarmRouter
    namespace["__name__"] = (
        "not_main"  # prevent __main__ block from running
    )

    exec("\n".join(code_lines), namespace)
    return namespace


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_basic(self):
        assert _slugify("Research Pipeline") == "research_pipeline"

    def test_special_characters(self):
        assert _slugify("My Swarm! @#$%") == "my_swarm"

    def test_multiple_spaces_and_dashes(self):
        assert _slugify("hello---world   foo") == "hello_world_foo"

    def test_empty_string(self):
        assert _slugify("") == ""


class TestFormatValue:
    def test_string(self):
        assert _format_value("hello") == "'hello'"

    def test_multiline_string(self):
        result = _format_value("line1\nline2")
        assert '"""' in result

    def test_bool(self):
        assert _format_value(True) == "True"
        assert _format_value(False) == "False"

    def test_int(self):
        assert _format_value(42) == "42"

    def test_float(self):
        assert _format_value(0.5) == "0.5"


class TestAgentVarName:
    def test_basic(self):
        assert _agent_var_name("Researcher") == "researcher"

    def test_with_dashes(self):
        assert (
            _agent_var_name("Data-Analysis Agent")
            == "data_analysis_agent"
        )

    def test_leading_digit(self):
        assert _agent_var_name("1st-Agent") == "agent_1st_agent"

    def test_empty(self):
        assert _agent_var_name("") == "agent"


# ---------------------------------------------------------------------------
# Functional tests — execute generated code, verify real objects
# ---------------------------------------------------------------------------


class TestGeneratedFileCreatesRealAgents:
    """Execute the generated Python and verify Agent objects are correct."""

    def test_agents_are_real_agent_instances(self):
        from swarms.structs.agent import Agent

        _, source = _generate_file()
        ns = _exec_generated_code(source)

        assert isinstance(ns["researcher"], Agent)
        assert isinstance(ns["analyst"], Agent)
        assert isinstance(ns["writer"], Agent)

    def test_agent_names_match(self):
        _, source = _generate_file()
        ns = _exec_generated_code(source)

        assert ns["researcher"].agent_name == "Researcher"
        assert ns["analyst"].agent_name == "Analyst"
        assert ns["writer"].agent_name == "Writer"

    def test_agent_system_prompts_match(self):
        _, source = _generate_file()
        ns = _exec_generated_code(source)

        assert "research specialist" in ns["researcher"].system_prompt
        assert "data analyst" in ns["analyst"].system_prompt
        assert "technical writer" in ns["writer"].system_prompt

    def test_agent_model_names_match(self):
        _, source = _generate_file()
        ns = _exec_generated_code(source)

        assert ns["researcher"].model_name == "gpt-4.1"
        assert ns["analyst"].model_name == "gpt-4.1"

    def test_agent_max_loops_match(self):
        _, source = _generate_file()
        ns = _exec_generated_code(source)

        assert ns["researcher"].max_loops == 1
        assert ns["analyst"].max_loops == 2
        assert ns["writer"].max_loops == 1

    def test_agent_temperature_match(self):
        _, source = _generate_file()
        ns = _exec_generated_code(source)

        assert ns["researcher"].temperature == 0.5
        assert ns["analyst"].temperature == 0.3

    def test_description_mapped_to_agent_description(self):
        config = {
            "agents": [
                {
                    "agent_name": "Describer",
                    "system_prompt": "You describe things.",
                    "description": "An agent that describes",
                },
            ],
        }
        _, source = _generate_file(config=config)
        ns = _exec_generated_code(source)

        assert (
            ns["describer"].agent_description
            == "An agent that describes"
        )

    def test_all_params_passed_to_agent(self):
        """Every supported YAML key ends up as the correct Agent attribute.

        Note: Agent.__init__ hardcodes context_length=16000 (overriding the
        passed value), so we verify the param appears in the generated code
        but don't assert the runtime value.
        """
        config = {
            "agents": [
                {
                    "agent_name": "Full-Agent",
                    "system_prompt": "Full config.",
                    "description": "A full agent",
                    "model_name": "gpt-4.1",
                    "max_loops": 3,
                    "temperature": 0.7,
                    "max_tokens": 4096,
                    "autosave": True,
                    "verbose": True,
                    "dynamic_temperature_enabled": True,
                    "context_length": 100000,
                    "output_type": "json",
                    "saved_state_path": "state.json",
                    "user_name": "tester",
                    "retry_attempts": 5,
                    "return_step_meta": True,
                    "dashboard": False,
                    "auto_generate_prompt": False,
                },
            ],
        }
        _, source = _generate_file(config=config)
        ns = _exec_generated_code(source)
        agent = ns["full_agent"]

        assert agent.agent_name == "Full-Agent"
        assert agent.agent_description == "A full agent"
        assert agent.model_name == "gpt-4.1"
        assert agent.max_loops == 3
        assert agent.temperature == 0.7
        assert agent.max_tokens == 4096
        assert agent.autosave is True
        assert agent.verbose is True
        assert agent.dynamic_temperature_enabled is True
        # context_length and saved_state_path are passed correctly in generated
        # code but Agent.__init__ overwrites them unconditionally — verify the
        # code emits them, not the runtime values
        assert "context_length=100000" in source
        assert "saved_state_path='state.json'" in source
        assert agent.output_type == "json"
        assert agent.user_name == "tester"
        assert agent.retry_attempts == 5
        # return_step_meta is not a current Agent attribute but the generated
        # code should still emit it (Agent accepts **kwargs)
        assert "return_step_meta=True" in source
        assert agent.dashboard is False
        assert agent.auto_generate_prompt is False


class TestGeneratedFileCreatesRealSwarmRouter:
    """Execute the generated Python and verify SwarmRouter is correct."""

    def test_swarm_is_real_swarm_router(self):
        from swarms.structs.swarm_router import SwarmRouter

        _, source = _generate_file()
        ns = _exec_generated_code(source)

        assert isinstance(ns["swarm"], SwarmRouter)

    def test_swarm_name_matches(self):
        _, source = _generate_file()
        ns = _exec_generated_code(source)

        assert ns["swarm"].name == "Research-Pipeline"

    def test_swarm_description_matches(self):
        _, source = _generate_file()
        ns = _exec_generated_code(source)

        assert (
            ns["swarm"].description == "End-to-end research pipeline"
        )

    def test_swarm_type_matches(self):
        _, source = _generate_file()
        ns = _exec_generated_code(source)

        assert ns["swarm"].swarm_type == "SequentialWorkflow"

    def test_swarm_max_loops_matches(self):
        _, source = _generate_file()
        ns = _exec_generated_code(source)

        assert ns["swarm"].max_loops == 1

    def test_swarm_has_all_agents(self):
        from swarms.structs.agent import Agent

        _, source = _generate_file()
        ns = _exec_generated_code(source)

        agents = ns["swarm"].agents
        assert len(agents) == 3
        assert all(isinstance(a, Agent) for a in agents)
        names = [a.agent_name for a in agents]
        assert names == ["Researcher", "Analyst", "Writer"]

    def test_no_swarm_architecture_creates_default_router(self):
        from swarms.structs.swarm_router import SwarmRouter

        config = {
            "agents": [
                {
                    "agent_name": "Solo",
                    "system_prompt": "I work alone.",
                },
            ],
        }
        _, source = _generate_file(config=config)
        ns = _exec_generated_code(source)

        assert isinstance(ns["swarm"], SwarmRouter)
        assert ns["swarm"].swarm_type == "SequentialWorkflow"
        assert len(ns["swarm"].agents) == 1

    def test_concurrent_workflow_type(self):
        config = {
            "agents": [
                {"agent_name": "A", "system_prompt": "Agent A."},
                {"agent_name": "B", "system_prompt": "Agent B."},
            ],
            "swarm_architecture": {
                "name": "Parallel-Swarm",
                "description": "Runs in parallel",
                "swarm_type": "ConcurrentWorkflow",
                "max_loops": 3,
            },
        }
        _, source = _generate_file(config=config)
        ns = _exec_generated_code(source)

        assert ns["swarm"].swarm_type == "ConcurrentWorkflow"
        assert ns["swarm"].max_loops == 3
        assert ns["swarm"].name == "Parallel-Swarm"


class TestDuplicateAgentNames:
    """Agents with identical names should get unique variables and all appear in the router."""

    def test_duplicate_names_deduplicated(self):
        config = {
            "agents": [
                {"agent_name": "Worker", "system_prompt": "First."},
                {"agent_name": "Worker", "system_prompt": "Second."},
            ],
            "swarm_architecture": {
                "name": "Dup-Test",
                "description": "Test",
                "swarm_type": "SequentialWorkflow",
            },
        }
        _, source = _generate_file(config=config)
        ns = _exec_generated_code(source)

        # Both agents exist with different var names
        assert ns["worker"].agent_name == "Worker"
        assert ns["worker_2"].agent_name == "Worker"
        # Different system prompts prove they're different objects
        assert (
            ns["worker"].system_prompt != ns["worker_2"].system_prompt
        )
        # Both are in the router
        assert len(ns["swarm"].agents) == 2


class TestMultilinePrompts:
    """System prompts with newlines should produce valid, executable code."""

    def test_multiline_prompt_executes(self):
        config = {
            "agents": [
                {
                    "agent_name": "Multiline-Bot",
                    "system_prompt": "Line one.\nLine two.\nLine three.",
                },
            ],
        }
        _, source = _generate_file(config=config)
        ns = _exec_generated_code(source)

        # Agent appends collaboration prompts to system_prompt, so check with `in`
        assert (
            "Line one.\nLine two.\nLine three."
            in ns["multiline_bot"].system_prompt
        )

    def test_prompt_with_quotes(self):
        config = {
            "agents": [
                {
                    "agent_name": "Quoter",
                    "system_prompt": "She said \"hello\" and he said 'goodbye'",
                },
            ],
        }
        _, source = _generate_file(config=config)
        # Must parse without error
        ast.parse(source)
        ns = _exec_generated_code(source)
        assert "hello" in ns["quoter"].system_prompt
        assert "goodbye" in ns["quoter"].system_prompt


# ---------------------------------------------------------------------------
# File output tests
# ---------------------------------------------------------------------------


class TestFileOutput:
    def test_file_written_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_output.py")
            result = write_autoswarm_file(
                config=SAMPLE_CONFIG, task=TASK, output_path=path
            )
            assert os.path.exists(result)
            assert os.path.getsize(result) > 0

    def test_custom_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = os.path.join(tmpdir, "custom_name.py")
            result = write_autoswarm_file(
                config=SAMPLE_CONFIG, task=TASK, output_path=custom
            )
            assert result == os.path.abspath(custom)

    def test_auto_filename_from_swarm_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = write_autoswarm_file(
                    config=SAMPLE_CONFIG, task=TASK
                )
                assert (
                    "autoswarm_research_pipeline.py"
                    in os.path.basename(result)
                )
            finally:
                os.chdir(orig)

    def test_output_dir_creates_file_in_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = os.path.join(tmpdir, "my_swarms")
            result = write_autoswarm_file(
                config=SAMPLE_CONFIG, task=TASK, output_dir=target_dir
            )
            assert os.path.dirname(result) == os.path.abspath(
                target_dir
            )
            assert (
                os.path.basename(result)
                == "autoswarm_research_pipeline.py"
            )
            assert os.path.exists(result)

    def test_output_dir_creates_missing_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "c")
            result = write_autoswarm_file(
                config=SAMPLE_CONFIG, task=TASK, output_dir=nested
            )
            assert os.path.exists(result)
            assert os.path.dirname(result) == os.path.abspath(nested)

    def test_output_path_takes_precedence_over_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            explicit = os.path.join(tmpdir, "explicit.py")
            other_dir = os.path.join(tmpdir, "other")
            result = write_autoswarm_file(
                config=SAMPLE_CONFIG,
                task=TASK,
                output_path=explicit,
                output_dir=other_dir,
            )
            assert result == os.path.abspath(explicit)
            assert not os.path.exists(other_dir)

    def test_valid_python_syntax(self):
        _, source = _generate_file()
        # ast.parse raises SyntaxError if invalid
        ast.parse(source)

    def test_contains_auto_generated_comment(self):
        _, source = _generate_file()
        assert "Auto-generated by" in source


# ---------------------------------------------------------------------------
# CLI argument wiring
# ---------------------------------------------------------------------------


class TestCLIAutoswarmArgs:
    def test_output_flag_short(self):
        from swarms.cli.main import setup_argument_parser

        parser = setup_argument_parser()
        args = parser.parse_args(
            [
                "autoswarm",
                "--task",
                "test",
                "--model",
                "gpt-4",
                "-o",
                "out.py",
            ]
        )
        assert args.output == "out.py"

    def test_output_flag_long(self):
        from swarms.cli.main import setup_argument_parser

        parser = setup_argument_parser()
        args = parser.parse_args(
            [
                "autoswarm",
                "--task",
                "test",
                "--model",
                "gpt-4",
                "--output",
                "out.py",
            ]
        )
        assert args.output == "out.py"

    def test_no_run_flag(self):
        from swarms.cli.main import setup_argument_parser

        parser = setup_argument_parser()
        args = parser.parse_args(
            [
                "autoswarm",
                "--task",
                "test",
                "--model",
                "gpt-4",
                "--no-run",
            ]
        )
        assert args.no_run is True

    def test_output_dir_flag_short(self):
        from swarms.cli.main import setup_argument_parser

        parser = setup_argument_parser()
        args = parser.parse_args(
            [
                "autoswarm",
                "--task",
                "test",
                "--model",
                "gpt-4",
                "-d",
                "/tmp/out",
            ]
        )
        assert args.output_dir == "/tmp/out"

    def test_output_dir_flag_long(self):
        from swarms.cli.main import setup_argument_parser

        parser = setup_argument_parser()
        args = parser.parse_args(
            [
                "autoswarm",
                "--task",
                "test",
                "--model",
                "gpt-4",
                "--output-dir",
                "/tmp/out",
            ]
        )
        assert args.output_dir == "/tmp/out"

    def test_defaults(self):
        from swarms.cli.main import setup_argument_parser

        parser = setup_argument_parser()
        args = parser.parse_args(
            ["autoswarm", "--task", "test", "--model", "gpt-4"]
        )
        assert args.output is None
        assert args.output_dir is None
        assert args.no_run is False

    def test_all_flags_combined(self):
        from swarms.cli.main import setup_argument_parser

        parser = setup_argument_parser()
        args = parser.parse_args(
            [
                "autoswarm",
                "--task",
                "do stuff",
                "--model",
                "gpt-4",
                "-o",
                "my_file.py",
                "-d",
                "/tmp/out",
                "--no-run",
            ]
        )
        assert args.task == "do stuff"
        assert args.model == "gpt-4"
        assert args.output == "my_file.py"
        assert args.output_dir == "/tmp/out"
        assert args.no_run is True


# ---------------------------------------------------------------------------
# End-to-end pipeline: YAML parse → file write → exec (real swarms objects)
# ---------------------------------------------------------------------------


class TestFullPipelineE2E:
    """End-to-end pipeline test using the same code path as the real CLI.

    Exercises: parse_yaml_from_swarm_markdown → yaml.safe_load →
    write_autoswarm_file → exec the generated file → verify real
    Agent and SwarmRouter objects with correct attributes.

    No mocks. Every swarms class is the real thing.
    """

    # This is what a real LLM returns — markdown with embedded YAML
    RAW_LLM_OUTPUT = """Here is the configuration for your research pipeline swarm:

```yaml
agents:
  - agent_name: "Summarizer"
    system_prompt: "You are a summarization specialist. Read the provided text and produce a concise, accurate summary capturing all key points."
    max_loops: 1
    autosave: true
    verbose: true
    context_length: 100000
    output_type: "str"

  - agent_name: "Translator"
    system_prompt: "You are a professional French translator. Translate the provided English text into fluent, natural French."
    max_loops: 1
    output_type: "str"

swarm_architecture:
  name: "Summarize-Translate-Pipeline"
  description: "Summarizes text then translates the summary to French"
  swarm_type: "SequentialWorkflow"
  max_loops: 1
  task: "Summarize and translate text to French"
```

This creates a two-agent sequential pipeline.
"""

    def test_full_pipeline_creates_real_objects(self):
        """Parse LLM output → write file → exec → verify real objects."""
        import yaml
        from swarms.agents.auto_generate_swarm_config import (
            parse_yaml_from_swarm_markdown,
            write_autoswarm_file,
        )
        from swarms.structs.agent import Agent
        from swarms.structs.swarm_router import SwarmRouter

        # Step 1: Parse the YAML from LLM markdown (same as generate_swarm_config)
        yaml_content = parse_yaml_from_swarm_markdown(
            self.RAW_LLM_OUTPUT
        )
        config_dict = yaml.safe_load(yaml_content)

        # Step 2: Write the file
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "pipeline_test.py")
            result_path = write_autoswarm_file(
                config=config_dict,
                task="summarize and translate to French",
                output_path=out_path,
            )

            assert os.path.exists(result_path)

            # Step 3: Read and validate syntax
            with open(result_path) as f:
                source = f.read()
            ast.parse(source)

            # Step 4: Execute the generated code with real swarms classes
            ns = _exec_generated_code(source)

        # Step 5: Verify real Agent objects
        assert isinstance(ns["summarizer"], Agent)
        assert isinstance(ns["translator"], Agent)

        assert ns["summarizer"].agent_name == "Summarizer"
        assert ns["translator"].agent_name == "Translator"
        assert (
            "summarization specialist"
            in ns["summarizer"].system_prompt
        )
        assert "French translator" in ns["translator"].system_prompt
        assert ns["summarizer"].max_loops == 1
        assert ns["translator"].max_loops == 1
        assert ns["summarizer"].autosave is True
        assert ns["summarizer"].verbose is True
        assert ns["summarizer"].output_type == "str"
        assert ns["translator"].output_type == "str"

        # Step 6: Verify real SwarmRouter
        assert isinstance(ns["swarm"], SwarmRouter)
        assert ns["swarm"].name == "Summarize-Translate-Pipeline"
        assert (
            ns["swarm"].description
            == "Summarizes text then translates the summary to French"
        )
        assert ns["swarm"].swarm_type == "SequentialWorkflow"
        assert ns["swarm"].max_loops == 1

        # Step 7: Verify agent wiring — router has both agents in correct order
        assert len(ns["swarm"].agents) == 2
        assert all(isinstance(a, Agent) for a in ns["swarm"].agents)
        assert ns["swarm"].agents[0].agent_name == "Summarizer"
        assert ns["swarm"].agents[1].agent_name == "Translator"

    def test_complex_config_with_many_agents(self):
        """A more complex config with 4 agents and various params."""
        import yaml
        from swarms.agents.auto_generate_swarm_config import (
            parse_yaml_from_swarm_markdown,
            write_autoswarm_file,
        )

        llm_output = """
```yaml
agents:
  - agent_name: "Data-Collector"
    system_prompt: "You collect and organize raw data from various sources."
    max_loops: 2
    model_name: "gpt-4.1"
    temperature: 0.3
    verbose: true

  - agent_name: "Analyzer"
    system_prompt: "You analyze data patterns and extract insights."
    max_loops: 3
    model_name: "gpt-4.1"
    temperature: 0.5

  - agent_name: "Visualizer"
    system_prompt: "You create data visualizations and charts."
    max_loops: 1
    model_name: "gpt-4.1"
    temperature: 0.2

  - agent_name: "Report-Writer"
    system_prompt: "You write comprehensive reports based on analysis and visualizations."
    max_loops: 1
    model_name: "gpt-4.1"
    temperature: 0.7
    autosave: true

swarm_architecture:
  name: "Data-Analysis-Pipeline"
  description: "End-to-end data analysis from collection to reporting"
  swarm_type: "SequentialWorkflow"
  max_loops: 1
  task: "Analyze the quarterly sales data"
```
"""
        yaml_content = parse_yaml_from_swarm_markdown(llm_output)
        config_dict = yaml.safe_load(yaml_content)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "complex_test.py")
            write_autoswarm_file(
                config=config_dict,
                task="analyze quarterly sales data",
                output_path=out_path,
            )
            with open(out_path) as f:
                source = f.read()

        ast.parse(source)
        ns = _exec_generated_code(source)

        # 4 real agents
        assert len(ns["swarm"].agents) == 4
        names = [a.agent_name for a in ns["swarm"].agents]
        assert names == [
            "Data-Collector",
            "Analyzer",
            "Visualizer",
            "Report-Writer",
        ]

        # Verify individual agent params
        assert ns["data_collector"].max_loops == 2
        assert ns["data_collector"].temperature == 0.3
        assert ns["data_collector"].verbose is True

        assert ns["analyzer"].max_loops == 3
        assert ns["analyzer"].temperature == 0.5

        assert ns["visualizer"].max_loops == 1
        assert ns["visualizer"].temperature == 0.2

        assert ns["report_writer"].temperature == 0.7
        assert ns["report_writer"].autosave is True

        # Router
        assert ns["swarm"].name == "Data-Analysis-Pipeline"
        assert ns["swarm"].swarm_type == "SequentialWorkflow"
