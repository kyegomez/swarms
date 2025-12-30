import os
import tempfile
import asyncio
import json

try:
    import pytest
except ImportError:
    pytest = None

from loguru import logger

try:
    from swarms.structs.base_structure import BaseStructure
except (ImportError, ModuleNotFoundError) as e:
    import importlib.util

    _current_dir = os.path.dirname(os.path.abspath(__file__))

    base_structure_path = os.path.join(
        _current_dir,
        "..",
        "..",
        "swarms",
        "structs",
        "base_structure.py",
    )

    if os.path.exists(base_structure_path):
        spec = importlib.util.spec_from_file_location(
            "base_structure", base_structure_path
        )
        base_structure_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base_structure_module)
        BaseStructure = base_structure_module.BaseStructure
    else:
        raise ImportError(
            f"Could not find base_structure.py at {base_structure_path}"
        ) from e

logger.remove()
logger.add(lambda msg: None, level="ERROR")


class TestStructure(BaseStructure):
    def run(self, task: str = "test"):
        return f"Processed: {task}"


def test_base_structure_initialization():
    """Test BaseStructure initialization."""
    try:
        # Set environment variable for workspace_dir
        original_value = os.environ.get("WORKSPACE_DIR")
        test_workspace = "/tmp/test_workspace"
        os.environ["WORKSPACE_DIR"] = test_workspace

        structure = BaseStructure()
        assert (
            structure is not None
        ), "BaseStructure should not be None"
        assert structure.name is None, "Default name should be None"
        assert (
            structure.description is None
        ), "Default description should be None"
        assert (
            structure.save_metadata_on is True
        ), "save_metadata_on should default to True"
        assert (
            structure.save_artifact_path == "./artifacts"
        ), "Default artifact path should be set"
        assert (
            structure.save_metadata_path == "./metadata"
        ), "Default metadata path should be set"
        assert (
            structure.save_error_path == "./errors"
        ), "Default error path should be set"
        assert (
            structure.workspace_dir == test_workspace
        ), "Workspace dir should be set from environment variable"

        structure2 = BaseStructure(
            name="TestStructure",
            description="Test description",
            save_metadata_on=False,
            save_artifact_path="/tmp/artifacts",
            save_metadata_path="/tmp/metadata",
            save_error_path="/tmp/errors",
        )

        # Restore original value
        if original_value is None:
            os.environ.pop("WORKSPACE_DIR", None)
        else:
            os.environ["WORKSPACE_DIR"] = original_value
        assert (
            structure2.name == "TestStructure"
        ), "Custom name should be set"
        assert (
            structure2.description == "Test description"
        ), "Custom description should be set"
        assert (
            structure2.save_metadata_on is False
        ), "save_metadata_on should be False"
        assert (
            structure2.save_artifact_path == "/tmp/artifacts"
        ), "Custom artifact path should be set"

        logger.info("✓ BaseStructure initialization test passed")

    except Exception as e:
        logger.error(
            f"Error in test_base_structure_initialization: {str(e)}"
        )
        raise


def test_save_and_load_file():
    """Test saving and loading files."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = BaseStructure(name="TestFileOps")
            test_file = os.path.join(tmpdir, "test_data.json")
            test_data = {
                "key": "value",
                "number": 42,
                "list": [1, 2, 3],
            }

            structure.save_to_file(test_data, test_file)

            assert os.path.exists(test_file), "File should be created"

            loaded_data = structure.load_from_file(test_file)

            assert (
                loaded_data is not None
            ), "Loaded data should not be None"
            assert isinstance(
                loaded_data, dict
            ), "Loaded data should be a dict"
            assert loaded_data["key"] == "value", "Data should match"
            assert loaded_data["number"] == 42, "Number should match"
            assert loaded_data["list"] == [
                1,
                2,
                3,
            ], "List should match"

            logger.info("✓ Save and load file test passed")

    except Exception as e:
        logger.error(f"Error in test_save_and_load_file: {str(e)}")
        raise


def test_save_and_load_metadata():
    """Test saving and loading metadata."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = BaseStructure(
                name="TestMetadata", save_metadata_path=tmpdir
            )
            metadata = {
                "timestamp": "2024-01-01",
                "status": "active",
                "count": 5,
            }

            structure.save_metadata(metadata)

            metadata_file = os.path.join(
                tmpdir, "TestMetadata_metadata.json"
            )
            assert os.path.exists(
                metadata_file
            ), "Metadata file should be created"

            loaded_metadata = structure.load_metadata()

            assert (
                loaded_metadata is not None
            ), "Loaded metadata should not be None"
            assert isinstance(
                loaded_metadata, dict
            ), "Metadata should be a dict"
            assert (
                loaded_metadata["status"] == "active"
            ), "Metadata should match"
            assert loaded_metadata["count"] == 5, "Count should match"

            logger.info("✓ Save and load metadata test passed")

    except Exception as e:
        logger.error(
            f"Error in test_save_and_load_metadata: {str(e)}"
        )
        raise


def test_save_and_load_artifact():
    """Test saving and loading artifacts."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = BaseStructure(
                name="TestArtifact", save_artifact_path=tmpdir
            )
            artifact = {"result": "success", "data": [1, 2, 3, 4, 5]}

            structure.save_artifact(artifact, "test_artifact")

            artifact_file = os.path.join(tmpdir, "test_artifact.json")
            assert os.path.exists(
                artifact_file
            ), "Artifact file should be created"

            loaded_artifact = structure.load_artifact("test_artifact")

            assert (
                loaded_artifact is not None
            ), "Loaded artifact should not be None"
            assert isinstance(
                loaded_artifact, dict
            ), "Artifact should be a dict"
            assert (
                loaded_artifact["result"] == "success"
            ), "Artifact result should match"
            assert (
                len(loaded_artifact["data"]) == 5
            ), "Artifact data should match"

            logger.info("✓ Save and load artifact test passed")

    except Exception as e:
        logger.error(
            f"Error in test_save_and_load_artifact: {str(e)}"
        )
        raise


def test_log_error():
    """Test error logging."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = BaseStructure(
                name="TestErrorLog", save_error_path=tmpdir
            )
            error_message = "Test error message"

            structure.log_error(error_message)

            error_file = os.path.join(
                tmpdir, "TestErrorLog_errors.log"
            )
            assert os.path.exists(
                error_file
            ), "Error log file should be created"

            with open(error_file, "r") as f:
                content = f.read()
                assert (
                    error_message in content
                ), "Error message should be in log"

            structure.log_error("Another error")

            with open(error_file, "r") as f:
                content = f.read()
                assert (
                    "Another error" in content
                ), "Second error should be in log"

            logger.info("✓ Log error test passed")

    except Exception as e:
        logger.error(f"Error in test_log_error: {str(e)}")
        raise


def test_log_event():
    """Test event logging."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = BaseStructure(
                name="TestEventLog", save_metadata_path=tmpdir
            )
            event_message = "Test event occurred"

            structure.log_event(event_message, "INFO")

            event_file = os.path.join(
                tmpdir, "TestEventLog_events.log"
            )
            assert os.path.exists(
                event_file
            ), "Event log file should be created"

            with open(event_file, "r") as f:
                content = f.read()
                assert (
                    event_message in content
                ), "Event message should be in log"
                assert (
                    "INFO" in content
                ), "Event type should be in log"

            structure.log_event("Warning event", "WARNING")

            with open(event_file, "r") as f:
                content = f.read()
                assert (
                    "WARNING" in content
                ), "Warning type should be in log"

            logger.info("✓ Log event test passed")

    except Exception as e:
        logger.error(f"Error in test_log_event: {str(e)}")
        raise


def test_compress_and_decompress_data():
    """Test data compression and decompression."""
    try:
        structure = BaseStructure()
        test_data = {"key": "value", "large_data": "x" * 1000}

        compressed = structure.compress_data(test_data)

        assert (
            compressed is not None
        ), "Compressed data should not be None"
        assert isinstance(
            compressed, bytes
        ), "Compressed data should be bytes"
        assert len(compressed) < len(
            json.dumps(test_data).encode()
        ), "Compressed should be smaller"

        decompressed = structure.decompres_data(compressed)

        assert (
            decompressed is not None
        ), "Decompressed data should not be None"
        assert isinstance(
            decompressed, dict
        ), "Decompressed data should be a dict"
        assert (
            decompressed["key"] == "value"
        ), "Decompressed data should match"
        assert (
            len(decompressed["large_data"]) == 1000
        ), "Large data should match"

        logger.info("✓ Compress and decompress data test passed")

    except Exception as e:
        logger.error(
            f"Error in test_compress_and_decompress_data: {str(e)}"
        )
        raise


def test_to_dict():
    """Test converting structure to dictionary."""
    try:
        structure = BaseStructure(
            name="TestDict", description="Test description"
        )

        structure_dict = structure.to_dict()

        assert (
            structure_dict is not None
        ), "Dictionary should not be None"
        assert isinstance(
            structure_dict, dict
        ), "Should return a dict"
        assert (
            structure_dict["name"] == "TestDict"
        ), "Name should be in dict"
        assert (
            structure_dict["description"] == "Test description"
        ), "Description should be in dict"

        logger.info("✓ To dict test passed")

    except Exception as e:
        logger.error(f"Error in test_to_dict: {str(e)}")
        raise


def test_to_json():
    """Test converting structure to JSON."""
    try:
        structure = BaseStructure(
            name="TestJSON", description="Test JSON description"
        )

        json_output = structure.to_json()

        assert (
            json_output is not None
        ), "JSON output should not be None"
        assert isinstance(json_output, str), "Should return a string"
        assert "TestJSON" in json_output, "Name should be in JSON"
        assert (
            "Test JSON description" in json_output
        ), "Description should be in JSON"

        parsed = json.loads(json_output)
        assert isinstance(parsed, dict), "Should be valid JSON dict"

        logger.info("✓ To JSON test passed")

    except Exception as e:
        logger.error(f"Error in test_to_json: {str(e)}")
        raise


def test_to_yaml():
    """Test converting structure to YAML."""
    try:
        structure = BaseStructure(
            name="TestYAML", description="Test YAML description"
        )

        yaml_output = structure.to_yaml()

        assert (
            yaml_output is not None
        ), "YAML output should not be None"
        assert isinstance(yaml_output, str), "Should return a string"
        assert "TestYAML" in yaml_output, "Name should be in YAML"

        logger.info("✓ To YAML test passed")

    except Exception as e:
        logger.error(f"Error in test_to_yaml: {str(e)}")
        raise


def test_to_toml():
    """Test converting structure to TOML."""
    try:
        structure = BaseStructure(
            name="TestTOML", description="Test TOML description"
        )

        toml_output = structure.to_toml()

        assert (
            toml_output is not None
        ), "TOML output should not be None"
        assert isinstance(toml_output, str), "Should return a string"

        logger.info("✓ To TOML test passed")

    except Exception as e:
        logger.error(f"Error in test_to_toml: {str(e)}")
        raise


def test_run_async():
    """Test async run method."""
    try:
        structure = TestStructure(name="TestAsync")

        async def run_test():
            result = await structure.run_async("test_task")
            return result

        result = asyncio.run(run_test())

        assert result is not None, "Async result should not be None"

        logger.info("✓ Run async test passed")

    except Exception as e:
        logger.error(f"Error in test_run_async: {str(e)}")
        raise


def test_save_metadata_async():
    """Test async save metadata."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = BaseStructure(
                name="TestAsyncMetadata", save_metadata_path=tmpdir
            )
            metadata = {"async": "test", "value": 123}

            async def save_test():
                await structure.save_metadata_async(metadata)

            asyncio.run(save_test())

            loaded = structure.load_metadata()

            assert (
                loaded is not None
            ), "Loaded metadata should not be None"
            assert loaded["async"] == "test", "Metadata should match"

            logger.info("✓ Save metadata async test passed")

    except Exception as e:
        logger.error(f"Error in test_save_metadata_async: {str(e)}")
        raise


def test_load_metadata_async():
    """Test async load metadata."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = BaseStructure(
                name="TestAsyncLoad", save_metadata_path=tmpdir
            )
            metadata = {"load": "async", "number": 456}
            structure.save_metadata(metadata)

            async def load_test():
                return await structure.load_metadata_async()

            loaded = asyncio.run(load_test())

            assert (
                loaded is not None
            ), "Loaded metadata should not be None"
            assert loaded["load"] == "async", "Metadata should match"

            logger.info("✓ Load metadata async test passed")

    except Exception as e:
        logger.error(f"Error in test_load_metadata_async: {str(e)}")
        raise


def test_save_artifact_async():
    """Test async save artifact."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = BaseStructure(
                name="TestAsyncArtifact", save_artifact_path=tmpdir
            )
            artifact = {"async_artifact": True, "data": [1, 2, 3]}

            async def save_test():
                await structure.save_artifact_async(
                    artifact, "async_artifact"
                )

            asyncio.run(save_test())

            loaded = structure.load_artifact("async_artifact")

            assert (
                loaded is not None
            ), "Loaded artifact should not be None"
            assert (
                loaded["async_artifact"] is True
            ), "Artifact should match"

            logger.info("✓ Save artifact async test passed")

    except Exception as e:
        logger.error(f"Error in test_save_artifact_async: {str(e)}")
        raise


def test_load_artifact_async():
    """Test async load artifact."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = BaseStructure(
                name="TestAsyncLoadArtifact",
                save_artifact_path=tmpdir,
            )
            artifact = {"load_async": True, "items": ["a", "b", "c"]}
            structure.save_artifact(artifact, "load_async_artifact")

            async def load_test():
                return await structure.load_artifact_async(
                    "load_async_artifact"
                )

            loaded = asyncio.run(load_test())

            assert (
                loaded is not None
            ), "Loaded artifact should not be None"
            assert (
                loaded["load_async"] is True
            ), "Artifact should match"

            logger.info("✓ Load artifact async test passed")

    except Exception as e:
        logger.error(f"Error in test_load_artifact_async: {str(e)}")
        raise


def test_asave_and_aload_from_file():
    """Test async save and load from file."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = BaseStructure()
            test_file = os.path.join(tmpdir, "async_test.json")
            test_data = {"async": "file", "test": True}

            async def save_and_load():
                await structure.asave_to_file(test_data, test_file)
                return await structure.aload_from_file(test_file)

            loaded = asyncio.run(save_and_load())

            assert (
                loaded is not None
            ), "Loaded data should not be None"
            assert loaded["async"] == "file", "Data should match"
            assert loaded["test"] is True, "Boolean should match"

            logger.info("✓ Async save and load from file test passed")

    except Exception as e:
        logger.error(
            f"Error in test_asave_and_aload_from_file: {str(e)}"
        )
        raise


def test_run_in_thread():
    """Test running in thread."""
    try:
        structure = TestStructure(name="TestThread")

        future = structure.run_in_thread("thread_task")
        result = future.result()

        assert result is not None, "Thread result should not be None"

        logger.info("✓ Run in thread test passed")

    except Exception as e:
        logger.error(f"Error in test_run_in_thread: {str(e)}")
        raise


def test_save_metadata_in_thread():
    """Test saving metadata in thread."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = BaseStructure(
                name="TestThreadMetadata", save_metadata_path=tmpdir
            )
            metadata = {"thread": "test", "value": 789}

            future = structure.save_metadata_in_thread(metadata)
            future.result()

            loaded = structure.load_metadata()

            assert (
                loaded is not None
            ), "Loaded metadata should not be None"
            assert loaded["thread"] == "test", "Metadata should match"

            logger.info("✓ Save metadata in thread test passed")

    except Exception as e:
        logger.error(
            f"Error in test_save_metadata_in_thread: {str(e)}"
        )
        raise


def test_run_batched():
    """Test batched execution."""
    try:
        structure = TestStructure(name="TestBatched")
        batched_data = ["task1", "task2", "task3", "task4", "task5"]

        results = structure.run_batched(batched_data, batch_size=3)

        assert results is not None, "Results should not be None"
        assert isinstance(results, list), "Results should be a list"
        assert len(results) == 5, "Should have 5 results"

        for result in results:
            assert (
                result is not None
            ), "Each result should not be None"
            assert (
                "Processed:" in result
            ), "Result should contain processed message"

        logger.info("✓ Run batched test passed")

    except Exception as e:
        logger.error(f"Error in test_run_batched: {str(e)}")
        raise


def test_load_config():
    """Test loading configuration."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = BaseStructure()
            config_file = os.path.join(tmpdir, "config.json")
            config_data = {"setting1": "value1", "setting2": 42}

            structure.save_to_file(config_data, config_file)

            loaded_config = structure.load_config(config_file)

            assert (
                loaded_config is not None
            ), "Loaded config should not be None"
            assert isinstance(
                loaded_config, dict
            ), "Config should be a dict"
            assert (
                loaded_config["setting1"] == "value1"
            ), "Config should match"
            assert (
                loaded_config["setting2"] == 42
            ), "Config number should match"

            logger.info("✓ Load config test passed")

    except Exception as e:
        logger.error(f"Error in test_load_config: {str(e)}")
        raise


def test_backup_data():
    """Test backing up data."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = BaseStructure()
            backup_path = os.path.join(tmpdir, "backups")
            os.makedirs(backup_path, exist_ok=True)

            backup_data = {"backup": "test", "items": [1, 2, 3]}

            structure.backup_data(backup_data, backup_path)

            backup_files = os.listdir(backup_path)
            assert (
                len(backup_files) > 0
            ), "Backup file should be created"

            backup_file = os.path.join(backup_path, backup_files[0])
            loaded_backup = structure.load_from_file(backup_file)

            assert (
                loaded_backup is not None
            ), "Loaded backup should not be None"
            assert (
                loaded_backup["backup"] == "test"
            ), "Backup data should match"

            logger.info("✓ Backup data test passed")

    except Exception as e:
        logger.error(f"Error in test_backup_data: {str(e)}")
        raise


def test_monitor_resources():
    """Test resource monitoring."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = BaseStructure(
                name="TestResources", save_metadata_path=tmpdir
            )

            structure.monitor_resources()

            event_file = os.path.join(
                tmpdir, "TestResources_events.log"
            )
            assert os.path.exists(
                event_file
            ), "Event log should be created"

            with open(event_file, "r") as f:
                content = f.read()
                assert (
                    "Resource usage" in content
                ), "Resource usage should be logged"
                assert "Memory" in content, "Memory should be logged"
                assert "CPU" in content, "CPU should be logged"

            logger.info("✓ Monitor resources test passed")

    except Exception as e:
        logger.error(f"Error in test_monitor_resources: {str(e)}")
        raise


def test_run_with_resources():
    """Test running with resource monitoring."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = TestStructure(
                name="TestRunResources", save_metadata_path=tmpdir
            )

            result = structure.run_with_resources("monitored_task")

            assert result is not None, "Result should not be None"

            event_file = os.path.join(
                tmpdir, "TestRunResources_events.log"
            )
            assert os.path.exists(
                event_file
            ), "Event log should be created"

            logger.info("✓ Run with resources test passed")

    except Exception as e:
        logger.error(f"Error in test_run_with_resources: {str(e)}")
        raise


def test_run_with_resources_batched():
    """Test batched execution with resource monitoring."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = TestStructure(
                name="TestBatchedResources", save_metadata_path=tmpdir
            )
            batched_data = ["task1", "task2", "task3"]

            results = structure.run_with_resources_batched(
                batched_data, batch_size=2
            )

            assert results is not None, "Results should not be None"
            assert isinstance(
                results, list
            ), "Results should be a list"
            assert len(results) == 3, "Should have 3 results"

            event_file = os.path.join(
                tmpdir, "TestBatchedResources_events.log"
            )
            assert os.path.exists(
                event_file
            ), "Event log should be created"

            logger.info("✓ Run with resources batched test passed")

    except Exception as e:
        logger.error(
            f"Error in test_run_with_resources_batched: {str(e)}"
        )
        raise


def test_serialize_callable():
    """Test serializing callable attributes."""
    try:

        def test_function():
            """Test function docstring."""
            pass

        structure = BaseStructure()
        serialized = structure._serialize_callable(test_function)

        assert (
            serialized is not None
        ), "Serialized callable should not be None"
        assert isinstance(serialized, dict), "Should return a dict"
        assert "name" in serialized, "Should have name"
        assert "doc" in serialized, "Should have doc"
        assert (
            serialized["name"] == "test_function"
        ), "Name should match"

        logger.info("✓ Serialize callable test passed")

    except Exception as e:
        logger.error(f"Error in test_serialize_callable: {str(e)}")
        raise


def test_serialize_attr():
    """Test serializing attributes."""
    try:
        structure = BaseStructure()

        serialized_str = structure._serialize_attr(
            "test_attr", "test_value"
        )
        assert (
            serialized_str == "test_value"
        ), "String should serialize correctly"

        serialized_dict = structure._serialize_attr(
            "test_attr", {"key": "value"}
        )
        assert serialized_dict == {
            "key": "value"
        }, "Dict should serialize correctly"

        def test_func():
            pass

        serialized_func = structure._serialize_attr(
            "test_func", test_func
        )
        assert isinstance(
            serialized_func, dict
        ), "Function should serialize to dict"

        logger.info("✓ Serialize attr test passed")

    except Exception as e:
        logger.error(f"Error in test_serialize_attr: {str(e)}")
        raise


if __name__ == "__main__":
    import sys

    test_dict = {
        "test_base_structure_initialization": test_base_structure_initialization,
        "test_save_and_load_file": test_save_and_load_file,
        "test_save_and_load_metadata": test_save_and_load_metadata,
        "test_save_and_load_artifact": test_save_and_load_artifact,
        "test_log_error": test_log_error,
        "test_log_event": test_log_event,
        "test_compress_and_decompress_data": test_compress_and_decompress_data,
        "test_to_dict": test_to_dict,
        "test_to_json": test_to_json,
        "test_to_yaml": test_to_yaml,
        "test_to_toml": test_to_toml,
        "test_run_async": test_run_async,
        "test_save_metadata_async": test_save_metadata_async,
        "test_load_metadata_async": test_load_metadata_async,
        "test_save_artifact_async": test_save_artifact_async,
        "test_load_artifact_async": test_load_artifact_async,
        "test_asave_and_aload_from_file": test_asave_and_aload_from_file,
        "test_run_in_thread": test_run_in_thread,
        "test_save_metadata_in_thread": test_save_metadata_in_thread,
        "test_run_batched": test_run_batched,
        "test_load_config": test_load_config,
        "test_backup_data": test_backup_data,
        "test_monitor_resources": test_monitor_resources,
        "test_run_with_resources": test_run_with_resources,
        "test_run_with_resources_batched": test_run_with_resources_batched,
        "test_serialize_callable": test_serialize_callable,
        "test_serialize_attr": test_serialize_attr,
    }

    if len(sys.argv) > 1:
        requested_tests = []
        for test_name in sys.argv[1:]:
            if test_name in test_dict:
                requested_tests.append(test_dict[test_name])
            elif test_name == "all" or test_name == "--all":
                requested_tests = list(test_dict.values())
                break
            else:
                print(f"⚠ Warning: Test '{test_name}' not found.")
                print(
                    f"Available tests: {', '.join(test_dict.keys())}"
                )
                sys.exit(1)

        tests_to_run = requested_tests
    else:
        tests_to_run = list(test_dict.values())

    if len(tests_to_run) == 1:
        print(f"Running: {tests_to_run[0].__name__}")
    else:
        print(f"Running {len(tests_to_run)} test(s)...")

    passed = 0
    failed = 0

    for test_func in tests_to_run:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {test_func.__name__}")
            print(f"{'='*60}")
            test_func()
            print(f"✓ PASSED: {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {test_func.__name__}")
            print(f"  Error: {str(e)}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Test Summary: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if len(sys.argv) == 1:
        print("\n💡 Tip: Run a specific test with:")
        print(
            "   python test_base_structure.py test_base_structure_initialization"
        )
        print("\n   Or use pytest:")
        print("   pytest test_base_structure.py")
        print(
            "   pytest test_base_structure.py::test_base_structure_initialization"
        )
