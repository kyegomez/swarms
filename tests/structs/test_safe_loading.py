import pytest
import json
import os
import tempfile
from datetime import datetime
from uuid import UUID, uuid4
from swarms.structs.safe_loading import (
    SafeLoaderUtils,
    SafeStateManager,
)


# Test SafeLoaderUtils.is_class_instance
def test_is_class_instance_with_custom_class():
    """Test that is_class_instance detects custom class instances"""
    class CustomClass:
        def __init__(self):
            self.value = 42

    obj = CustomClass()
    assert SafeLoaderUtils.is_class_instance(obj) is True


def test_is_class_instance_with_none():
    """Test that is_class_instance returns False for None"""
    assert SafeLoaderUtils.is_class_instance(None) is False


def test_is_class_instance_with_builtin_types():
    """Test that is_class_instance returns False for built-in types"""
    assert SafeLoaderUtils.is_class_instance(42) is False
    assert SafeLoaderUtils.is_class_instance("string") is False
    assert SafeLoaderUtils.is_class_instance([1, 2, 3]) is False
    assert SafeLoaderUtils.is_class_instance({"key": "value"}) is False
    assert SafeLoaderUtils.is_class_instance(True) is False


def test_is_class_instance_with_class_itself():
    """Test that is_class_instance returns False for class itself"""
    class CustomClass:
        pass

    assert SafeLoaderUtils.is_class_instance(CustomClass) is False


# Test SafeLoaderUtils.is_safe_type
def test_is_safe_type_with_basic_types():
    """Test that is_safe_type returns True for basic safe types"""
    assert SafeLoaderUtils.is_safe_type(None) is True
    assert SafeLoaderUtils.is_safe_type(True) is True
    assert SafeLoaderUtils.is_safe_type(42) is True
    assert SafeLoaderUtils.is_safe_type(3.14) is True
    assert SafeLoaderUtils.is_safe_type("string") is True


def test_is_safe_type_with_datetime():
    """Test that is_safe_type returns True for datetime"""
    dt = datetime.now()
    assert SafeLoaderUtils.is_safe_type(dt) is True


def test_is_safe_type_with_uuid():
    """Test that is_safe_type returns True for UUID"""
    uid = uuid4()
    assert SafeLoaderUtils.is_safe_type(uid) is True


def test_is_safe_type_with_safe_list():
    """Test that is_safe_type returns True for list of safe types"""
    assert SafeLoaderUtils.is_safe_type([1, 2, 3]) is True
    assert SafeLoaderUtils.is_safe_type(["a", "b", "c"]) is True


def test_is_safe_type_with_unsafe_list():
    """Test that is_safe_type returns False for list with unsafe types"""
    class CustomClass:
        pass

    assert SafeLoaderUtils.is_safe_type([CustomClass()]) is False


def test_is_safe_type_with_safe_dict():
    """Test that is_safe_type returns True for dict with safe types"""
    assert SafeLoaderUtils.is_safe_type({"key": "value"}) is True
    assert SafeLoaderUtils.is_safe_type({"num": 42, "str": "hello"}) is True


def test_is_safe_type_with_unsafe_dict():
    """Test that is_safe_type returns False for dict with unsafe types"""
    class CustomClass:
        pass

    assert SafeLoaderUtils.is_safe_type({"obj": CustomClass()}) is False


def test_is_safe_type_with_non_string_dict_keys():
    """Test that is_safe_type returns False for dict with non-string keys"""
    assert SafeLoaderUtils.is_safe_type({1: "value"}) is False


# Test SafeLoaderUtils.get_class_attributes
def test_get_class_attributes_basic():
    """Test that get_class_attributes returns all attributes"""
    class TestClass:
        class_var = "class"

        def __init__(self):
            self.instance_var = "instance"

    obj = TestClass()
    attrs = SafeLoaderUtils.get_class_attributes(obj)
    assert "class_var" in attrs
    assert "instance_var" in attrs


def test_get_class_attributes_inheritance():
    """Test that get_class_attributes includes inherited attributes"""
    class Parent:
        parent_var = "parent"

    class Child(Parent):
        child_var = "child"

        def __init__(self):
            self.instance_var = "instance"

    obj = Child()
    attrs = SafeLoaderUtils.get_class_attributes(obj)
    assert "parent_var" in attrs
    assert "child_var" in attrs
    assert "instance_var" in attrs


# Test SafeLoaderUtils.create_state_dict
def test_create_state_dict_basic():
    """Test that create_state_dict creates dict of safe values"""
    class TestClass:
        def __init__(self):
            self.safe_value = 42
            self.safe_string = "hello"
            self._private = "private"

    obj = TestClass()
    state = SafeLoaderUtils.create_state_dict(obj)
    assert state["safe_value"] == 42
    assert state["safe_string"] == "hello"
    assert "_private" not in state


def test_create_state_dict_skips_unsafe():
    """Test that create_state_dict skips unsafe types"""
    class Inner:
        pass

    class TestClass:
        def __init__(self):
            self.safe = 42
            self.unsafe = Inner()

    obj = TestClass()
    state = SafeLoaderUtils.create_state_dict(obj)
    assert "safe" in state
    assert "unsafe" not in state


# Test SafeLoaderUtils.preserve_instances
def test_preserve_instances_basic():
    """Test that preserve_instances preserves class instances"""
    class Inner:
        def __init__(self):
            self.value = 100

    class Outer:
        def __init__(self):
            self.safe = 42
            self.instance = Inner()

    obj = Outer()
    preserved = SafeLoaderUtils.preserve_instances(obj)
    assert "instance" in preserved
    assert isinstance(preserved["instance"], Inner)
    assert "safe" not in preserved


def test_preserve_instances_skips_private():
    """Test that preserve_instances skips private attributes"""
    class Inner:
        pass

    class Outer:
        def __init__(self):
            self._private_instance = Inner()
            self.public_instance = Inner()

    obj = Outer()
    preserved = SafeLoaderUtils.preserve_instances(obj)
    assert "_private_instance" not in preserved
    assert "public_instance" in preserved


# Test SafeStateManager.save_state and load_state
def test_save_and_load_state():
    """Test that save_state and load_state work correctly"""
    class TestClass:
        def __init__(self):
            self.value = 42
            self.text = "hello"

    obj = TestClass()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "state.json")

        # Save state
        SafeStateManager.save_state(obj, file_path)

        # Verify file exists
        assert os.path.exists(file_path)

        # Load state into new object
        new_obj = TestClass()
        new_obj.value = 0  # Change value to test loading
        new_obj.text = ""

        SafeStateManager.load_state(new_obj, file_path)

        assert new_obj.value == 42
        assert new_obj.text == "hello"


def test_save_state_creates_directory():
    """Test that save_state creates directory if it doesn't exist"""
    class TestClass:
        def __init__(self):
            self.value = 42

    obj = TestClass()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "subdir", "state.json")

        SafeStateManager.save_state(obj, file_path)

        assert os.path.exists(file_path)


def test_load_state_preserves_instances():
    """Test that load_state preserves existing class instances"""
    class Inner:
        def __init__(self, val):
            self.val = val

    class Outer:
        def __init__(self):
            self.safe = 42
            self.instance = Inner(100)

    obj = Outer()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "state.json")

        # Save state
        SafeStateManager.save_state(obj, file_path)

        # Create new object with different values
        new_obj = Outer()
        new_obj.safe = 0
        original_instance = new_obj.instance

        # Load state
        SafeStateManager.load_state(new_obj, file_path)

        # Safe value should be updated
        assert new_obj.safe == 42

        # Instance should be preserved (same object)
        assert new_obj.instance is original_instance
        assert new_obj.instance.val == 100


def test_load_state_file_not_found():
    """Test that load_state raises FileNotFoundError for missing file"""
    class TestClass:
        pass

    obj = TestClass()

    with pytest.raises(FileNotFoundError, match="State file not found"):
        SafeStateManager.load_state(obj, "/nonexistent/path.json")


def test_save_state_with_datetime():
    """Test that save_state handles datetime objects"""
    class TestClass:
        def __init__(self):
            self.created_at = datetime(2024, 1, 1, 12, 0, 0)

    obj = TestClass()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "state.json")

        # Should not raise an error
        SafeStateManager.save_state(obj, file_path)
        assert os.path.exists(file_path)


def test_save_state_with_uuid():
    """Test that save_state handles UUID objects"""
    class TestClass:
        def __init__(self):
            self.id = uuid4()

    obj = TestClass()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "state.json")

        # Should not raise an error
        SafeStateManager.save_state(obj, file_path)
        assert os.path.exists(file_path)


def test_is_safe_type_with_tuple():
    """Test that is_safe_type returns True for tuple of safe types"""
    assert SafeLoaderUtils.is_safe_type((1, 2, 3)) is True
    assert SafeLoaderUtils.is_safe_type(("a", "b")) is True
