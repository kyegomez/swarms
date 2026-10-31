import pytest

from swarms.utils.docstring_parser import (
    DocstringParam,
    parse,
)


class TestDocstringParser:
    """Test cases for the docstring parser functionality."""

    def test_empty_docstring(self):
        """Test parsing of empty docstring."""
        result = parse("")
        assert result.short_description is None
        assert result.params == []

    def test_none_docstring(self):
        """Test parsing of None docstring."""
        result = parse(None)
        assert result.short_description is None
        assert result.params == []

    def test_whitespace_only_docstring(self):
        """Test parsing of whitespace-only docstring."""
        result = parse("   \n  \t  \n  ")
        assert result.short_description is None
        assert result.params == []

    def test_simple_docstring_no_args(self):
        """Test parsing of simple docstring without Args section."""
        docstring = """
        This is a simple function.
        
        Returns:
            str: A simple string
        """
        result = parse(docstring)
        assert (
            result.short_description == "This is a simple function."
        )
        assert result.params == []

    def test_docstring_with_args(self):
        """Test parsing of docstring with Args section."""
        docstring = """
        This is a test function.

        Args:
            param1 (str): First parameter
            param2 (int): Second parameter
            param3 (bool, optional): Third parameter with default

        Returns:
            str: Return value description
        """
        result = parse(docstring)
        assert result.short_description == "This is a test function."
        assert len(result.params) == 3
        assert result.params[0] == DocstringParam(
            "param1", "First parameter"
        )
        assert result.params[1] == DocstringParam(
            "param2", "Second parameter"
        )
        assert result.params[2] == DocstringParam(
            "param3", "Third parameter with default"
        )

    def test_docstring_with_parameters_section(self):
        """Test parsing of docstring with Parameters section."""
        docstring = """
        Another test function.

        Parameters:
            name (str): The name parameter
            age (int): The age parameter

        Returns:
            None: Nothing is returned
        """
        result = parse(docstring)
        assert result.short_description == "Another test function."
        assert len(result.params) == 2
        assert result.params[0] == DocstringParam(
            "name", "The name parameter"
        )
        assert result.params[1] == DocstringParam(
            "age", "The age parameter"
        )

    def test_docstring_with_multiline_param_description(self):
        """Test parsing of docstring with multiline parameter descriptions."""
        docstring = """
        Function with multiline descriptions.

        Args:
            param1 (str): This is a very long description
                that spans multiple lines and should be
                properly concatenated.
            param2 (int): Short description

        Returns:
            str: Result
        """
        result = parse(docstring)
        assert (
            result.short_description
            == "Function with multiline descriptions."
        )
        assert len(result.params) == 2
        expected_desc = "This is a very long description that spans multiple lines and should be properly concatenated."
        assert result.params[0] == DocstringParam(
            "param1", expected_desc
        )
        assert result.params[1] == DocstringParam(
            "param2", "Short description"
        )

    def test_docstring_without_type_annotations(self):
        """Test parsing of docstring without type annotations."""
        docstring = """
        Function without type annotations.

        Args:
            param1: First parameter without type
            param2: Second parameter without type

        Returns:
            str: Result
        """
        result = parse(docstring)
        assert (
            result.short_description
            == "Function without type annotations."
        )
        assert len(result.params) == 2
        assert result.params[0] == DocstringParam(
            "param1", "First parameter without type"
        )
        assert result.params[1] == DocstringParam(
            "param2", "Second parameter without type"
        )

    def test_pydantic_style_docstring(self):
        """Test parsing of Pydantic-style docstring."""
        docstring = """
        Convert a Pydantic model to a dictionary representation of functions.

        Args:
            pydantic_type (type[BaseModel]): The Pydantic model type to convert.

        Returns:
            dict[str, Any]: A dictionary representation of the functions.
        """
        result = parse(docstring)
        assert (
            result.short_description
            == "Convert a Pydantic model to a dictionary representation of functions."
        )
        assert len(result.params) == 1
        assert result.params[0] == DocstringParam(
            "pydantic_type", "The Pydantic model type to convert."
        )

    def test_docstring_with_various_sections(self):
        """Test parsing of docstring with multiple sections."""
        docstring = """
        Complex function with multiple sections.

        Args:
            input_data (dict): Input data dictionary
            validate (bool): Whether to validate input

        Returns:
            dict: Processed data

        Raises:
            ValueError: If input is invalid

        Note:
            This is a note section

        Example:
            >>> result = complex_function({"key": "value"})
        """
        result = parse(docstring)
        assert (
            result.short_description
            == "Complex function with multiple sections."
        )
        assert len(result.params) == 2
        assert result.params[0] == DocstringParam(
            "input_data", "Input data dictionary"
        )
        assert result.params[1] == DocstringParam(
            "validate", "Whether to validate input"
        )

    def test_docstring_with_see_also_section(self):
        """Test parsing of docstring with See Also section."""
        docstring = """
        Function with See Also section.

        Args:
            param1 (str): First parameter

        See Also:
            related_function: For related functionality
        """
        result = parse(docstring)
        assert (
            result.short_description
            == "Function with See Also section."
        )
        assert len(result.params) == 1
        assert result.params[0] == DocstringParam(
            "param1", "First parameter"
        )

    def test_docstring_with_see_also_underscore_section(self):
        """Test parsing of docstring with See_Also section (underscore variant)."""
        docstring = """
        Function with See_Also section.

        Args:
            param1 (str): First parameter

        See_Also:
            related_function: For related functionality
        """
        result = parse(docstring)
        assert (
            result.short_description
            == "Function with See_Also section."
        )
        assert len(result.params) == 1
        assert result.params[0] == DocstringParam(
            "param1", "First parameter"
        )

    def test_docstring_with_yields_section(self):
        """Test parsing of docstring with Yields section."""
        docstring = """
        Generator function.

        Args:
            items (list): List of items to process

        Yields:
            str: Processed item
        """
        result = parse(docstring)
        assert result.short_description == "Generator function."
        assert len(result.params) == 1
        assert result.params[0] == DocstringParam(
            "items", "List of items to process"
        )

    def test_docstring_with_raises_section(self):
        """Test parsing of docstring with Raises section."""
        docstring = """
        Function that can raise exceptions.

        Args:
            value (int): Value to process

        Raises:
            ValueError: If value is negative
        """
        result = parse(docstring)
        assert (
            result.short_description
            == "Function that can raise exceptions."
        )
        assert len(result.params) == 1
        assert result.params[0] == DocstringParam(
            "value", "Value to process"
        )

    def test_docstring_with_examples_section(self):
        """Test parsing of docstring with Examples section."""
        docstring = """
        Function with examples.

        Args:
            x (int): Input value

        Examples:
            >>> result = example_function(5)
            >>> print(result)
        """
        result = parse(docstring)
        assert result.short_description == "Function with examples."
        assert len(result.params) == 1
        assert result.params[0] == DocstringParam("x", "Input value")

    def test_docstring_with_note_section(self):
        """Test parsing of docstring with Note section."""
        docstring = """
        Function with a note.

        Args:
            data (str): Input data

        Note:
            This function is deprecated
        """
        result = parse(docstring)
        assert result.short_description == "Function with a note."
        assert len(result.params) == 1
        assert result.params[0] == DocstringParam(
            "data", "Input data"
        )

    def test_docstring_with_complex_type_annotations(self):
        """Test parsing of docstring with complex type annotations."""
        docstring = """
        Function with complex types.

        Args:
            data (List[Dict[str, Any]]): Complex data structure
            callback (Callable[[str], int]): Callback function
            optional (Optional[str], optional): Optional parameter

        Returns:
            Union[str, None]: Result or None
        """
        result = parse(docstring)
        assert (
            result.short_description == "Function with complex types."
        )
        assert len(result.params) == 3
        assert result.params[0] == DocstringParam(
            "data", "Complex data structure"
        )
        assert result.params[1] == DocstringParam(
            "callback", "Callback function"
        )
        assert result.params[2] == DocstringParam(
            "optional", "Optional parameter"
        )

    def test_docstring_with_no_description(self):
        """Test parsing of docstring with no description, only Args."""
        docstring = """
        Args:
            param1 (str): First parameter
            param2 (int): Second parameter
        """
        result = parse(docstring)
        assert result.short_description is None
        assert len(result.params) == 2
        assert result.params[0] == DocstringParam(
            "param1", "First parameter"
        )
        assert result.params[1] == DocstringParam(
            "param2", "Second parameter"
        )

    def test_docstring_with_empty_args_section(self):
        """Test parsing of docstring with empty Args section."""
        docstring = """
        Function with empty Args section.

        Args:

        Returns:
            str: Result
        """
        result = parse(docstring)
        assert (
            result.short_description
            == "Function with empty Args section."
        )
        assert result.params == []

    def test_docstring_with_mixed_indentation(self):
        """Test parsing of docstring with mixed indentation."""
        docstring = """
        Function with mixed indentation.

        Args:
            param1 (str): First parameter
                with continuation
            param2 (int): Second parameter
        """
        result = parse(docstring)
        assert (
            result.short_description
            == "Function with mixed indentation."
        )
        assert len(result.params) == 2
        assert result.params[0] == DocstringParam(
            "param1", "First parameter with continuation"
        )
        assert result.params[1] == DocstringParam(
            "param2", "Second parameter"
        )

    def test_docstring_with_tab_indentation(self):
        """Test parsing of docstring with tab indentation."""
        docstring = """
        Function with tab indentation.

        Args:
        	param1 (str): First parameter
        	param2 (int): Second parameter
        """
        result = parse(docstring)
        assert (
            result.short_description
            == "Function with tab indentation."
        )
        assert len(result.params) == 2
        assert result.params[0] == DocstringParam(
            "param1", "First parameter"
        )
        assert result.params[1] == DocstringParam(
            "param2", "Second parameter"
        )


if __name__ == "__main__":
    pytest.main([__file__])
