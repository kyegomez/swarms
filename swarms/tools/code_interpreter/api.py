from ..tool import Tool


class CodeInterpreter:
    def __init__(self, timeout=300):
        self.globals = {}
        self.locals = {}
        self.timeout = timeout

    def execute_code(self, code):
        try:
            # Wrap the code in an eval() call to return the result
            wrapped_code = f"__result__ = eval({repr(code)}, globals(), locals())"
            exec(wrapped_code, self.globals, self.locals)
            return self.locals.get("__result__", None)
        except Exception as e:
            try:
                # If eval fails, attempt to exec the code without returning a result
                exec(code, self.globals, self.locals)
                return "Code executed successfully."
            except Exception as e:
                return f"Error: {str(e)}"

    def reset_session(self):
        self.globals = {}
        self.locals = {}


def build_tool(config) -> Tool:
    tool = Tool(
        "Python Code Interpreter Tool",
        "Execute Python Codes",
        name_for_model="code_interpreter",
        description_for_model="Plugin for executing python codes",
        logo_url=None,
        contact_email=None,
        legal_info_url=None,
    )

    # Usage example
    interpreter = CodeInterpreter()

    @tool.get("/execute_code")
    def execute_python_code(code: str):
        """execute Python expressions with Python Interpreter, can be used as a simple calculator e.g., "(123 + 234) / 23 * 19" """
        return interpreter.execute_code(code)

    return tool
