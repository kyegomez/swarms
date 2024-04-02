import os
import subprocess
import tempfile


class CodeExecutor:
    """
    A class for executing code snippets.

    Args:
        code (str, optional): The code snippet to be executed. Defaults to None.

    Methods:
        is_python_code(code: str = None) -> bool:
            Checks if the given code is Python code.

        run_python(code: str = None) -> str:
            Executes the given Python code and returns the output.

        run(code: str = None) -> str:
            Executes the given code and returns the output.

        __call__() -> str:
            Executes the code and returns the output.
    """

    def __init__(self):
        self.code = None

    def run_python(self, code: str = None) -> str:
        """
        Executes the given Python code and returns the output.

        Args:
            code (str, optional): The Python code to be executed. Defaults to None.

        Returns:
            str: The output of the code execution.
        """
        code = code or self.code
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                suffix=".py", delete=False
            ) as temp:
                temp.write(code.encode())
                temp_filename = temp.name

            # Execute the temporary file
            output = subprocess.check_output(
                f"python {temp_filename}",
                shell=True,
            )

            # Delete the temporary file
            os.remove(temp_filename)

            return output.decode("utf-8")
        except subprocess.CalledProcessError as error:
            return error.output.decode("utf-8")
        except Exception as error:
            return str(error)

    def run(self, code: str = None) -> str:
        """
        Executes the given code and returns the output.

        Args:
            code (str, optional): The code to be executed. Defaults to None.

        Returns:
            str: The output of the code execution.
        """
        try:
            output = subprocess.check_output(
                code,
                shell=True,
            )
            return output.decode("utf-8")
        except subprocess.CalledProcessError as e:
            return e.output.decode("utf-8")
        except Exception as e:
            return str(e)

    def __call__(self, task: str, *args, **kwargs) -> str:
        """
        Executes the code and returns the output.

        Returns:
            str: The output of the code execution.
        """
        return self.run(task, *args, **kwargs)


# model = CodeExecutor()
# out = model.run("python3")
# print(out)
