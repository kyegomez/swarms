import queue
import subprocess
import threading
import time
import traceback
from swarms.utils.loguru_logger import logger
from swarms.utils.terminal_output import terminal, OutputChunk


class SubprocessCodeInterpreter:
    """
    SubprocessCodeinterpreter is a base class for code interpreters that run code in a subprocess.


    Attributes:
        start_cmd (str): The command to start the subprocess. Should be a string that can be split by spaces.
        process (subprocess.Popen): The subprocess that is running the code.
        debug_mode (bool): Whether to print debug statements.
        output_queue (queue.Queue): A queue that is filled with output from the subprocess.
        done (threading.Event): An event that is set when the subprocess is done running code.

    Example:
    """

    def __init__(
        self,
        start_cmd: str = "python3",
        debug_mode: bool = False,
        max_retries: int = 3,
        verbose: bool = False,
        retry_count: int = 0,
        *args,
        **kwargs,
    ):
        self.process = None
        self.start_cmd = start_cmd
        self.debug_mode = debug_mode
        self.max_retries = max_retries
        self.verbose = verbose
        self.retry_count = retry_count
        self.output_queue = queue.Queue()
        self.done = threading.Event()
        self.line_postprocessor = lambda x: x

    def detect_active_line(self, line):
        """Detect if the line is an active line

        Args:
            line (_type_): _description_

        Returns:
            _type_: _description_
        """
        return None

    def detect_end_of_execution(self, line):
        """detect if the line is an end of execution line

        Args:
            line (_type_): _description_

        Returns:
            _type_: _description_
        """
        return None

    def line_postprocessor(self, line):
        """Line postprocessor

        Args:
            line (_type_): _description_

        Returns:
            _type_: _description_
        """
        return line

    def preprocess_code(self, code):
        """
        This needs to insert an end_of_execution marker of some kind,
        which can be detected by detect_end_of_execution.

        Optionally, add active line markers for detect_active_line.
        """
        return code

    def terminate(self):
        """terminate the subprocess"""
        self.process.terminate()

    def start_process(self):
        """start the subprocess"""
        if self.process:
            self.terminate()

        logger.info(
            f"Starting subprocess with command: {self.start_cmd}"
        )
        self.process = subprocess.Popen(
            self.start_cmd.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            universal_newlines=True,
        )
        threading.Thread(
            target=self.handle_stream_output,
            args=(self.process.stdout, False),
            daemon=True,
        ).start()
        threading.Thread(
            target=self.handle_stream_output,
            args=(self.process.stderr, True),
            daemon=True,
        ).start()

        return self.process

    def run(self, code: str):
        """Run the code in the subprocess

        Args:
            code (str): _description_

        Yields:
            _type_: _description_
        """

        # Setup
        logger.info("Running code in subprocess")
        try:
            code = self.preprocess_code(code)
            if not self.process:
                self.start_process()
        except BaseException:
            yield {"output": traceback.format_exc()}
            return

        while self.retry_count <= self.max_retries:
            if self.debug_mode:
                print(f"Running code:\n{code}\n---")

            self.done.clear()

            try:
                self.process.stdin.write(code + "\n")
                self.process.stdin.flush()
                break
            except BaseException:
                if self.retry_count != 0:
                    # For UX, I like to hide this if it happens once. Obviously feels better to not see errors
                    # Most of the time it doesn't matter, but we should figure out why it happens frequently with:
                    # applescript
                    yield {"output": traceback.format_exc()}
                    yield {
                        "output": (
                            "Retrying..."
                            f" ({self.retry_count}/{self.max_retries})"
                        )
                    }
                    yield {"output": "Restarting process."}

                self.start_process()

                self.retry_count += 1
                if self.retry_count > self.max_retries:
                    yield {
                        "output": (
                            "Maximum retries reached. Could not"
                            " execute code."
                        )
                    }
                    return

        while True:
            if not self.output_queue.empty():
                yield self.output_queue.get()
            else:
                time.sleep(0.1)
            try:
                output = self.output_queue.get(
                    timeout=0.3
                )  # Waits for 0.3 seconds
                yield output
            except queue.Empty:
                if self.done.is_set():
                    # Try to yank 3 more times from it... maybe there's something in there...
                    # (I don't know if this actually helps. Maybe we just need to yank 1 more time)
                    for _ in range(3):
                        if not self.output_queue.empty():
                            yield self.output_queue.get()
                        time.sleep(0.2)
                    break

    def handle_stream_output(self, stream, is_error_stream):
        """Handle the output from the subprocess with enhanced formatting

        Args:
            stream (_type_): The stream to read from
            is_error_stream (bool): Whether this is an error stream
        """
        for line in iter(stream.readline, ""):
            if self.debug_mode:
                terminal.status_panel(f"Debug: {line}", "info")

            line = self.line_postprocessor(line)

            if line is None:
                continue

            chunk_type = "error" if is_error_stream else "text"
            
            if self.detect_active_line(line):
                active_line = self.detect_active_line(line)
                self.output_queue.put(OutputChunk(
                    content=f"Active line: {active_line}",
                    type="info",
                    metadata={"active_line": active_line}
                ))
            elif self.detect_end_of_execution(line):
                self.output_queue.put(OutputChunk(
                    content="Execution completed",
                    type="success"
                ))
                time.sleep(0.1)
                self.done.set()
            elif is_error_stream and "KeyboardInterrupt" in line:
                self.output_queue.put(OutputChunk(
                    content="KeyboardInterrupt",
                    type="warning"
                ))
                time.sleep(0.1)
                self.done.set()
            else:
                self.output_queue.put(OutputChunk(
                    content=line,
                    type=chunk_type
                ))

    def run_code(self, code: str) -> str:
        """Run code with enhanced output handling"""
        try:
            process = subprocess.Popen(
                code,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Start output handling threads
            stdout_thread = threading.Thread(target=self.handle_stream_output, args=(process.stdout, False))
            stderr_thread = threading.Thread(target=self.handle_stream_output, args=(process.stderr, True))
            stdout_thread.start()
            stderr_thread.start()

            # Handle streaming output
            terminal.handle_stream(self.output_queue, self.done)

            # Wait for completion
            stdout_thread.join()
            stderr_thread.join()
            process.wait()

            if process.returncode != 0:
                terminal.status_panel(f"Process exited with code {process.returncode}", "error")
            
            return "Code execution completed"

        except Exception as e:
            terminal.status_panel(f"Error executing code: {str(e)}", "error")
            raise


# interpreter = SubprocessCodeInterpreter()
# interpreter.start_cmd = "python3"
# out = interpreter.run("""
# print("hello")
# print("world")
# """)
# print(out)
