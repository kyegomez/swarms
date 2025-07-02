import threading
import time
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text


def choose_random_color():
    import random

    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "magenta",
        "cyan",
        "white",
    ]
    random_color = random.choice(colors)

    return random_color


class Formatter:
    """
    A class for formatting and printing rich text to the console.
    """

    def __init__(self):
        """
        Initializes the Formatter with a Rich Console instance.
        """
        self.console = Console()

    def _print_panel(
        self, content: str, title: str = "", style: str = "bold blue"
    ) -> None:
        """
        Prints a rich panel to the console with a random color.

        Args:
            content (str): The content of the panel.
            title (str, optional): The title of the panel. Defaults to "".
            style (str, optional): The style of the panel. Defaults to "bold blue".
        """
        random_color = choose_random_color()

        panel = Panel(
            content, title=title, style=f"bold {random_color}"
        )
        self.console.print(panel)

    def print_panel(
        self,
        content: str,
        title: str = "",
        style: str = "bold blue",
    ) -> None:
        process_thread = threading.Thread(
            target=self._print_panel,
            args=(content, title, style),
            daemon=True,
        )
        process_thread.start()

    def print_table(
        self, title: str, data: Dict[str, List[str]]
    ) -> None:
        """
        Prints a rich table to the console.

        Args:
            title (str): The title of the table.
            data (Dict[str, List[str]]): A dictionary where keys are categories and values are lists of capabilities.
        """
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan")
        table.add_column("Capabilities", style="green")

        for category, items in data.items():
            table.add_row(category, "\n".join(items))

        self.console.print(f"\n🔥 {title}:", style="bold yellow")
        self.console.print(table)

    def print_progress(
        self,
        description: str,
        task_fn: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Prints a progress bar to the console and executes a task function.

        Args:
            description (str): The description of the task.
            task_fn (Callable): The function to execute.
            *args (Any): Arguments to pass to the task function.
            **kwargs (Any): Keyword arguments to pass to the task function.

        Returns:
            Any: The result of the task function.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task(description, total=None)
            result = task_fn(*args, **kwargs)
            progress.update(task, completed=True)
        return result

    def print_panel_token_by_token(
        self,
        tokens: str,
        title: str = "Output",
        style: str = "bold cyan",
        delay: float = 0.01,
        by_word: bool = False,
    ) -> None:
        """
        Prints a string in real-time, token by token (character or word) inside a Rich panel.

        Args:
            tokens (str): The string to display in real-time.
            title (str): Title of the panel.
            style (str): Style for the text inside the panel.
            delay (float): Delay in seconds between displaying each token.
            by_word (bool): If True, display by words; otherwise, display by characters.
        """
        text = Text(style=style)

        # Split tokens into characters or words
        token_list = tokens.split() if by_word else tokens

        with Live(
            Panel(text, title=title, border_style=style),
            console=self.console,
            refresh_per_second=10,
        ) as live:
            for token in token_list:
                text.append(token + (" " if by_word else ""))
                live.update(
                    Panel(text, title=title, border_style=style)
                )
                time.sleep(delay)

    def print_streaming_panel(
        self,
        streaming_response,
        title: str = "🤖 Agent Streaming Response",
        style: str = None,
        collect_chunks: bool = False,
        on_chunk_callback: Optional[Callable] = None,
    ) -> str:
        """
        Display real-time streaming response using Rich Live and Panel.
        Similar to the approach used in litellm_stream.py.

        Args:
            streaming_response: The streaming response generator from LiteLLM.
            title (str): Title of the panel.
            style (str): Style for the panel border (if None, will use random color).
            collect_chunks (bool): Whether to collect individual chunks for conversation saving.
            on_chunk_callback (Optional[Callable]): Callback function to call for each chunk.

        Returns:
            str: The complete accumulated response text.
        """
        # Get random color similar to non-streaming approach
        random_color = choose_random_color()
        panel_style = (
            f"bold {random_color}" if style is None else style
        )
        text_style = (
            "white"  # Make text white instead of random color
        )

        def create_streaming_panel(text_obj, is_complete=False):
            """Create panel with proper text wrapping using Rich's built-in capabilities"""
            panel_title = f"[white]{title}[/white]"
            if is_complete:
                panel_title += " [bold green]✅[/bold green]"

            # Add blinking cursor if still streaming
            display_text = Text.from_markup("")
            display_text.append_text(text_obj)
            if not is_complete:
                display_text.append("▊", style="bold green blink")

            panel = Panel(
                display_text,
                title=panel_title,
                border_style=panel_style,
                padding=(1, 2),
                width=self.console.size.width,  # Rich handles wrapping automatically
            )
            return panel

        # Create a Text object for streaming content
        streaming_text = Text()
        complete_response = ""
        chunks_collected = []

        # TRUE streaming with Rich's automatic text wrapping
        with Live(
            create_streaming_panel(streaming_text),
            console=self.console,
            refresh_per_second=20,
        ) as live:
            try:
                for part in streaming_response:
                    if (
                        hasattr(part, "choices")
                        and part.choices
                        and part.choices[0].delta.content
                    ):
                        # Add ONLY the new chunk to the Text object with random color style
                        chunk = part.choices[0].delta.content
                        streaming_text.append(chunk, style=text_style)
                        complete_response += chunk

                        # Collect chunks if requested
                        if collect_chunks:
                            chunks_collected.append(chunk)

                        # Call chunk callback if provided
                        if on_chunk_callback:
                            on_chunk_callback(chunk)

                        # Update display with new text - Rich handles all wrapping automatically
                        live.update(
                            create_streaming_panel(
                                streaming_text, is_complete=False
                            )
                        )

                # Final update to show completion
                live.update(
                    create_streaming_panel(
                        streaming_text, is_complete=True
                    )
                )

            except Exception as e:
                # Handle any streaming errors gracefully
                streaming_text.append(
                    f"\n[Error: {str(e)}]", style="bold red"
                )
                live.update(
                    create_streaming_panel(
                        streaming_text, is_complete=True
                    )
                )

        return complete_response


formatter = Formatter()
