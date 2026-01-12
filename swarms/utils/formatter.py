import time
import re
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.spinner import Spinner
from rich.tree import Tree

from rich.markdown import Markdown


# Global Live display for the dashboard
dashboard_live = None

# Create a spinner for loading animation
spinner = Spinner("dots", style="yellow")


class MarkdownOutputHandler:
    """Custom output handler to render content as markdown with simplified syntax highlighting."""

    def __init__(self, console: "Console"):
        """Initialize the MarkdownOutputHandler with a console instance.

        Args:
            console (Console): Rich console instance for rendering.
        """
        self.console = console

    def _clean_output(self, output: str) -> str:
        """Clean up the output for better markdown rendering"""
        if not output:
            return ""

        # Remove log prefixes and timestamps
        output = re.sub(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| INFO.*?\|.*?\|",
            "",
            output,
        )
        output = re.sub(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| DEBUG.*?\|.*?\|",
            "",
            output,
        )
        output = re.sub(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| WARNING.*?\|.*?\|",
            "",
            output,
        )
        output = re.sub(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| ERROR.*?\|.*?\|",
            "",
            output,
        )

        # Remove spinner characters and progress indicators
        spinner_chars = "[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]"
        output = re.sub(rf"{spinner_chars}", "", output)
        output = re.sub(
            rf"{spinner_chars} Processing\.\.\.", "", output
        )
        output = re.sub(rf"{spinner_chars} Loop \d+/\d+", "", output)

        # Remove any remaining log messages
        output = re.sub(r"INFO.*?\|.*?\|.*?\|", "", output)
        output = re.sub(r"DEBUG.*?\|.*?\|.*?\|", "", output)
        output = re.sub(r"WARNING.*?\|.*?\|.*?\|", "", output)
        output = re.sub(r"ERROR.*?\|.*?\|.*?\|", "", output)

        # Clean up extra whitespace and empty lines
        output = re.sub(r"\n\s*\n\s*\n", "\n\n", output)
        output = re.sub(r"^\s+", "", output, flags=re.MULTILINE)
        output = re.sub(r"\s+$", "", output, flags=re.MULTILINE)

        # Remove any remaining plaintext artifacts
        output = re.sub(r"Generated content:", "", output)
        output = re.sub(r"Evaluation result:", "", output)
        output = re.sub(r"Refined content:", "", output)

        # Ensure proper markdown formatting
        lines = output.strip().split("\n")
        if lines and not any(
            line.strip().startswith("#") for line in lines[:3]
        ):
            # Check if first line looks like a title (not already formatted)
            first_line = lines[0].strip()
            if (
                first_line
                and not first_line.startswith(
                    ("**", "#", "-", "*", ">", "```")
                )
                and len(first_line) < 100  # Reasonable title length
                and not first_line.endswith((",", ".", ":", ";"))
                or first_line.endswith(":")
            ):
                # Make it a header
                output = f"## {first_line}\n\n" + "\n".join(lines[1:])
            else:
                # Keep original formatting
                output = "\n".join(lines)

        return output.strip()

    def render_with_simple_syntax_highlighting(
        self, content: str
    ) -> list:
        """Render content with syntax highlighting for code blocks.

        Args:
            content (str): The content to parse and highlight.

        Returns:
            list: List of tuples (type, content) where type is 'markdown' or 'code'.
        """
        parts = []
        current_pos = 0

        # Pattern to match code blocks with optional language specifier
        code_block_pattern = re.compile(
            r"```(?P<lang>\w+)?\n(?P<code>.*?)\n```",
            re.DOTALL | re.MULTILINE,
        )

        for match in code_block_pattern.finditer(content):
            # Add markdown content before code block
            if match.start() > current_pos:
                markdown_content = content[
                    current_pos : match.start()
                ].strip()
                if markdown_content:
                    parts.append(("markdown", markdown_content))

            # Add code block
            lang = match.group("lang") or "text"
            code = match.group("code")
            parts.append(("code", (lang, code)))

            current_pos = match.end()

        # Add remaining markdown content
        if current_pos < len(content):
            remaining = content[current_pos:].strip()
            if remaining:
                parts.append(("markdown", remaining))

        # If no parts found, treat entire content as markdown
        if not parts:
            parts.append(("markdown", content))

        return parts

    def render_content_parts(self, parts: list) -> list:
        """Render different content parts with appropriate formatting.

        Args:
            parts (list): List of tuples (type, content) to render.

        Returns:
            list: List of rendered Rich objects.
        """
        rendered_parts = []

        for part_type, content in parts:
            if part_type == "markdown":
                # Render markdown
                try:
                    md = Markdown(content, code_theme="monokai")
                    rendered_parts.append(md)
                except Exception:
                    # Fallback to plain text with error indication
                    rendered_parts.append(
                        Text(content, style="white")
                    )

            elif part_type == "code":
                # Render code with syntax highlighting
                lang, code = content
                try:
                    from rich.syntax import Syntax

                    syntax = Syntax(
                        code,
                        lang,
                        theme="monokai",
                        line_numbers=True,
                        word_wrap=True,
                    )
                    rendered_parts.append(syntax)
                except Exception:
                    # Fallback to text with code styling
                    rendered_parts.append(
                        Text(
                            f"```{lang}\n{code}\n```",
                            style="white on grey23",
                        )
                    )

        return rendered_parts

    def render_markdown_output(
        self,
        content: str,
        title: str = "",
        border_style: str = "blue",
    ):
        """Render content as markdown with syntax highlighting.

        Args:
            content (str): The markdown content to render.
            title (str): Title for the panel.
            border_style (str): Border style for the panel.
        """
        if not content or content.strip() == "":
            return

        # Clean up the output
        cleaned_content = self._clean_output(content)

        # Render with syntax highlighting
        try:
            # Split content into parts (markdown and code blocks)
            parts = self.render_with_simple_syntax_highlighting(
                cleaned_content
            )

            # Render each part appropriately
            rendered_parts = self.render_content_parts(parts)

            # Create a group of rendered parts
            from rich.console import Group

            if rendered_parts:
                content_group = Group(*rendered_parts)

                self.console.print(
                    Panel(
                        content_group,
                        title=title,
                        border_style=border_style,
                        padding=(1, 2),
                        expand=False,
                    )
                )
            else:
                # No content to render
                self.console.print(
                    Panel(
                        Text(
                            "No content to display",
                            style="dim italic",
                        ),
                        title=title,
                        border_style="yellow",
                    )
                )
        except Exception as e:
            # Fallback to plain text if rendering fails with better error info
            error_msg = f"Markdown rendering error: {str(e)}"
            self.console.print(
                Panel(
                    cleaned_content,
                    title=(
                        f"{title} [dim](fallback mode)[/dim]"
                        if title
                        else "Content (fallback mode)"
                    ),
                    border_style="yellow",
                    subtitle=error_msg,
                    subtitle_align="left",
                )
            )


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

    def __init__(self, md: bool = True):
        """
        Initializes the Formatter with a Rich Console instance.

        Args:
            md (bool): Enable markdown output rendering. Defaults to True.
        """
        self.console = Console()
        self._dashboard_live = None
        self._spinner_frames = [
            "⠋",
            "⠙",
            "⠹",
            "⠸",
            "⠼",
            "⠴",
            "⠦",
            "⠧",
            "⠇",
            "⠏",
        ]
        self._spinner_idx = 0

        # Set markdown capability based on user preference
        self.markdown_handler = (
            MarkdownOutputHandler(self.console) if md else None
        )

    def _get_status_with_loading(self, status: str) -> Text:
        """
        Creates a status text with loading animation for running status.
        """
        if status.lower() == "running":
            # Create loading bar effect
            self._spinner_idx = (self._spinner_idx + 1) % len(
                self._spinner_frames
            )
            spinner_char = self._spinner_frames[self._spinner_idx]
            progress_bar = "█" * (self._spinner_idx % 5) + "░" * (
                4 - (self._spinner_idx % 5)
            )
            return Text(
                f"{spinner_char} {status} {progress_bar}",
                style="bold yellow",
            )

        # Style other statuses
        status_style = {
            "completed": "bold green",
            "pending": "bold red",
            "error": "bold red",
        }.get(status.lower(), "white")

        status_symbol = {
            "completed": "[OK]",
            "pending": "[..]",
            "error": "[!!]",
        }.get(status.lower(), "[--]")

        return Text(f"{status_symbol} {status}", style=status_style)

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
        """Print content in a panel with a title and style.

        Args:
            content (str): The content to display in the panel
            title (str): The title of the panel
            style (str): The style to apply to the panel
        """
        # Handle None content
        if content is None:
            content = "No content to display"

        # Convert non-string content to string
        if not isinstance(content, str):
            content = str(content)

        # Use markdown rendering if enabled
        if self.markdown_handler:
            self.markdown_handler.render_markdown_output(
                content, title, style
            )
        else:
            # Fallback to original panel printing
            try:
                self._print_panel(content, title, style)
            except Exception:
                # Fallback to basic printing if panel fails
                print(f"\n{title}:")
                print(content)

    def print_markdown(
        self,
        content: str,
        title: str = "",
        border_style: str = "blue",
    ) -> None:
        """Print content as markdown with syntax highlighting.

        Args:
            content (str): The content to display as markdown
            title (str): The title of the panel
            border_style (str): The border style for the panel
        """
        if self.markdown_handler:
            self.markdown_handler.render_markdown_output(
                content, title, border_style
            )
        else:
            # Fallback to regular panel if markdown is disabled
            self.print_panel(content, title, border_style)

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

        self.console.print(f"\n[*] {title}:", style="bold yellow")
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
        title: str = "Agent Streaming Response",
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
                panel_title += " [bold green][DONE][/bold green]"

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

    def _create_dashboard_table(
        self, agents_data: List[Dict[str, Any]], title: str
    ) -> Panel:
        """
        Creates the dashboard table with the current agent statuses.
        """
        # Create main table
        table = Table(
            show_header=True,
            header_style="bold magenta",
            expand=True,
            title=title,
            title_style="bold cyan",
            border_style="bright_blue",
            show_lines=True,  # Add lines between rows
        )

        # Add columns with adjusted widths
        table.add_column(
            "Agent Name", style="cyan", width=30, no_wrap=True
        )
        table.add_column(
            "Status", style="green", width=20, no_wrap=True
        )  # Increased width for loading animation
        table.add_column(
            "Output", style="white", width=100, overflow="fold"
        )  # Allow text to wrap

        # Add rows for each agent
        for agent in agents_data:
            name = Text(agent["name"], style="bold cyan")
            status = self._get_status_with_loading(agent["status"])
            output = Text(str(agent["output"]))
            table.add_row(name, status, output)

        # Create a panel to wrap the table
        dashboard_panel = Panel(
            table,
            border_style="bright_blue",
            padding=(1, 2),
            title=f"[bold cyan]{title}[/bold cyan] - Total Agents: [bold green]{len(agents_data)}[/bold green]",
            expand=True,  # Make panel expand to full width
        )

        return dashboard_panel

    def print_agent_dashboard(
        self,
        agents_data: List[Dict[str, Any]],
        title: str = "Concurrent Workflow Dashboard",
        is_final: bool = False,
    ) -> None:
        """
        Displays a beautiful dashboard showing agent information in a panel-like spreadsheet format.
        Updates in place instead of printing multiple times.

        Args:
            agents_data (List[Dict[str, Any]]): List of dictionaries containing agent information.
                Each dict should have: name, status, output
            title (str): The title of the dashboard.
            is_final (bool): Whether this is the final update of the dashboard.
        """
        if self._dashboard_live is None:
            # Create new Live display if none exists
            self._dashboard_live = Live(
                self._create_dashboard_table(agents_data, title),
                console=self.console,
                refresh_per_second=10,  # Increased refresh rate
                transient=False,  # Make display persistent
            )
            self._dashboard_live.start()
        else:
            # Update existing Live display
            self._dashboard_live.update(
                self._create_dashboard_table(agents_data, title)
            )

            # If this is the final update, add a newline to separate from future output
            if is_final:
                self.console.print()  # Add blank line after final display

    def stop_dashboard(self):
        """
        Stops and cleans up the dashboard display.
        """
        if self._dashboard_live is not None:
            self._dashboard_live.stop()
            self.console.print()  # Add blank line after stopping
            self._dashboard_live = None

    def print_plan_tree(
        self,
        task_description: str,
        steps: List[Dict[str, Any]],
        print_on: bool = True,
    ) -> None:
        """
        Print the plan as a beautiful tree using Rich.

        Args:
            task_description: Description of the main task
            steps: List of step dictionaries with step_id, description, priority, and optional dependencies
            print_on: Whether to print to console (True) or just log (False)
        """
        import logging

        logger = logging.getLogger(__name__)

        # Create root tree
        tree = Tree(
            f"[bold cyan][PLAN] {task_description}[/bold cyan]"
        )

        # Priority color mapping
        priority_colors = {
            "critical": "red",
            "high": "yellow",
            "medium": "blue",
            "low": "green",
        }

        # ASCII priority indicators
        priority_icons = {
            "critical": "[!!!]",
            "high": "[!!]",
            "medium": "[!]",
            "low": "[.]",
        }

        # Create a mapping of step_id to tree nodes for dependency handling
        step_nodes = {}

        # First pass: create all nodes
        for step in steps:
            step_id = step.get("step_id", "")
            description = step.get("description", "")
            priority = step.get("priority", "medium").lower()
            dependencies = step.get("dependencies", [])

            priority_color = priority_colors.get(priority, "white")
            priority_icon = priority_icons.get(priority, "[-]")

            # Create step label with priority indicator
            step_label = (
                f"[{priority_color}]{priority_icon} {step_id}[/{priority_color}]: "
                f"{description}"
            )

            # Add dependencies info if present
            if dependencies:
                deps_text = ", ".join(dependencies)
                step_label += f" [dim](depends on: {deps_text})[/dim]"

            # Add node to tree
            step_node = tree.add(step_label)
            step_nodes[step_id] = step_node

        # Print the tree
        if print_on:
            self.console.print("\n")
            self.console.print(tree)
            self.console.print("")
        else:
            # Even if print_on is False, log the tree structure
            logger.info(f"Plan created: {task_description}")
            for step in steps:
                logger.info(
                    f"  - {step.get('step_id')} ({step.get('priority')}): {step.get('description')}"
                )

    def display_hierarchy(
        self,
        director_name: str,
        director_model_name: str,
        agents: List[Any],
        swarm_name: str,
    ) -> None:
        """
        Display the hierarchical structure of the swarm using Rich Tree.

        This method creates a visual tree representation showing the Director
        at the top level and all worker agents as children branches. The tree
        is printed to the console with rich formatting.

        The hierarchy visualization helps understand the organizational structure
        of the swarm, with the Director coordinating all worker agents.

        Args:
            director_name (str): Name of the director agent.
            director_model_name (str): Model name used by the director.
            agents (List[Any]): List of worker agents in the swarm.
            swarm_name (str): Name of the hierarchical swarm.
        """
        # Create the root tree with Director
        director_label = Text()
        director_label.append("[DIR] ", style="bold red")
        director_label.append(director_name, style="bold white")
        director_label.append(
            f" [{director_model_name}]", style="dim cyan"
        )

        tree = Tree(director_label, guide_style="bold red")

        # Add each worker agent as a branch
        for agent in agents:
            agent_label = Text()

            # Get agent name
            if hasattr(agent, "agent_name"):
                agent_name = agent.agent_name
            elif hasattr(agent, "name"):
                agent_name = agent.name
            else:
                agent_name = f"Agent_{agents.index(agent)}"

            # Get agent model if available
            model_info = ""
            if hasattr(agent, "model_name"):
                model_info = f" [{agent.model_name}]"
            elif hasattr(agent, "llm") and hasattr(
                agent.llm, "model"
            ):
                model_info = f" [{agent.llm.model}]"

            # Get agent description if available
            description = ""
            if hasattr(agent, "agent_description"):
                description = f" - {agent.agent_description[:500]}"
            elif hasattr(agent, "description"):
                description = f" - {agent.description[:500]}"

            agent_label.append("[AGT] ", style="bold cyan")
            agent_label.append(agent_name, style="bold cyan")
            if model_info:
                agent_label.append(model_info, style="dim cyan")
            if description:
                agent_label.append(description, style="dim white")

            # Add agent as a branch
            tree.add(agent_label)

        # Create a panel with the tree
        panel = Panel(
            tree,
            title=f"[bold white]HierarchicalSwarm Hierarchy: {swarm_name}[/bold white]",
            border_style="red",
            padding=(1, 2),
        )

        self.console.print(panel)

    def print_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_type: str = "function",
    ) -> None:
        """
        Display a tool call with structured formatting and colors.

        Args:
            tool_name: Name of the tool being called
            tool_args: Dictionary of arguments passed to the tool
            tool_type: Type of tool - "function" or "mcp"
        """
        import time as time_module

        timestamp = time_module.strftime("%H:%M:%S")

        # Format arguments
        if tool_args:
            args_lines = []
            for key, value in tool_args.items():
                args_lines.append(f"  {key}: {value}")
            args_str = "\n".join(args_lines)
        else:
            args_str = "  (no arguments)"

        # Choose style based on tool type
        if tool_type == "mcp":
            indicator = "[>]"
            style = "cyan"
            title = f"MCP Tool Call [{timestamp}]"
        else:
            indicator = "[*]"
            style = "yellow"
            title = f"Tool Call [{timestamp}]"

        content = Text()
        content.append(f"{indicator} ", style=f"bold {style}")
        content.append(f"{tool_name}\n", style="bold white")
        content.append(args_str, style="dim white")

        panel = Panel(
            content,
            title=f"[bold {style}]{title}[/bold {style}]",
            border_style=style,
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_tool_result(
        self,
        tool_name: str,
        status: str,
        output: Any,
        duration: float,
        show_output: bool = True,
    ) -> None:
        """
        Display a tool execution result with status and timing.

        Args:
            tool_name: Name of the tool that was executed
            status: Execution status - "success" or "error"
            output: The output from the tool execution
            duration: Execution duration in seconds
            show_output: Whether to display the full output
        """
        import time as time_module

        timestamp = time_module.strftime("%H:%M:%S")

        # Choose style and indicator based on status
        if status == "success":
            indicator = "[+]"
            style = "green"
        else:
            indicator = "[x]"
            style = "red"

        content = Text()
        content.append(f"{indicator} ", style=f"bold {style}")
        content.append(f"{tool_name}", style="bold white")
        content.append(f" ({duration:.2f}s)", style="dim white")

        if show_output and output:
            output_str = str(output)
            # Truncate if too long
            if len(output_str) > 500:
                output_str = output_str[:500] + "..."
            content.append(f"\n\nOutput:\n", style="bold white")
            content.append(output_str, style="white")

        panel = Panel(
            content,
            title=f"[bold {style}]Tool Result [{timestamp}][/bold {style}]",
            border_style=style,
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_mcp_tool_result(
        self,
        output: Any,
        duration: float,
    ) -> None:
        """
        Display an MCP tool execution result.

        Args:
            output: The output from the MCP tool execution
            duration: Execution duration in seconds
        """
        import time as time_module

        timestamp = time_module.strftime("%H:%M:%S")

        content = Text()
        content.append("[>] ", style="bold cyan")
        content.append("MCP Tool", style="bold white")
        content.append(f" ({duration:.2f}s)", style="dim white")

        output_str = str(output)
        if len(output_str) > 1000:
            output_str = output_str[:1000] + "..."
        content.append(f"\n\n{output_str}", style="white")

        panel = Panel(
            content,
            title=f"[bold cyan]MCP Tool Result [{timestamp}][/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_tool_execution_summary(
        self,
        executions: List[Dict[str, Any]],
        title: str = "Tool Execution Summary",
    ) -> None:
        """
        Display a summary table of all tool executions.

        Args:
            executions: List of execution records, each containing:
                - tool_name: Name of the tool
                - status: "success" or "error"
                - duration: Execution time in seconds
                - tokens: Optional token count
                - result_preview: Optional preview of result
            title: Title for the summary table
        """
        if not executions:
            return

        table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="bright_blue",
            title=f"[bold cyan]{title}[/bold cyan]",
            title_justify="left",
        )

        table.add_column("Tool", style="cyan", width=25)
        table.add_column("Status", style="white", width=10, justify="center")
        table.add_column("Duration", style="white", width=12, justify="right")
        table.add_column("Tokens", style="white", width=10, justify="right")
        table.add_column("Result", style="dim white", width=40, overflow="ellipsis")

        total_duration = 0.0
        total_tokens = 0
        success_count = 0
        error_count = 0

        for exec_info in executions:
            tool_name = exec_info.get("tool_name", "Unknown")
            status = exec_info.get("status", "unknown")
            duration = exec_info.get("duration", 0.0)
            tokens = exec_info.get("tokens", 0)
            result_preview = exec_info.get("result_preview", "")

            total_duration += duration
            total_tokens += tokens if tokens else 0

            # Status indicator
            if status == "success":
                status_text = Text("[OK]", style="bold green")
                success_count += 1
            elif status == "error":
                status_text = Text("[!!]", style="bold red")
                error_count += 1
            else:
                status_text = Text("[--]", style="dim white")

            # Truncate result preview
            if result_preview and len(str(result_preview)) > 40:
                result_preview = str(result_preview)[:37] + "..."

            table.add_row(
                tool_name,
                status_text,
                f"{duration:.2f}s",
                str(tokens) if tokens else "-",
                str(result_preview) if result_preview else "-",
            )

        # Add summary row
        table.add_section()
        summary_status = Text(
            f"[OK]:{success_count} [!!]:{error_count}",
            style="bold white",
        )
        table.add_row(
            Text("TOTAL", style="bold white"),
            summary_status,
            Text(f"{total_duration:.2f}s", style="bold white"),
            Text(str(total_tokens) if total_tokens else "-", style="bold white"),
            Text(f"{len(executions)} executions", style="bold white"),
        )

        self.console.print(table)

    def create_tool_progress(
        self,
        tool_name: str,
        tool_type: str = "function",
    ) -> Progress:
        """
        Create a progress indicator for tool execution.

        Args:
            tool_name: Name of the tool being executed
            tool_type: Type of tool - "function" or "mcp"

        Returns:
            Progress: A Rich Progress instance configured for tool execution
        """
        if tool_type == "mcp":
            style = "cyan"
            prefix = "[>]"
        else:
            style = "yellow"
            prefix = "[*]"

        progress = Progress(
            SpinnerColumn(style=style),
            TextColumn(f"[bold {style}]{prefix}[/bold {style}]"),
            TextColumn(f"[bold white]{tool_name}[/bold white]"),
            BarColumn(bar_width=20, style=style, complete_style=f"bold {style}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        return progress

    def print_tool_with_progress(
        self,
        tool_name: str,
        tool_fn: Callable,
        tool_args: Dict[str, Any],
        tool_type: str = "function",
    ) -> Any:
        """
        Execute a tool with a progress indicator.

        Args:
            tool_name: Name of the tool
            tool_fn: The tool function to execute
            tool_args: Arguments to pass to the tool
            tool_type: Type of tool - "function" or "mcp"

        Returns:
            The result of the tool execution
        """
        import time as time_module

        timestamp = time_module.strftime("%H:%M:%S")

        # Display tool call info
        self.print_tool_call(tool_name, tool_args, tool_type)

        # Create and run progress
        progress = self.create_tool_progress(tool_name, tool_type)

        start_time = time.time()
        status = "success"
        result = None

        with progress:
            task = progress.add_task("Executing...", total=None)
            try:
                result = tool_fn(**tool_args)
                progress.update(task, completed=True)
            except Exception as e:
                status = "error"
                result = str(e)
                progress.update(task, completed=True)

        duration = time.time() - start_time

        # Display result
        self.print_tool_result(tool_name, status, result, duration, True)

        return result

    def print_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        model: str = "",
    ) -> None:
        """
        Display token usage metrics.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            total_tokens: Total tokens used
            model: Optional model name
        """
        import time as time_module

        timestamp = time_module.strftime("%H:%M:%S")

        content = Text()
        content.append("[i] ", style="bold blue")
        content.append("Token Usage", style="bold white")
        if model:
            content.append(f" ({model})", style="dim white")
        content.append("\n\n", style="white")
        content.append(f"  Prompt:     {prompt_tokens:,}\n", style="white")
        content.append(f"  Completion: {completion_tokens:,}\n", style="white")
        content.append(f"  Total:      {total_tokens:,}", style="bold white")

        panel = Panel(
            content,
            title=f"[bold blue]Token Metrics [{timestamp}][/bold blue]",
            border_style="blue",
            padding=(0, 1),
        )
        self.console.print(panel)


# Global formatter instance with markdown output enabled by default
formatter = Formatter(md=False)
