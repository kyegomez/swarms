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

    def __init__(self, md: bool = True, show_swarm_structure: bool = False):
        """
        Initializes the Formatter with a Rich Console instance.

        Args:
            md (bool): Enable markdown output rendering. Defaults to True.
            show_swarm_structure (bool): Enable automatic swarm structure visualization. Defaults to False.
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
        
        # Swarm structure visualization setting
        self.show_swarm_structure = show_swarm_structure

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
            "completed": "✓",
            "pending": "○",
            "error": "✗",
        }.get(status.lower(), "•")

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

    def _is_swarm(self, obj: Any) -> bool:
        """
        Check if an object is a swarm by examining its attributes and class name.
        
        This method works globally with any object that has swarm-like characteristics,
        making it work across all swarm types without needing individual implementations.

        Args:
            obj (Any): The object to check.

        Returns:
            bool: True if the object appears to be a swarm, False otherwise.
        """
        if obj is None:
            return False
        
        # Check if object has 'agents' attribute (common to swarms)
        has_agents = hasattr(obj, "agents")
        
        # Check if object has 'name' or 'agent_name' attribute (common to swarms)
        has_name = hasattr(obj, "name") or hasattr(obj, "agent_name")
        
        # Check class name for swarm indicators
        class_name = type(obj).__name__.lower()
        is_swarm_class = (
            "swarm" in class_name
            or "board" in class_name
            or "hierarchical" in class_name
            or "heavy" in class_name
            or "parliament" in class_name
        )
        
        # Object is likely a swarm if it has agents and (name or is a swarm class)
        return has_agents and (has_name or is_swarm_class)

    def _get_swarm_name(self, swarm: Any) -> str:
        """
        Get the name of a swarm object.

        Args:
            swarm (Any): The swarm object.

        Returns:
            str: The name of the swarm, or a default name if not available.
        """
        # Prefer the agent-facing name when available (`agent_name`), then `name`,
        # and finally fall back to the class name. This ties displayed labels to
        # the Agent API (`agent_name`) where swarms are used as agents.
        agent_name = getattr(swarm, "agent_name", None)
        if agent_name:
            return agent_name

        name = getattr(swarm, "name", None)
        if name:
            return name

            return type(swarm).__name__

    def _get_swarm_type(self, swarm: Any) -> str:
        """
        Get the type/class name of a swarm object.

        Args:
            swarm (Any): The swarm object.

        Returns:
            str: The class name of the swarm.
        """
        # If this object is an adapter/wrapper with an inner `swarm` attribute,
        # prefer the inner swarm's class name so tree labels show the real swarm type.
        inner = getattr(swarm, "swarm", None)
        if inner is not None:
            return type(inner).__name__
        return type(swarm).__name__

    def _get_agent_type_summary(self, agents: List[Any]) -> str:
        """
        Get a summary of agent types in a list.

        Args:
            agents (List[Any]): List of agents.

        Returns:
            str: Summary string describing the agent types.
        """
        if not agents:
            return "No agents"
        
        # Count different agent types
        agent_types = {}
        for agent in agents:
            agent_type = type(agent).__name__
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        
        # Format summary
        if len(agent_types) == 1:
            agent_type_name = list(agent_types.keys())[0]
            count = agent_types[agent_type_name]
            return f"{count} {agent_type_name}{'s' if count > 1 else ''} (Leaf Level)"
        else:
            parts = [f"{count} {name}" for name, count in agent_types.items()]
            return f"{', '.join(parts)} (Leaf Level)"

    def _build_rich_tree(
        self, swarm: Any, tree_node: Optional[Tree] = None, visited: Optional[set] = None, root_tree: Optional[Tree] = None
    ) -> Tree:
        """
        Recursively build a Rich Tree structure representation of a swarm hierarchy.

        Args:
            swarm (Any): The swarm object to visualize.
            tree_node (Optional[Tree]): Parent Rich Tree node. If None, creates root.
            visited (Optional[set]): Set of visited swarm IDs to prevent cycles.
            root_tree (Optional[Tree]): Root tree node for tracking. Internal use.

        Returns:
            Tree: Rich Tree object representing the swarm hierarchy (root tree).
        """
        if visited is None:
            visited = set()
        
        # Get swarm name and type
        swarm_name = self._get_swarm_name(swarm)
        swarm_type = self._get_swarm_type(swarm)
        
        # Create tree node label with styling
        label = Text()
        label.append(swarm_name, style="bold cyan")
        label.append(" (", style="white")
        label.append(swarm_type, style="yellow")
        label.append(")", style="white")
        
        # Check if we've already visited this swarm (prevent cycles)
        swarm_id = id(swarm)
        if swarm_id in visited:
            if tree_node is None:
                tree = Tree(label)
                root_tree = tree
            else:
                tree = tree_node.add(label)
            tree.add("[dim italic]Circular Reference[/dim italic]", style="red")
            return root_tree if root_tree else tree
        
        visited.add(swarm_id)
        
        # Create root tree or add to parent
        if tree_node is None:
            # Create root tree expanded so deep branches are visible by default
            tree = Tree(label, expanded=True)
            root_tree = tree
        else:
            # Add child node and expand it to show its children inline
            tree = tree_node.add(label, expanded=True)
        
        # Get agents if available
        agents = []
        if hasattr(swarm, "agents"):
            agents = swarm.agents if isinstance(swarm.agents, list) else []
        elif hasattr(swarm, "create_agents"):
            # For HeavySwarm and similar, agents might be in a dict
            try:
                agents_dict = swarm.create_agents()
                if isinstance(agents_dict, dict):
                    agents = list(agents_dict.values())
                elif isinstance(agents_dict, list):
                    agents = agents_dict
            except Exception:
                pass
        
        if not agents:
            visited.remove(swarm_id)
            return root_tree if root_tree else tree
        
        # Separate swarms from leaf agents
        swarm_agents = []
        leaf_agents = []
        
        for agent in agents:
            if self._is_swarm(agent):
                swarm_agents.append(agent)
            else:
                leaf_agents.append(agent)
        
        # Process swarm agents first - recursively build subtrees
        for agent in swarm_agents:
            self._build_rich_tree(agent, tree, visited.copy(), root_tree)
        
        # Process leaf agents - show each agent individually with its name
        for agent in leaf_agents:
            # Use canonical name and type helpers so labels consistently reflect
            # `agent_name`/`name` and the underlying class (unwrapping adapters).
            agent_name = self._get_swarm_name(agent)
            agent_type = self._get_swarm_type(agent)
            
            # Create label for individual agent
            agent_label = Text()
            agent_label.append(agent_name, style="green")
            agent_label.append(" (", style="white")
            agent_label.append(agent_type, style="dim green")
            agent_label.append(")", style="white")
            
            tree.add(agent_label)
        
        visited.remove(swarm_id)
        return root_tree if root_tree else tree

    def print_swarm_structure(self, swarm: Any, title: str = "Nested Structure:") -> None:
        """
        Print a visual tree representation of a nested swarm structure using Rich Tree.
        
        Note: Rich tree visualization is only available for HierarchicalSwarm instances.
        Other swarm types will not be visualized.

        Args:
            swarm (Any): The root swarm object to visualize.
            title (str): Title to display above the structure. Defaults to "Nested Structure:".
        """
        if not swarm:
            return
        
        # Only use Rich tree for HierarchicalSwarm instances
        swarm_type_name = type(swarm).__name__
        if swarm_type_name != "HierarchicalSwarm":
            # For non-HierarchicalSwarm types, do nothing
            return
        
        # Always print title first using regular print to ensure visibility
        print(f"\n{title}")
        
        try:
            # Build the Rich Tree structure
            tree = self._build_rich_tree(swarm)
            
            if not tree:
                print("[yellow]No tree structure to display[/yellow]")
                return
            
            # Try to print using Rich console
            try:
                self.console.print(tree)
            except Exception:
                # If Rich fails, fall back to simple text representation
                self._print_simple_tree(swarm)
        except Exception as e:
            # Fallback to basic printing if Rich Tree fails
            print(f"Warning: Could not render swarm structure: {e}")
            import traceback
            traceback.print_exc()
            # Try simple fallback
            self._print_simple_tree(swarm)
    
    def _print_simple_tree(self, swarm: Any, prefix: str = "", is_last: bool = True) -> None:
        """
        Print a simple text-based tree representation as fallback.
        
        Args:
            swarm (Any): The swarm object to print.
            prefix (str): Current prefix for indentation.
            is_last (bool): Whether this is the last item at its level.
        """
        swarm_name = self._get_swarm_name(swarm)
        swarm_type = self._get_swarm_type(swarm)
        connector = "└─ " if is_last else "├─ "
        print(f"{prefix}{connector}{swarm_name} ({swarm_type})")
        
        # Get agents
        agents = []
        if hasattr(swarm, "agents"):
            agents = swarm.agents if isinstance(swarm.agents, list) else []
        elif hasattr(swarm, "create_agents"):
            try:
                agents_dict = swarm.create_agents()
                if isinstance(agents_dict, dict):
                    agents = list(agents_dict.values())
                elif isinstance(agents_dict, list):
                    agents = agents_dict
            except Exception:
                pass
        
        if not agents:
            return
        
        # Separate swarms from leaf agents
        swarm_agents = [a for a in agents if self._is_swarm(a)]
        leaf_agents = [a for a in agents if not self._is_swarm(a)]
        
        # Print swarm agents
        extension = "   " if is_last else "│  "
        new_prefix = prefix + extension
        for i, agent in enumerate(swarm_agents):
            is_last_agent = (i == len(swarm_agents) - 1) and len(leaf_agents) == 0
            self._print_simple_tree(agent, new_prefix, is_last_agent)
        
        # Print leaf agents individually
        for i, agent in enumerate(leaf_agents):
            is_last_agent = (i == len(leaf_agents) - 1) and len(swarm_agents) == 0
            agent_name = getattr(agent, "agent_name", getattr(agent, "name", "Unknown"))
            # Prefer underlying swarm/agent type if wrapped by an adapter
            agent_type = type(getattr(agent, "swarm")).__name__ if hasattr(agent, "swarm") else type(agent).__name__
            connector = "└─ " if is_last_agent else "├─ "
            print(f"{new_prefix}{connector}{agent_name} ({agent_type})")


# Global formatter instance with markdown output enabled by default
formatter = Formatter(md=False)

# Internal helpers to avoid duplicate/child prints when nested BaseSwarm
# instances are created during a top-level swarm initialization.
# We accumulate requested-print instances and only render once when the
# outermost initializer finishes.
_swarm_init_depth = 0
_pending_swarm_prints: List[Any] = []


# Note: monkeypatch removed — buffering is handled directly in
# `swarms.structs.various_alt_swarms.BaseSwarm.__init__` for explicitness.


def enable_swarm_structure_visualization(obj: Any, show: bool = True) -> None:
    """
    Backwards-compatible public helper to trigger swarm structure visualization.

    Keeps the API stable: callers can import this name from
    `swarms.utils.formatter` (used in examples) and it will invoke the
    formatter's visualization.
    """
    if show and obj is not None:
        try:
            formatter.print_swarm_structure(obj)
        except Exception:
            # Best-effort only
            pass
