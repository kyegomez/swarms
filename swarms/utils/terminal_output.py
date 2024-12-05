import time
from typing import Any, Callable, Dict, List, Optional, Union
from queue import Queue
from threading import Event, Thread
from dataclasses import dataclass

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from rich.style import Style

@dataclass
class OutputChunk:
    content: str
    type: str = "text"  # text, error, warning, success
    metadata: Dict[str, Any] = None

class TerminalOutput:
    """Enhanced terminal output system with streaming, chunking and rich formatting"""
    
    def __init__(self):
        self.console = Console()
        self.output_queue = Queue()
        self.done = Event()
        self.chunk_size = 1000  # Default chunk size for large outputs
        self.styles = {
            "text": "white",
            "error": "bold red",
            "warning": "bold yellow", 
            "success": "bold green",
            "info": "bold blue"
        }

    def stream_output(self, content: str, chunk_size: int = None, delay: float = 0.01):
        """Stream output in chunks with real-time updates"""
        if chunk_size is None:
            chunk_size = self.chunk_size

        text = Text()
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

        with Live(Panel(text, title="Streaming Output", border_style="blue"),
                 console=self.console, refresh_per_second=20) as live:
            for chunk in chunks:
                text.append(chunk)
                live.update(Panel(text, title="Streaming Output", border_style="blue"))
                time.sleep(delay)

    def progress_bar(self, total: int = None, description: str = "Processing") -> Progress:
        """Create a rich progress bar with spinner and time remaining"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        )

    def status_panel(self, message: str, style: str = "info"):
        """Display a status message in a styled panel"""
        style_str = self.styles.get(style, "white")
        panel = Panel(
            Text(message, style=style_str),
            title=style.capitalize(),
            border_style=style_str
        )
        self.console.print(panel)

    def chunked_table(self, data: List[Dict[str, Any]], chunk_size: int = 10):
        """Display large datasets in chunked tables"""
        if not data:
            return

        headers = list(data[0].keys())
        table = Table(show_header=True, header_style="bold magenta")
        for header in headers:
            table.add_column(header)

        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        for chunk_idx, chunk in enumerate(chunks):
            self.console.print(f"\nShowing chunk {chunk_idx + 1}/{len(chunks)}")
            
            for row in chunk:
                table.add_row(*[str(row[header]) for header in headers])
            
            self.console.print(table)
            table.rows.clear()  # Clear rows for next chunk
            
            if chunk_idx < len(chunks) - 1:
                input("Press Enter to see next chunk...")

    def handle_stream(self, stream: Queue[OutputChunk], stop_event: Event):
        """Handle streaming output from a queue"""
        with Live(console=self.console) as live:
            while not stop_event.is_set() or not stream.empty():
                try:
                    chunk = stream.get_nowait()
                    style = self.styles.get(chunk.type, "white")
                    text = Text(chunk.content, style=style)
                    live.update(Panel(text))
                except:
                    time.sleep(0.1)

    def execute_with_progress(self, 
                            func: Callable, 
                            description: str = "Processing", 
                            *args, **kwargs) -> Any:
        """Execute a function with progress bar"""
        with self.progress_bar() as progress:
            task = progress.add_task(description, total=None)
            result = func(*args, **kwargs)
            progress.update(task, completed=True)
            return result

# Create singleton instance
terminal = TerminalOutput() 