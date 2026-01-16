import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from .manager import ChiefOfStaff

app = typer.Typer()
console = Console()
chief = ChiefOfStaff(db_path="chief_of_staff.db")


@app.command()
def create(title: str = typer.Argument(...), description: str = typer.Argument(...)):
    t = chief.create_task(title, description)
    console.print(Panel(f"Created task [bold]{t.title}[/bold] with id [cyan]{t.id}[/cyan]", title="Task Created"))


@app.command()
def list_tasks():
    table = Table(title="Tasks")
    table.add_column("ID", style="cyan")
    table.add_column("Status")
    table.add_column("Title")
    for t in chief.list_tasks():
        table.add_row(t.id, t.status, t.title)
    console.print(table)


@app.command()
def plan(task_id: str):
    out = chief.plan_task(task_id)
    console.print(Panel(str(out), title="Plan"))


@app.command()
def approve(task_id: str):
    chief.approve_plan(task_id)
    console.print(f"Approved {task_id}")


@app.command()
def interactive_approve():
    """Interactive approval flow for all planned tasks."""
    pending = [t for t in chief.list_tasks() if t.status == "planned"]
    if not pending:
        console.print("No planned tasks to approve.")
        return
    from rich.prompt import Confirm

    for t in pending:
        console.print(Panel(f"{t.title}\n\n{t.description}", title=f"{t.id} - Approve?"))
        if Confirm.ask("Approve this plan?"):
            chief.approve_plan(t.id)
            console.print(f"Approved {t.id}")
        else:
            console.print(f"Skipped {t.id}")


@app.command()
def spawn(task_id: str, sub_title: str = typer.Argument(...), sub_desc: str = typer.Argument(...)):
    spec = {"title": sub_title, "description": sub_desc}
    out = chief.spawn_subagent(task_id, spec)
    console.print(Panel(str(out), title="Subagent Output"))


@app.command()
def show(task_id: str):
    """Show task details and recent execution outputs."""
    t = chief.get_task(task_id)
    if not t:
        console.print(f"Task {task_id} not found.")
        raise typer.Exit(1)
    console.print(Panel(f"Status: {t.status}\n\n{t.description}", title=f"{t.id} - {t.title}"))
    if t.result:
        console.print(Panel(str(t.result), title="Result"))


@app.command()
def execute(task_id: str):
    """Execute planned subtasks for a task concurrently."""
    try:
        out = chief.execute_plan(task_id)
        console.print(Panel(str(out), title="Execution Outputs"))
    except Exception as e:
        console.print(f"Error executing plan: {e}")
        raise typer.Exit(2)


def main():
    app()


if __name__ == "__main__":
    main()
