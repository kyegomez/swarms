import typer
from rich import print
from .manager import TaskManager

app = typer.Typer()
manager = TaskManager()


@app.command()
def create(title: str = typer.Argument(...), description: str = typer.Argument(...)):
    """Create a new task"""
    t = manager.create_task(title, description)
    print(f"Created task {t.id} - {t.title}")


@app.command()
def list_tasks():
    """List tasks"""
    for t in manager.list_tasks():
        print(f"- {t.id} [{t.status}] {t.title}")


@app.command()
def plan(task_id: str):
    """Run research to propose a plan for a task"""
    out = manager.plan_task(task_id)
    print("Proposed plan:")
    print(out)


@app.command()
def approve(task_id: str):
    """Approve a planned task"""
    manager.approve_plan(task_id)
    print(f"Approved {task_id}")


@app.command()
def spawn(task_id: str, sub_title: str = typer.Argument(...), sub_desc: str = typer.Argument(...)):
    """Spawn a sub-agent for a subtask"""
    spec = {"title": sub_title, "description": sub_desc}
    out = manager.spawn_subagent(task_id, spec)
    print(out)


if __name__ == "__main__":
    app()
