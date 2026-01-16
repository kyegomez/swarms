"""Simple SQLite storage for ChiefOfStaff tasks."""
import sqlite3
from typing import List, Optional

# Avoid circular import at module import time: import Task locally in functions


class ChiefStorage:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "chief_of_staff.db"
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    status TEXT,
                    result TEXT,
                    subtasks TEXT
                )
                """
            )

    def save_task(self, task: "Task") -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO tasks (id,title,description,status,result,subtasks) VALUES (?,?,?,?,?,?)",
                (task.id, task.title, task.description, task.status, str(task.result), ";".join(task.subtasks)),
            )

    def update_task(self, task: "Task") -> None:
        self.save_task(task)

    def load_all_tasks(self) -> List["Task"]:
        from .manager import Task

        out: List[Task] = []
        with self._conn() as c:
            for row in c.execute("SELECT id,title,description,status,result,subtasks FROM tasks"):
                tid, title, desc, status, result, subtasks = row
                t = Task(id=tid, title=title, description=desc)
                t.status = status
                t.result = result
                t.subtasks = subtasks.split(";") if subtasks else []
                out.append(t)
        return out

    def load_task(self, task_id: str) -> Optional["Task"]:
        from .manager import Task

        with self._conn() as c:
            row = c.execute("SELECT id,title,description,status,result,subtasks FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not row:
                return None
            tid, title, desc, status, result, subtasks = row
            t = Task(id=tid, title=title, description=desc)
            t.status = status
            t.result = result
            t.subtasks = subtasks.split(";") if subtasks else []
            return t
