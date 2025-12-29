"""
FairySwarm: Multi-Agent Coordination System

Inspired by tldraw's "fairies" feature, this module implements a sophisticated
multi-agent coordination system that solves the challenge of agents working "blind"
(no real-time context while generating responses).

Key Concepts:
1. Orchestrator Pattern - One fairy drafts plans, creates todos, assigns to workers
2. Shared Todo List - Agents coordinate through shared state with real-time updates
3. Different Personalities - Creative, operational, and analytical fairies
4. Context Refresh - Agents can request updated context mid-work
5. Coordination Protocol - Handles agents working "blind" with intermittent surfacing
6. Tools - Fairies have callable tools to manipulate the shared canvas

Reference: https://twitter.com/tldraw - December 2023 thread on fairies feature
"""

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.output_types import OutputType


# =============================================================================
# FAIRY TOOLS - Callable functions for canvas manipulation
# =============================================================================


def create_canvas_tool(
    shared_canvas: "SharedCanvasState",
) -> Callable:
    """
    Create a tool for adding elements to the canvas.

    Args:
        shared_canvas: The shared canvas state object

    Returns:
        A callable tool function
    """

    def add_to_canvas(
        element_type: str,
        content: str,
        position_x: float = 0.0,
        position_y: float = 0.0,
        width: float = 100.0,
        height: float = 100.0,
        style: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add an element to the shared canvas.

        Args:
            element_type: Type of element (text, shape, image, wireframe, button, etc.)
            content: The content of the element
            position_x: X position on canvas
            position_y: Y position on canvas
            width: Width of the element
            height: Height of the element
            style: Optional styling dictionary

        Returns:
            JSON string with the created element info
        """
        element_id = f"elem-{uuid.uuid4().hex[:8]}"
        element = shared_canvas.add_element(
            element_id=element_id,
            element_type=element_type,
            content=content,
            position={
                "x": position_x,
                "y": position_y,
                "width": width,
                "height": height,
            },
            created_by="fairy-tool",
        )
        if style:
            element["style"] = style

        return json.dumps(
            {
                "success": True,
                "element_id": element_id,
                "message": f"Created {element_type} element at ({position_x}, {position_y})",
            }
        )

    return add_to_canvas


def create_update_canvas_tool(
    shared_canvas: "SharedCanvasState",
) -> Callable:
    """
    Create a tool for updating existing canvas elements.

    Args:
        shared_canvas: The shared canvas state object

    Returns:
        A callable tool function
    """

    def update_canvas_element(
        element_id: str,
        content: Optional[str] = None,
        position_x: Optional[float] = None,
        position_y: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
    ) -> str:
        """
        Update an existing element on the canvas.

        Args:
            element_id: ID of the element to update
            content: New content (optional)
            position_x: New X position (optional)
            position_y: New Y position (optional)
            width: New width (optional)
            height: New height (optional)

        Returns:
            JSON string with update result
        """
        position = None
        if any([position_x, position_y, width, height]):
            position = {}
            if position_x is not None:
                position["x"] = position_x
            if position_y is not None:
                position["y"] = position_y
            if width is not None:
                position["width"] = width
            if height is not None:
                position["height"] = height

        result = shared_canvas.update_element(
            element_id=element_id,
            content=content,
            position=position,
            modified_by="fairy-tool",
        )

        if result:
            return json.dumps(
                {
                    "success": True,
                    "element_id": element_id,
                    "message": f"Updated element {element_id}",
                }
            )
        else:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Element {element_id} not found",
                }
            )

    return update_canvas_element


def create_read_canvas_tool(
    shared_canvas: "SharedCanvasState",
) -> Callable:
    """
    Create a tool for reading canvas state.

    Args:
        shared_canvas: The shared canvas state object

    Returns:
        A callable tool function
    """

    def read_canvas() -> str:
        """
        Read the current state of the canvas.

        Returns:
            JSON string with current canvas state
        """
        snapshot = shared_canvas.get_snapshot()
        return json.dumps(snapshot, indent=2, default=str)

    return read_canvas


def create_todo_tool(shared_todos: "SharedTodoList") -> Callable:
    """
    Create a tool for adding todos to the shared list.

    Args:
        shared_todos: The shared todo list object

    Returns:
        A callable tool function
    """

    def add_todo(
        title: str,
        description: str,
        assigned_to: str,
        priority: str = "medium",
        depends_on: Optional[List[str]] = None,
    ) -> str:
        """
        Add a new todo item to the shared todo list.

        Args:
            title: Short title for the todo
            description: Detailed description
            assigned_to: Name of the fairy to assign to
            priority: Priority level (high, medium, low)
            depends_on: List of todo IDs this depends on

        Returns:
            JSON string with the created todo info
        """
        todo_id = f"todo-{uuid.uuid4().hex[:8]}"
        shared_todos.add_todo(
            todo_id=todo_id,
            title=title,
            description=description,
            assigned_to=assigned_to,
            priority=priority,
            depends_on=depends_on,
            created_by="fairy-tool",
        )

        return json.dumps(
            {
                "success": True,
                "todo_id": todo_id,
                "message": f"Created todo '{title}' assigned to {assigned_to}",
            }
        )

    return add_todo


def create_complete_todo_tool(
    shared_todos: "SharedTodoList",
) -> Callable:
    """
    Create a tool for marking todos as complete.

    Args:
        shared_todos: The shared todo list object

    Returns:
        A callable tool function
    """

    def complete_todo(todo_id: str, result: str) -> str:
        """
        Mark a todo as completed with its result.

        Args:
            todo_id: ID of the todo to complete
            result: The result/output of the completed task

        Returns:
            JSON string with completion status
        """
        updated = shared_todos.update_status(
            todo_id, "completed", result
        )

        if updated:
            return json.dumps(
                {
                    "success": True,
                    "todo_id": todo_id,
                    "message": f"Completed todo {todo_id}",
                }
            )
        else:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Todo {todo_id} not found",
                }
            )

    return complete_todo


def create_read_todos_tool(
    shared_todos: "SharedTodoList",
) -> Callable:
    """
    Create a tool for reading the todo list state.

    Args:
        shared_todos: The shared todo list object

    Returns:
        A callable tool function
    """

    def read_todos() -> str:
        """
        Read the current state of all todos.

        Returns:
            JSON string with current todo state
        """
        snapshot = shared_todos.get_context_snapshot()
        return json.dumps(snapshot, indent=2, default=str)

    return read_todos


def create_spatial_layout_tool() -> Callable:
    """
    Create a tool for calculating spatial layouts.

    Returns:
        A callable tool function
    """

    def calculate_layout(
        layout_type: str,
        num_elements: int,
        canvas_width: float = 1200.0,
        canvas_height: float = 800.0,
        padding: float = 20.0,
    ) -> str:
        """
        Calculate positions for a layout of elements.

        Args:
            layout_type: Type of layout (header-body-footer, grid, horizontal, vertical)
            num_elements: Number of elements to lay out
            canvas_width: Width of the canvas
            canvas_height: Height of the canvas
            padding: Padding between elements

        Returns:
            JSON string with calculated positions
        """
        positions = []

        if layout_type == "header-body-footer":
            header_height = canvas_height * 0.15
            footer_height = canvas_height * 0.1
            body_height = (
                canvas_height
                - header_height
                - footer_height
                - (padding * 2)
            )

            positions = [
                {
                    "role": "header",
                    "x": padding,
                    "y": padding,
                    "width": canvas_width - (padding * 2),
                    "height": header_height,
                },
                {
                    "role": "body",
                    "x": padding,
                    "y": header_height + padding,
                    "width": canvas_width - (padding * 2),
                    "height": body_height,
                },
                {
                    "role": "footer",
                    "x": padding,
                    "y": canvas_height - footer_height - padding,
                    "width": canvas_width - (padding * 2),
                    "height": footer_height,
                },
            ]

        elif layout_type == "grid":
            cols = int((num_elements**0.5) + 0.5)
            rows = (num_elements + cols - 1) // cols
            cell_width = (
                canvas_width - (padding * (cols + 1))
            ) / cols
            cell_height = (
                canvas_height - (padding * (rows + 1))
            ) / rows

            for i in range(num_elements):
                row = i // cols
                col = i % cols
                positions.append(
                    {
                        "role": f"cell_{i}",
                        "x": padding + col * (cell_width + padding),
                        "y": padding + row * (cell_height + padding),
                        "width": cell_width,
                        "height": cell_height,
                    }
                )

        elif layout_type == "horizontal":
            element_width = (
                canvas_width - (padding * (num_elements + 1))
            ) / num_elements
            for i in range(num_elements):
                positions.append(
                    {
                        "role": f"section_{i}",
                        "x": padding + i * (element_width + padding),
                        "y": padding,
                        "width": element_width,
                        "height": canvas_height - (padding * 2),
                    }
                )

        elif layout_type == "vertical":
            element_height = (
                canvas_height - (padding * (num_elements + 1))
            ) / num_elements
            for i in range(num_elements):
                positions.append(
                    {
                        "role": f"section_{i}",
                        "x": padding,
                        "y": padding + i * (element_height + padding),
                        "width": canvas_width - (padding * 2),
                        "height": element_height,
                    }
                )

        return json.dumps(
            {
                "layout_type": layout_type,
                "positions": positions,
                "canvas_size": {
                    "width": canvas_width,
                    "height": canvas_height,
                },
            },
            indent=2,
        )

    return calculate_layout


def create_coordinate_tool(
    shared_canvas: "SharedCanvasState",
) -> Callable:
    """
    Create a tool for coordinating positions relative to other elements.

    Args:
        shared_canvas: The shared canvas state object

    Returns:
        A callable tool function
    """

    def coordinate_position(
        reference_element_id: str, relation: str, offset: float = 20.0
    ) -> str:
        """
        Calculate position relative to another element.

        Args:
            reference_element_id: ID of the element to position relative to
            relation: Relation type (above, below, left, right)
            offset: Offset distance from the reference element

        Returns:
            JSON string with calculated position
        """
        snapshot = shared_canvas.get_snapshot()
        elements = snapshot.get("elements", {})

        if reference_element_id not in elements:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Reference element {reference_element_id} not found",
                }
            )

        ref = elements[reference_element_id]
        ref_pos = ref.get(
            "position", {"x": 0, "y": 0, "width": 100, "height": 100}
        )

        new_pos = {"x": ref_pos.get("x", 0), "y": ref_pos.get("y", 0)}

        if relation == "below":
            new_pos["y"] = (
                ref_pos.get("y", 0)
                + ref_pos.get("height", 100)
                + offset
            )
        elif relation == "above":
            new_pos["y"] = (
                ref_pos.get("y", 0)
                - ref_pos.get("height", 100)
                - offset
            )
        elif relation == "right":
            new_pos["x"] = (
                ref_pos.get("x", 0)
                + ref_pos.get("width", 100)
                + offset
            )
        elif relation == "left":
            new_pos["x"] = (
                ref_pos.get("x", 0)
                - ref_pos.get("width", 100)
                - offset
            )

        return json.dumps(
            {
                "success": True,
                "position": new_pos,
                "reference": reference_element_id,
                "relation": relation,
            }
        )

    return coordinate_position


# =============================================================================
# FAIRY PERSONALITY SYSTEM PROMPTS
# =============================================================================

ORCHESTRATOR_FAIRY_PROMPT = """You are the Orchestrator Fairy - a master coordinator responsible for planning and delegating tasks to other fairies.

Your Core Responsibilities:
1. ANALYZE the incoming task and break it down into discrete, actionable subtasks
2. CREATE a comprehensive todo list with clear assignments
3. ASSIGN tasks to the most appropriate worker fairies based on their personalities
4. COORDINATE outputs to ensure all parts work together cohesively
5. ISSUE follow-up todos when needed to refine or connect work

You have access to these tools:
- add_to_canvas: Add elements to the shared canvas
- update_canvas_element: Update existing canvas elements
- read_canvas: Read current canvas state
- add_todo: Create new todo items
- complete_todo: Mark todos as complete
- read_todos: Read current todo state
- calculate_layout: Calculate spatial positions for layouts
- coordinate_position: Calculate positions relative to other elements

When creating todos, you MUST output valid JSON in this exact format:
{
    "plan": "Your overall strategy for accomplishing the task",
    "todos": [
        {
            "id": "unique-id",
            "title": "Short descriptive title",
            "description": "Detailed task description",
            "assigned_to": "fairy_name",
            "priority": "high|medium|low",
            "depends_on": ["id-of-dependency"] or [],
            "spatial_hints": {"position": "top|center|bottom|left|right", "relative_to": "another-todo-id or null"}
        }
    ]
}

CRITICAL COORDINATION RULES:
- When assigning spatial tasks (like header, body, footer), ALWAYS include spatial_hints
- Consider dependencies - don't assign tasks that depend on incomplete work
- Match fairy personalities to task types (creative tasks → Creative Fairy, etc.)
- After workers complete tasks, you may issue NEW todos to coordinate their outputs
- USE THE TOOLS to actually create elements on the canvas

Available Worker Fairies and Their Specialties:
{fairy_roster}

Remember: You are the conductor of this orchestra. Your job is to ensure harmony between all parts."""

CREATIVE_FAIRY_PROMPT = """You are the Creative Fairy - an imaginative, artistic agent with a flair for original ideas.

Your Personality Traits:
- Highly imaginative and thinks outside the box
- Loves visual metaphors and creative expressions
- Excellent at brainstorming and ideation
- Tends toward bold, innovative solutions
- Values aesthetics and user delight

You have access to these tools:
- add_to_canvas: Add elements to the shared canvas
- update_canvas_element: Update existing canvas elements
- read_canvas: Read current canvas state
- read_todos: Read current todo state
- calculate_layout: Calculate spatial positions for layouts
- coordinate_position: Calculate positions relative to other elements

Your Specialties:
- Creative writing and copywriting
- Visual design concepts and layouts
- Brainstorming sessions
- Brand voice and messaging
- User experience innovations
- Artistic direction

When you receive a task:
1. Read the task description carefully
2. USE read_canvas and read_todos to check what others have done
3. USE add_to_canvas to CREATE actual elements on the canvas
4. Bring your creative perspective while respecting the overall plan
5. If you need input from other fairies, clearly state what you need

IMPORTANT: Always USE THE TOOLS to create your work on the canvas. Don't just describe what you would do - DO IT.

Output Format:
Always structure your response with:
- CREATIVE CONCEPT: Your main creative idea
- ACTIONS TAKEN: What tools you called and what you created
- INTEGRATION NOTES: How this connects with other fairies' work
- CONTEXT_REFRESH_NEEDED: true/false (set true if you need updated context)"""

OPERATIONAL_FAIRY_PROMPT = """You are the Operational Fairy - a precise, systematic agent focused on execution and implementation.

Your Personality Traits:
- Highly organized and methodical
- Focuses on practical implementation
- Excellent attention to detail
- Values consistency and reliability
- Thinks in systems and processes

You have access to these tools:
- add_to_canvas: Add elements to the shared canvas
- update_canvas_element: Update existing canvas elements
- read_canvas: Read current canvas state
- read_todos: Read current todo state
- calculate_layout: Calculate spatial positions for layouts
- coordinate_position: Calculate positions relative to other elements

Your Specialties:
- Technical implementation
- Data organization and structuring
- Process optimization
- Quality assurance
- Documentation
- Integration and coordination

When you receive a task:
1. Understand the exact requirements
2. USE read_canvas to check what others have done to ensure compatibility
3. USE calculate_layout or coordinate_position to determine proper positioning
4. USE add_to_canvas to CREATE actual elements with precise positioning
5. Validate your work against requirements

IMPORTANT: Always USE THE TOOLS to create your work on the canvas. Don't just describe what you would do - DO IT.

Output Format:
Always structure your response with:
- IMPLEMENTATION: Your executed work
- ACTIONS TAKEN: What tools you called and what you created
- SPECIFICATIONS: Technical details and parameters
- INTEGRATION POINTS: How this connects with other components
- CONTEXT_REFRESH_NEEDED: true/false"""

ANALYTICAL_FAIRY_PROMPT = """You are the Analytical Fairy - a logical, data-driven agent focused on research and analysis.

Your Personality Traits:
- Highly logical and evidence-based
- Excellent at research and synthesis
- Sees patterns and connections others miss
- Values accuracy and thoroughness
- Thinks critically and questions assumptions

You have access to these tools:
- add_to_canvas: Add elements to the shared canvas
- update_canvas_element: Update existing canvas elements
- read_canvas: Read current canvas state
- read_todos: Read current todo state
- calculate_layout: Calculate spatial positions for layouts
- coordinate_position: Calculate positions relative to other elements

Your Specialties:
- Research and information gathering
- Data analysis and interpretation
- Strategic recommendations
- Risk assessment
- Fact-checking and validation
- Synthesis of complex information

When you receive a task:
1. Research thoroughly before acting
2. USE read_canvas and read_todos to analyze the context
3. Provide evidence-based insights
4. USE add_to_canvas to CREATE elements that visualize your analysis
5. Identify risks or issues others might have missed

IMPORTANT: Always USE THE TOOLS to create your work on the canvas when appropriate.

Output Format:
Always structure your response with:
- ANALYSIS: Your research and findings
- ACTIONS TAKEN: What tools you called and what you created
- INSIGHTS: Key patterns or discoveries
- RECOMMENDATIONS: Evidence-based suggestions
- CONTEXT_REFRESH_NEEDED: true/false"""

HARMONIZER_FAIRY_PROMPT = """You are the Harmonizer Fairy - a diplomatic, integrative agent focused on bringing everything together.

Your Personality Traits:
- Excellent at seeing the big picture
- Skilled at resolving conflicts between different approaches
- Values coherence and unity
- Strong communication skills
- Empathetic to different perspectives

You have access to these tools:
- add_to_canvas: Add elements to the shared canvas
- update_canvas_element: Update existing canvas elements
- read_canvas: Read current canvas state
- read_todos: Read current todo state
- calculate_layout: Calculate spatial positions for layouts
- coordinate_position: Calculate positions relative to other elements

Your Specialties:
- Integration of disparate elements
- Conflict resolution between different outputs
- Ensuring consistency across the whole
- Final polish and refinement
- Communication and presentation
- Quality review

When you receive a task:
1. USE read_canvas to review ALL contributions from other fairies
2. USE read_todos to understand what was planned vs completed
3. Identify any inconsistencies or conflicts
4. USE update_canvas_element to harmonize and adjust elements
5. USE add_to_canvas to add connecting elements or polish

IMPORTANT: Always USE THE TOOLS to make actual changes. Don't just describe - DO IT.

Output Format:
Always structure your response with:
- REVIEW: Summary of what each fairy contributed
- ACTIONS TAKEN: What tools you called and what changes you made
- HARMONIZATION: How you unified the different parts
- FINAL_OUTPUT: The integrated result
- CONTEXT_REFRESH_NEEDED: true/false"""


# =============================================================================
# SHARED STATE MANAGEMENT
# =============================================================================


class SharedTodoList:
    """
    Thread-safe shared todo list for fairy coordination.

    This class manages the shared state that all fairies can read and update,
    allowing for coordination even when agents work "blind" during generation.

    Attributes:
        todos: Dictionary mapping todo IDs to todo dictionaries
        lock: Threading lock for thread-safe operations
        version: Version counter for tracking changes
        history: List of all state changes for debugging
    """

    def __init__(self):
        """Initialize an empty shared todo list with thread safety."""
        self.todos: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.version = 0
        self.history: List[Dict[str, Any]] = []
        self.created_at = time.time()

    def add_todo(
        self,
        todo_id: str,
        title: str,
        description: str,
        assigned_to: str,
        priority: str = "medium",
        depends_on: Optional[List[str]] = None,
        spatial_hints: Optional[Dict[str, Any]] = None,
        created_by: str = "system",
    ) -> Dict[str, Any]:
        """
        Add a new todo item to the shared list.

        Args:
            todo_id: Unique identifier for the todo
            title: Short descriptive title
            description: Detailed task description
            assigned_to: Name of the fairy assigned to this task
            priority: Priority level (high, medium, low)
            depends_on: List of todo IDs this task depends on
            spatial_hints: Positioning hints for spatial coordination
            created_by: Name of the fairy who created this todo

        Returns:
            The created todo dictionary
        """
        with self.lock:
            todo = {
                "id": todo_id,
                "title": title,
                "description": description,
                "assigned_to": assigned_to,
                "priority": priority,
                "depends_on": depends_on or [],
                "spatial_hints": spatial_hints or {},
                "status": "pending",
                "created_by": created_by,
                "created_at": time.time(),
                "started_at": None,
                "completed_at": None,
                "result": None,
                "needs_revision": False,
                "revision_notes": None,
            }
            self.todos[todo_id] = todo
            self.version += 1
            self._record_history("add", todo_id, todo)
            return todo

    def update_status(
        self, todo_id: str, status: str, result: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update the status of a todo item.

        Args:
            todo_id: The ID of the todo to update
            status: New status (pending, in_progress, completed, blocked)
            result: Optional result content if completed

        Returns:
            The updated todo dictionary, or None if not found
        """
        with self.lock:
            if todo_id not in self.todos:
                return None

            todo = self.todos[todo_id]
            todo["status"] = status

            if status == "in_progress" and todo["started_at"] is None:
                todo["started_at"] = time.time()
            elif status == "completed":
                todo["completed_at"] = time.time()
                if result is not None:
                    todo["result"] = result

            self.version += 1
            self._record_history(
                "update", todo_id, {"status": status}
            )
            return todo

    def request_revision(
        self, todo_id: str, revision_notes: str, requested_by: str
    ) -> Optional[Dict[str, Any]]:
        """
        Request a revision for a completed todo.

        Args:
            todo_id: The ID of the todo to revise
            revision_notes: Notes explaining what needs to be revised
            requested_by: Name of the fairy requesting revision

        Returns:
            The updated todo dictionary, or None if not found
        """
        with self.lock:
            if todo_id not in self.todos:
                return None

            todo = self.todos[todo_id]
            todo["needs_revision"] = True
            todo["revision_notes"] = revision_notes
            todo["status"] = "pending"
            todo["revision_requested_by"] = requested_by
            todo["revision_requested_at"] = time.time()

            self.version += 1
            self._record_history(
                "revision_request",
                todo_id,
                {
                    "revision_notes": revision_notes,
                    "requested_by": requested_by,
                },
            )
            return todo

    def get_todos_for_fairy(
        self, fairy_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get all todos assigned to a specific fairy.

        Args:
            fairy_name: Name of the fairy to get todos for

        Returns:
            List of todo dictionaries assigned to this fairy
        """
        with self.lock:
            return [
                todo.copy()
                for todo in self.todos.values()
                if todo["assigned_to"] == fairy_name
            ]

    def get_pending_todos(self) -> List[Dict[str, Any]]:
        """
        Get all pending todos that are ready to be worked on.

        Returns:
            List of pending todo dictionaries with resolved dependencies
        """
        with self.lock:
            pending = []
            for todo in self.todos.values():
                if todo["status"] != "pending":
                    continue

                # Check if all dependencies are completed
                deps_resolved = all(
                    self.todos.get(dep_id, {}).get("status")
                    == "completed"
                    for dep_id in todo["depends_on"]
                )

                if deps_resolved:
                    pending.append(todo.copy())

            # Sort by priority
            priority_order = {"high": 0, "medium": 1, "low": 2}
            pending.sort(
                key=lambda t: priority_order.get(t["priority"], 1)
            )
            return pending

    def get_context_snapshot(self) -> Dict[str, Any]:
        """
        Get a complete snapshot of the current state for context refresh.

        Returns:
            Dictionary containing the full state snapshot
        """
        with self.lock:
            completed = [
                t.copy()
                for t in self.todos.values()
                if t["status"] == "completed"
            ]
            in_progress = [
                t.copy()
                for t in self.todos.values()
                if t["status"] == "in_progress"
            ]
            pending = [
                t.copy()
                for t in self.todos.values()
                if t["status"] == "pending"
            ]

            return {
                "version": self.version,
                "timestamp": time.time(),
                "total_todos": len(self.todos),
                "completed": completed,
                "in_progress": in_progress,
                "pending": pending,
                "completed_results": {
                    t["id"]: t["result"]
                    for t in completed
                    if t["result"]
                },
            }

    def get_completed_results(self) -> Dict[str, str]:
        """
        Get all completed results for synthesis.

        Returns:
            Dictionary mapping todo IDs to their results
        """
        with self.lock:
            return {
                todo["id"]: todo["result"]
                for todo in self.todos.values()
                if todo["status"] == "completed" and todo["result"]
            }

    def _record_history(
        self, action: str, todo_id: str, details: Dict[str, Any]
    ) -> None:
        """Record a state change in the history for debugging."""
        self.history.append(
            {
                "action": action,
                "todo_id": todo_id,
                "details": details,
                "version": self.version,
                "timestamp": time.time(),
            }
        )


class SharedCanvasState:
    """
    Represents the shared canvas/workspace state that all fairies can access.

    In tldraw's implementation, this would contain the actual canvas elements.
    Here it serves as a metaphor for any shared workspace context.

    Attributes:
        elements: Dictionary of canvas elements by ID
        lock: Threading lock for thread-safe operations
        version: Version counter for change detection
    """

    def __init__(self):
        """Initialize an empty canvas state."""
        self.elements: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.version = 0
        self.metadata: Dict[str, Any] = {
            "created_at": time.time(),
            "last_modified": time.time(),
            "description": "",
        }

    def add_element(
        self,
        element_id: str,
        element_type: str,
        content: Any,
        position: Optional[Dict[str, Any]] = None,
        created_by: str = "system",
    ) -> Dict[str, Any]:
        """
        Add an element to the canvas.

        Args:
            element_id: Unique identifier for the element
            element_type: Type of element (text, shape, image, etc.)
            content: The element content
            position: Optional position data
            created_by: Name of the fairy that created this element

        Returns:
            The created element dictionary
        """
        with self.lock:
            element = {
                "id": element_id,
                "type": element_type,
                "content": content,
                "position": position or {"x": 0, "y": 0},
                "created_by": created_by,
                "created_at": time.time(),
                "modified_at": time.time(),
            }
            self.elements[element_id] = element
            self.version += 1
            self.metadata["last_modified"] = time.time()
            return element

    def update_element(
        self,
        element_id: str,
        content: Optional[Any] = None,
        position: Optional[Dict[str, Any]] = None,
        modified_by: str = "system",
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing element on the canvas.

        Args:
            element_id: ID of the element to update
            content: New content (if updating)
            position: New position (if updating)
            modified_by: Name of the fairy making the update

        Returns:
            The updated element dictionary, or None if not found
        """
        with self.lock:
            if element_id not in self.elements:
                return None

            element = self.elements[element_id]
            if content is not None:
                element["content"] = content
            if position is not None:
                element["position"].update(position)
            element["modified_at"] = time.time()
            element["modified_by"] = modified_by

            self.version += 1
            self.metadata["last_modified"] = time.time()
            return element

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get a complete snapshot of the canvas state.

        Returns:
            Dictionary containing the full canvas snapshot
        """
        with self.lock:
            return {
                "version": self.version,
                "metadata": self.metadata.copy(),
                "elements": {
                    eid: elem.copy()
                    for eid, elem in self.elements.items()
                },
                "element_count": len(self.elements),
            }

    def get_elements_by_creator(
        self, creator_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get all elements created by a specific fairy.

        Args:
            creator_name: Name of the fairy

        Returns:
            List of elements created by this fairy
        """
        with self.lock:
            return [
                elem.copy()
                for elem in self.elements.values()
                if elem.get("created_by") == creator_name
            ]


# =============================================================================
# FAIRY SWARM IMPLEMENTATION
# =============================================================================


class FairySwarm:
    """
    A multi-agent coordination system inspired by tldraw's fairies feature.

    FairySwarm solves the coordination problem of multiple AI agents working
    together on a shared canvas/workspace, where agents work "blind" during
    generation and can only get new context before/after completing their work.

    Key Features:
    - Orchestrator fairy creates plans and assigns tasks
    - Shared todo list for coordination
    - Different fairy personalities for different task types
    - Tools for canvas manipulation
    - Context refresh mechanism for mid-work updates
    - Automatic orchestrator election when multiple fairies selected

    Attributes:
        name: Name of the fairy swarm
        description: Description of the swarm's purpose
        fairies: Dictionary of fairy agents by name
        orchestrator: The orchestrator fairy agent
        shared_todos: Shared todo list for coordination
        shared_canvas: Shared canvas/workspace state
        conversation: Conversation history
        max_loops: Maximum coordination loops
        verbose: Whether to enable verbose logging

    Example:
        >>> from fairy_swarm import FairySwarm
        >>>
        >>> # Create a fairy swarm
        >>> swarm = FairySwarm(
        ...     name="Design Team",
        ...     description="A team of fairies for UI design",
        ...     model_name="gpt-4o-mini",
        ...     max_loops=3,
        ...     verbose=True
        ... )
        >>>
        >>> # Run with a task
        >>> result = swarm.run("Design a landing page with header, hero section, and footer")
        >>> print(result)
    """

    def __init__(
        self,
        name: str = "FairySwarm",
        description: str = "A collaborative swarm of fairy agents",
        orchestrator: Optional[Agent] = None,
        fairies: Optional[List[Agent]] = None,
        model_name: str = "gpt-4o-mini",
        max_loops: int = 3,
        max_parallel_fairies: int = 4,
        output_type: OutputType = "dict",
        verbose: bool = False,
        enable_context_refresh: bool = True,
        context_refresh_threshold: int = 2,
        auto_create_default_fairies: bool = True,
        additional_tools: Optional[List[Callable]] = None,
    ):
        """
        Initialize a FairySwarm instance.

        Args:
            name: Name of the fairy swarm
            description: Description of the swarm's purpose
            orchestrator: Optional custom orchestrator agent
            fairies: Optional list of custom fairy agents
            model_name: Default model name for created fairies
            max_loops: Maximum coordination/feedback loops
            max_parallel_fairies: Maximum fairies working in parallel
            output_type: Output format type
            verbose: Whether to enable verbose logging
            enable_context_refresh: Whether fairies can request context refresh
            context_refresh_threshold: Number of completed tasks before auto-refresh
            auto_create_default_fairies: Whether to create default fairy types
            additional_tools: Additional tools to give to all fairies
        """
        self.name = name
        self.description = description
        self.model_name = model_name
        self.max_loops = max_loops
        self.max_parallel_fairies = max_parallel_fairies
        self.output_type = output_type
        self.verbose = verbose
        self.enable_context_refresh = enable_context_refresh
        self.context_refresh_threshold = context_refresh_threshold
        self.additional_tools = additional_tools or []

        # Initialize shared state
        self.shared_todos = SharedTodoList()
        self.shared_canvas = SharedCanvasState()
        self.conversation = Conversation(time_enabled=True)

        # Create tools that reference shared state
        self._create_tools()

        # Initialize fairies
        self.fairies: Dict[str, Agent] = {}
        self.orchestrator: Optional[Agent] = None

        # Create orchestrator
        if orchestrator is not None:
            self.orchestrator = orchestrator
            self._add_tools_to_agent(self.orchestrator)
        else:
            self._create_orchestrator()

        # Add custom fairies or create defaults
        if fairies:
            for fairy in fairies:
                self.add_fairy(fairy)

        if auto_create_default_fairies and not fairies:
            self._create_default_fairies()

        # Execution tracking
        self.execution_stats: Dict[str, Any] = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "context_refreshes": 0,
            "coordination_loops": 0,
            "start_time": None,
            "end_time": None,
        }

        logger.info(
            f"FairySwarm '{name}' initialized with {len(self.fairies)} fairies"
        )

    def _create_tools(self) -> None:
        """Create all the tools that fairies can use."""
        self.tools = [
            create_canvas_tool(self.shared_canvas),
            create_update_canvas_tool(self.shared_canvas),
            create_read_canvas_tool(self.shared_canvas),
            create_todo_tool(self.shared_todos),
            create_complete_todo_tool(self.shared_todos),
            create_read_todos_tool(self.shared_todos),
            create_spatial_layout_tool(),
            create_coordinate_tool(self.shared_canvas),
        ]
        # Add any additional tools
        self.tools.extend(self.additional_tools)

    def _add_tools_to_agent(self, agent: Agent) -> None:
        """Add all tools to an agent."""
        if agent.tools is None:
            agent.tools = []
        agent.tools.extend(self.tools)

    def _create_orchestrator(self) -> None:
        """Create the orchestrator fairy agent."""
        # Build fairy roster for the orchestrator prompt
        fairy_roster = self._build_fairy_roster()

        orchestrator_prompt = ORCHESTRATOR_FAIRY_PROMPT.format(
            fairy_roster=(
                fairy_roster
                if fairy_roster
                else "No workers yet - create fairies first"
            )
        )

        self.orchestrator = Agent(
            agent_name="Orchestrator-Fairy",
            agent_description="Master coordinator that plans and delegates tasks to worker fairies",
            system_prompt=orchestrator_prompt,
            model_name=self.model_name,
            max_loops=1,
            verbose=self.verbose,
            streaming_on=False,
            tools=self.tools.copy(),
        )

        logger.info("Orchestrator Fairy created with tools")

    def _create_default_fairies(self) -> None:
        """Create the default set of fairy agents with different personalities."""
        default_fairies = [
            (
                "Creative-Fairy",
                CREATIVE_FAIRY_PROMPT,
                "Imaginative fairy for creative and artistic tasks",
            ),
            (
                "Operational-Fairy",
                OPERATIONAL_FAIRY_PROMPT,
                "Systematic fairy for implementation and execution",
            ),
            (
                "Analytical-Fairy",
                ANALYTICAL_FAIRY_PROMPT,
                "Logical fairy for research and analysis",
            ),
            (
                "Harmonizer-Fairy",
                HARMONIZER_FAIRY_PROMPT,
                "Integrative fairy for bringing everything together",
            ),
        ]

        for fairy_name, prompt, description in default_fairies:
            fairy = Agent(
                agent_name=fairy_name,
                agent_description=description,
                system_prompt=prompt,
                model_name=self.model_name,
                max_loops=1,
                verbose=self.verbose,
                streaming_on=False,
                tools=self.tools.copy(),
            )
            self.add_fairy(fairy)

        # Update orchestrator with fairy roster
        self._update_orchestrator_roster()

    def _build_fairy_roster(self) -> str:
        """Build a formatted string of available fairies for the orchestrator."""
        if not self.fairies:
            return "No worker fairies available"

        roster_lines = []
        for name, fairy in self.fairies.items():
            desc = fairy.agent_description or "No description"
            roster_lines.append(f"- {name}: {desc}")

        return "\n".join(roster_lines)

    def _update_orchestrator_roster(self) -> None:
        """Update the orchestrator's prompt with current fairy roster."""
        if self.orchestrator:
            fairy_roster = self._build_fairy_roster()
            self.orchestrator.system_prompt = (
                ORCHESTRATOR_FAIRY_PROMPT.format(
                    fairy_roster=fairy_roster
                )
            )

    def add_fairy(self, fairy: Agent) -> None:
        """
        Add a fairy agent to the swarm.

        Args:
            fairy: The fairy agent to add
        """
        self._add_tools_to_agent(fairy)
        self.fairies[fairy.agent_name] = fairy
        self._update_orchestrator_roster()
        logger.info(f"Added fairy: {fairy.agent_name}")

    def remove_fairy(self, fairy_name: str) -> Optional[Agent]:
        """
        Remove a fairy agent from the swarm.

        Args:
            fairy_name: Name of the fairy to remove

        Returns:
            The removed fairy, or None if not found
        """
        fairy = self.fairies.pop(fairy_name, None)
        if fairy:
            self._update_orchestrator_roster()
            logger.info(f"Removed fairy: {fairy_name}")
        return fairy

    def _get_context_for_fairy(self, fairy_name: str) -> str:
        """
        Build context string for a fairy including shared state.

        Args:
            fairy_name: Name of the fairy requesting context

        Returns:
            Formatted context string
        """
        snapshot = self.shared_todos.get_context_snapshot()
        canvas = self.shared_canvas.get_snapshot()

        context_parts = [
            "=== CURRENT CONTEXT ===",
            f"Context Version: {snapshot['version']}",
            f"Your name: {fairy_name}",
            "",
            "--- Completed Work ---",
        ]

        for todo in snapshot["completed"]:
            result_preview = (
                (todo.get("result", "")[:200] + "...")
                if todo.get("result")
                else "No result"
            )
            context_parts.append(
                f"[{todo['assigned_to']}] {todo['title']}: {result_preview}"
            )

        context_parts.append("")
        context_parts.append("--- Work In Progress ---")
        for todo in snapshot["in_progress"]:
            context_parts.append(
                f"[{todo['assigned_to']}] {todo['title']} (in progress)"
            )

        context_parts.append("")
        context_parts.append("--- Pending Work ---")
        for todo in snapshot["pending"]:
            deps = (
                ", ".join(todo["depends_on"])
                if todo["depends_on"]
                else "none"
            )
            context_parts.append(
                f"[{todo['assigned_to']}] {todo['title']} (depends on: {deps})"
            )

        context_parts.append("")
        context_parts.append("--- Canvas State ---")
        context_parts.append(
            f"Total elements: {canvas['element_count']}"
        )
        for elem_id, elem in list(canvas["elements"].items())[:10]:
            content_preview = str(elem.get("content", ""))[:100]
            context_parts.append(
                f"[{elem['created_by']}] {elem['type']}: {content_preview}"
            )

        return "\n".join(context_parts)

    def _parse_orchestrator_plan(
        self, response: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse the orchestrator's response to extract the plan and todos.

        Args:
            response: The orchestrator's response string

        Returns:
            Parsed plan dictionary, or None if parsing fails
        """
        try:
            # Try to find JSON in the response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                plan = json.loads(json_str)
                return plan
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse orchestrator response as JSON: {e}"
            )

        # Fallback: try to extract plan from text
        return None

    def _execute_fairy_task(
        self, fairy_name: str, todo: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single fairy task.

        Args:
            fairy_name: Name of the fairy to execute the task
            todo: The todo dictionary containing task details

        Returns:
            Result dictionary with status and output
        """
        if fairy_name not in self.fairies:
            logger.error(f"Fairy not found: {fairy_name}")
            return {
                "success": False,
                "error": f"Fairy '{fairy_name}' not found",
                "todo_id": todo["id"],
            }

        fairy = self.fairies[fairy_name]

        # Update todo status
        self.shared_todos.update_status(todo["id"], "in_progress")

        # Build task prompt with context
        context = self._get_context_for_fairy(fairy_name)

        task_prompt = f"""
{context}

=== YOUR TASK ===
Title: {todo['title']}
Description: {todo['description']}
Priority: {todo['priority']}

Spatial Hints: {json.dumps(todo.get('spatial_hints', {}))}

Dependencies completed: {', '.join(todo.get('depends_on', [])) or 'None'}

Please complete this task according to your specialty and personality.
USE YOUR TOOLS to create actual elements on the canvas.
Remember to indicate if you need a CONTEXT_REFRESH.
"""

        try:
            # Execute the fairy's task
            result = fairy.run(task_prompt)

            # Check if context refresh is requested
            needs_refresh = (
                "CONTEXT_REFRESH_NEEDED: true" in result.lower()
                or "context_refresh_needed: true" in result.lower()
            )

            # Update todo with result
            self.shared_todos.update_status(
                todo["id"], "completed", result
            )

            # Add to conversation
            self.conversation.add(
                role=fairy_name,
                content=f"[Task: {todo['title']}]\n{result}",
            )

            self.execution_stats["completed_tasks"] += 1

            return {
                "success": True,
                "todo_id": todo["id"],
                "fairy": fairy_name,
                "result": result,
                "needs_context_refresh": needs_refresh,
            }

        except Exception as e:
            logger.error(
                f"Error executing task for {fairy_name}: {e}"
            )
            self.shared_todos.update_status(todo["id"], "failed")
            self.execution_stats["failed_tasks"] += 1

            return {
                "success": False,
                "todo_id": todo["id"],
                "fairy": fairy_name,
                "error": str(e),
            }

    def _run_coordination_loop(self, task: str) -> Dict[str, Any]:
        """
        Run the main coordination loop.

        Args:
            task: The original task to accomplish

        Returns:
            Dictionary containing the final results
        """
        loop_results = []

        for loop_num in range(self.max_loops):
            self.execution_stats["coordination_loops"] = loop_num + 1
            logger.info(
                f"Starting coordination loop {loop_num + 1}/{self.max_loops}"
            )

            # Get current context for orchestrator
            context = self._get_context_for_fairy(
                "Orchestrator-Fairy"
            )

            # First loop: create initial plan
            if loop_num == 0:
                orchestrator_prompt = f"""
{context}

=== NEW TASK ===
{task}

Please analyze this task and create a comprehensive plan with todos for your fairy team.
Use your tools to interact with the canvas and todo list.
Output your plan as valid JSON with the structure specified in your instructions.
"""
            else:
                # Subsequent loops: coordinate and issue follow-ups
                completed_results = (
                    self.shared_todos.get_completed_results()
                )
                pending = self.shared_todos.get_pending_todos()

                if not pending and not any(
                    t["needs_revision"]
                    for t in self.shared_todos.todos.values()
                ):
                    # All work is done
                    logger.info(
                        "All todos completed, ending coordination loop"
                    )
                    break

                orchestrator_prompt = f"""
{context}

=== COORDINATION CHECK ===
Original Task: {task}

Completed Results:
{json.dumps(completed_results, indent=2)}

Review the completed work and:
1. Check if any outputs need revision or don't work together
2. Issue follow-up todos if coordination is needed
3. If everything is coherent, indicate completion

Use your tools to read the canvas state and make any adjustments.
Output your coordination decisions as valid JSON.
"""

            # Get orchestrator's plan/coordination
            orchestrator_response = self.orchestrator.run(
                orchestrator_prompt
            )

            self.conversation.add(
                role="Orchestrator-Fairy",
                content=orchestrator_response,
            )

            # Parse the plan
            plan = self._parse_orchestrator_plan(
                orchestrator_response
            )

            if plan and "todos" in plan:
                # Add new todos to shared list
                for todo_data in plan["todos"]:
                    self.shared_todos.add_todo(
                        todo_id=todo_data.get(
                            "id", str(uuid.uuid4())
                        ),
                        title=todo_data.get("title", "Untitled"),
                        description=todo_data.get("description", ""),
                        assigned_to=todo_data.get(
                            "assigned_to", "Creative-Fairy"
                        ),
                        priority=todo_data.get("priority", "medium"),
                        depends_on=todo_data.get("depends_on"),
                        spatial_hints=todo_data.get("spatial_hints"),
                        created_by="Orchestrator-Fairy",
                    )
                    self.execution_stats["total_tasks"] += 1

            # Execute pending todos in parallel
            pending_todos = self.shared_todos.get_pending_todos()

            if not pending_todos:
                logger.info("No pending todos to execute")
                continue

            # Group by fairy and execute in parallel
            context_refresh_needed = False

            with ThreadPoolExecutor(
                max_workers=self.max_parallel_fairies
            ) as executor:
                futures = {}
                for todo in pending_todos:
                    future = executor.submit(
                        self._execute_fairy_task,
                        todo["assigned_to"],
                        todo,
                    )
                    futures[future] = todo

                for future in as_completed(futures):
                    result = future.result()
                    loop_results.append(result)

                    if result.get("needs_context_refresh"):
                        context_refresh_needed = True

            # Handle context refresh if needed
            if context_refresh_needed and self.enable_context_refresh:
                self.execution_stats["context_refreshes"] += 1
                logger.info("Context refresh requested by fairy")

        return {
            "loop_results": loop_results,
            "final_todos": self.shared_todos.get_context_snapshot(),
            "canvas_state": self.shared_canvas.get_snapshot(),
        }

    def _synthesize_results(self) -> str:
        """
        Synthesize all fairy outputs into a final result.

        Returns:
            The synthesized final output
        """
        # Use Harmonizer if available, otherwise orchestrator
        synthesizer = self.fairies.get(
            "Harmonizer-Fairy", self.orchestrator
        )

        completed_results = self.shared_todos.get_completed_results()
        canvas_snapshot = self.shared_canvas.get_snapshot()

        synthesis_prompt = f"""
=== SYNTHESIS REQUEST ===

All fairy work has been completed. Please synthesize the following results into a coherent final output.

Completed Work:
{json.dumps(completed_results, indent=2)}

Canvas Elements:
{json.dumps(canvas_snapshot, indent=2, default=str)}

Use your tools to read and adjust the canvas as needed.
Create a unified, polished final output that integrates all the work done by the fairy team.
"""

        synthesis = synthesizer.run(synthesis_prompt)

        self.conversation.add(
            role=synthesizer.agent_name,
            content=f"[Final Synthesis]\n{synthesis}",
        )

        return synthesis

    def run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Run the fairy swarm on a task.

        Args:
            task: The task to accomplish
            img: Optional image for context
            imgs: Optional list of images for context

        Returns:
            The final output (format depends on output_type)
        """
        self.execution_stats["start_time"] = time.time()
        logger.info(f"FairySwarm starting task: {task[:100]}...")

        # Add task to conversation
        self.conversation.add(role="User", content=task)

        # Set canvas description
        self.shared_canvas.metadata["description"] = task

        try:
            # Run coordination loops
            self._run_coordination_loop(task)

            # Synthesize final result
            final_output = self._synthesize_results()

            self.execution_stats["end_time"] = time.time()

            # Format output
            if self.output_type == "dict":
                return {
                    "task": task,
                    "final_output": final_output,
                    "execution_stats": self.execution_stats,
                    "todos": self.shared_todos.get_context_snapshot(),
                    "canvas": self.shared_canvas.get_snapshot(),
                    "conversation": self.conversation.return_history_as_string(),
                }
            else:
                return history_output_formatter(
                    self.conversation, self.output_type
                )

        except Exception as e:
            logger.error(f"FairySwarm execution failed: {e}")
            self.execution_stats["end_time"] = time.time()
            raise

    def run_with_selected_fairies(
        self, task: str, fairy_names: List[str]
    ) -> Union[str, Dict[str, Any]]:
        """
        Run with a subset of fairies (one will be elected as orchestrator).

        This mimics tldraw's behavior when you select multiple fairies
        and prompt the group - one fairy becomes the orchestrator.

        Args:
            task: The task to accomplish
            fairy_names: Names of fairies to use (first becomes orchestrator)

        Returns:
            The final output
        """
        if len(fairy_names) < 2:
            raise ValueError(
                "At least 2 fairies required for group work"
            )

        # Elect first fairy as orchestrator for this task
        elected_orchestrator_name = fairy_names[0]
        worker_names = fairy_names[1:]

        if elected_orchestrator_name in self.fairies:
            # Temporarily give orchestrator capabilities
            original_prompt = self.fairies[
                elected_orchestrator_name
            ].system_prompt

            fairy_roster = "\n".join(
                [
                    f"- {name}: {self.fairies[name].agent_description}"
                    for name in worker_names
                    if name in self.fairies
                ]
            )

            self.fairies[elected_orchestrator_name].system_prompt = (
                ORCHESTRATOR_FAIRY_PROMPT.format(
                    fairy_roster=fairy_roster
                )
            )

            # Temporarily set as orchestrator
            original_orchestrator = self.orchestrator
            self.orchestrator = self.fairies[
                elected_orchestrator_name
            ]

            # Temporarily limit to selected workers
            original_fairies = self.fairies.copy()
            self.fairies = {
                name: fairy
                for name, fairy in self.fairies.items()
                if name in worker_names
            }

            try:
                result = self.run(task)
            finally:
                # Restore original state
                self.fairies[
                    elected_orchestrator_name
                ].system_prompt = original_prompt
                self.orchestrator = original_orchestrator
                self.fairies = original_fairies

            return result
        else:
            raise ValueError(
                f"Fairy '{elected_orchestrator_name}' not found"
            )

    def get_fairy_roster(self) -> List[Dict[str, str]]:
        """
        Get information about all fairies in the swarm.

        Returns:
            List of fairy info dictionaries
        """
        roster = []
        if self.orchestrator:
            roster.append(
                {
                    "name": self.orchestrator.agent_name,
                    "role": "orchestrator",
                    "description": self.orchestrator.agent_description
                    or "",
                }
            )

        for name, fairy in self.fairies.items():
            roster.append(
                {
                    "name": name,
                    "role": "worker",
                    "description": fairy.agent_description or "",
                }
            )

        return roster

    def reset(self) -> None:
        """Reset the swarm state for a new task."""
        self.shared_todos = SharedTodoList()
        self.shared_canvas = SharedCanvasState()
        self.conversation = Conversation(time_enabled=True)

        # Recreate tools with new shared state
        self._create_tools()

        # Update tools on all agents
        if self.orchestrator:
            self.orchestrator.tools = self.tools.copy()
        for fairy in self.fairies.values():
            fairy.tools = self.tools.copy()

        self.execution_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "context_refreshes": 0,
            "coordination_loops": 0,
            "start_time": None,
            "end_time": None,
        }
        logger.info("FairySwarm state reset")


if __name__ == "__main__":
    # Example usage
    swarm = FairySwarm(
        name="Design Fairies",
        description="A team of fairies for collaborative design work",
        model_name="gpt-4o-mini",
        max_loops=2,
        verbose=True,
    )

    # Show the fairy roster
    print("Fairy Roster:")
    for fairy in swarm.get_fairy_roster():
        print(
            f"  - {fairy['name']} ({fairy['role']}): {fairy['description']}"
        )

    # Run a task
    result = swarm.run(
        "Create a wireframe for a landing page with a header containing a logo and navigation, "
        "a hero section with a headline and call-to-action button, "
        "and a footer with links and copyright."
    )

    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2, default=str))
