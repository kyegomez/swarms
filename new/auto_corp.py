"""
CEO -> Finds department leader 
Department leader -> Finds employees 
Employees -> Do the work

Todo
- Create schemas that enable the ceo to find the department leader or leaders
- CEO then distributes orders to department leaders or just one leader
- Department leader then distributes orders to employees
- Employees can choose to do the work or delegate to another employee or work together
- When the employees are done, they report back to the department leader
- Department leader then reports back to the ceo
- CEO then reports back to the user



Logic
- dynamically setup conversations for each department -- Feed context to each agent in the department
- Feed context to each agent in the department
"""

from typing import Callable, List, Union

from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.ma_utils import list_all_agents
from swarms.utils.str_to_dict import str_to_dict
from swarms.utils.any_to_str import any_to_str


class Department(BaseModel):
    name: str = Field(description="The name of the department")
    description: str = Field(
        description="A description of the department"
    )
    employees: List[Union[Agent, Callable]] = Field(
        description="A list of employees in the department"
    )
    leader_name: str = Field(
        description="The name of the leader of the department"
    )

    class Config:
        arbitrary_types_allowed = True


CEO_SCHEMA = {
    "name": "delegate_task_to_department",
    "description": "CEO function to analyze and delegate tasks to appropriate department leaders",
    "parameters": {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "Reasoning about the task, its requirements, and potential approaches",
            },
            "plan": {
                "type": "string",
                "description": "Structured plan for how to accomplish the task across departments",
            },
            "tasks": {
                "type": "object",
                "properties": {
                    "task_description": {"type": "string"},
                    "selected_departments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of department names that should handle this task",
                    },
                    "selected_leaders": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of department leaders to assign the task to",
                    },
                    "success_criteria": {"type": "string"},
                },
                "required": [
                    "task_description",
                    "selected_departments",
                    "selected_leaders",
                ],
            },
        },
        "required": ["thought", "plan", "tasks"],
    },
}

DEPARTMENT_LEADER_SCHEMA = {
    "name": "manage_department_task",
    "description": "Department leader function to break down and assign tasks to employees",
    "parameters": {
        "type": "object",
        "properties": {
            "task_management": {
                "type": "object",
                "properties": {
                    "original_task": {"type": "string"},
                    "subtasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subtask_id": {"type": "string"},
                                "description": {"type": "string"},
                                "assigned_employees": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "estimated_duration": {
                                    "type": "string"
                                },
                                "dependencies": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                    "progress_tracking": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": [
                                    "not_started",
                                    "in_progress",
                                    "completed",
                                ],
                            },
                            "completion_percentage": {
                                "type": "number"
                            },
                            "blockers": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["original_task", "subtasks"],
            }
        },
        "required": ["task_management"],
    },
}

EMPLOYEE_SCHEMA = {
    "name": "handle_assigned_task",
    "description": "Employee function to process and execute assigned tasks",
    "parameters": {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "Reasoning about the task, its requirements, and potential approaches",
            },
            "plan": {
                "type": "string",
                "description": "Structured plan for how to accomplish the task across departments",
            },
            "task_execution": {
                "type": "object",
                "properties": {
                    "subtask_id": {"type": "string"},
                    "action_taken": {
                        "type": "string",
                        "enum": [
                            "execute",
                            "delegate",
                            "collaborate",
                        ],
                    },
                    "execution_details": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": [
                                    "in_progress",
                                    "completed",
                                    "blocked",
                                ],
                            },
                            "work_log": {"type": "string"},
                            "collaboration_partners": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "delegate_to": {"type": "string"},
                            "results": {"type": "string"},
                            "issues_encountered": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
                "required": [
                    "thought",
                    "plan",
                    "subtask_id",
                    "action_taken",
                    "execution_details",
                ],
            },
        },
        "required": ["task_execution"],
    },
}

# Status report schemas for the feedback loop
EMPLOYEE_REPORT_SCHEMA = {
    "name": "submit_task_report",
    "description": "Employee function to report task completion status to department leader",
    "parameters": {
        "type": "object",
        "properties": {
            "task_report": {
                "type": "object",
                "properties": {
                    "subtask_id": {"type": "string"},
                    "completion_status": {
                        "type": "string",
                        "enum": ["completed", "partial", "blocked"],
                    },
                    "work_summary": {"type": "string"},
                    "time_spent": {"type": "string"},
                    "challenges": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "next_steps": {"type": "string"},
                },
                "required": [
                    "subtask_id",
                    "completion_status",
                    "work_summary",
                ],
            }
        },
        "required": ["task_report"],
    },
}

DEPARTMENT_REPORT_SCHEMA = {
    "name": "submit_department_report",
    "description": "Department leader function to report department progress to CEO",
    "parameters": {
        "type": "object",
        "properties": {
            "department_report": {
                "type": "object",
                "properties": {
                    "department_name": {"type": "string"},
                    "task_summary": {"type": "string"},
                    "overall_status": {
                        "type": "string",
                        "enum": ["on_track", "at_risk", "completed"],
                    },
                    "completion_percentage": {"type": "number"},
                    "key_achievements": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "blockers": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "resource_needs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "next_milestones": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "department_name",
                    "task_summary",
                    "overall_status",
                ],
            }
        },
        "required": ["department_report"],
    },
}

CEO_FINAL_REPORT_SCHEMA = {
    "name": "generate_final_report",
    "description": "CEO function to compile final report for the user",
    "parameters": {
        "type": "object",
        "properties": {
            "final_report": {
                "type": "object",
                "properties": {
                    "task_overview": {"type": "string"},
                    "overall_status": {
                        "type": "string",
                        "enum": ["successful", "partial", "failed"],
                    },
                    "department_summaries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "department": {"type": "string"},
                                "contribution": {"type": "string"},
                                "performance": {"type": "string"},
                            },
                        },
                    },
                    "final_results": {"type": "string"},
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "next_steps": {"type": "string"},
                },
                "required": [
                    "task_overview",
                    "overall_status",
                    "final_results",
                ],
            }
        },
        "required": ["final_report"],
    },
}

# # Example output schemas
# CEO_EXAMPLE_OUTPUT = {
#     "thought": "This task requires coordination between the engineering and design departments to create a new feature. The engineering team will handle the backend implementation while design focuses on the user interface.",
#     "plan": "1. Assign backend development to engineering department\n2. Assign UI/UX design to design department\n3. Set up regular sync meetings between departments\n4. Establish clear success criteria",
#     "tasks": {
#         "task_description": "Develop a new user authentication system with social login integration",
#         "selected_departments": ["engineering", "design"],
#         "selected_leaders": ["engineering_lead", "design_lead"],
#         "success_criteria": "1. Social login working with 3 major providers\n2. UI/UX approved by design team\n3. Security audit passed\n4. Performance metrics met"
#     }
# }

# DEPARTMENT_LEADER_EXAMPLE_OUTPUT = {
#     "task_management": {
#         "original_task": "Develop a new user authentication system with social login integration",
#         "subtasks": [
#             {
#                 "subtask_id": "ENG-001",
#                 "description": "Implement OAuth2 integration for Google",
#                 "assigned_employees": ["dev1", "dev2"],
#                 "estimated_duration": "3 days",
#                 "dependencies": ["DES-001"]
#             },
#             {
#                 "subtask_id": "ENG-002",
#                 "description": "Implement OAuth2 integration for Facebook",
#                 "assigned_employees": ["dev3"],
#                 "estimated_duration": "2 days",
#                 "dependencies": ["DES-001"]
#             }
#         ],
#         "progress_tracking": {
#             "status": "in_progress",
#             "completion_percentage": 0.3,
#             "blockers": ["Waiting for design team to provide UI mockups"]
#         }
#     }
# }

# EMPLOYEE_EXAMPLE_OUTPUT = {
#     "thought": "The Google OAuth2 integration requires careful handling of token management and user data synchronization",
#     "plan": "1. Set up Google OAuth2 credentials\n2. Implement token refresh mechanism\n3. Create user data sync pipeline\n4. Add error handling and logging",
#     "task_execution": {
#         "subtask_id": "ENG-001",
#         "action_taken": "execute",
#         "execution_details": {
#             "status": "in_progress",
#             "work_log": "Completed OAuth2 credential setup and initial token handling implementation",
#             "collaboration_partners": ["dev2"],
#             "delegate_to": None,
#             "results": "Successfully implemented basic OAuth2 flow",
#             "issues_encountered": ["Need to handle token refresh edge cases"]
#         }
#     }
# }

# EMPLOYEE_REPORT_EXAMPLE = {
#     "task_report": {
#         "subtask_id": "ENG-001",
#         "completion_status": "partial",
#         "work_summary": "Completed initial OAuth2 implementation, working on token refresh mechanism",
#         "time_spent": "2 days",
#         "challenges": ["Token refresh edge cases", "Rate limiting considerations"],
#         "next_steps": "Implement token refresh mechanism and add rate limiting protection"
#     }
# }

# DEPARTMENT_REPORT_EXAMPLE = {
#     "department_report": {
#         "department_name": "Engineering",
#         "task_summary": "Making good progress on OAuth2 implementation, but waiting on design team for UI components",
#         "overall_status": "on_track",
#         "completion_percentage": 0.4,
#         "key_achievements": [
#             "Completed Google OAuth2 basic flow",
#             "Set up secure token storage"
#         ],
#         "blockers": ["Waiting for UI mockups from design team"],
#         "resource_needs": ["Additional QA resources for testing"],
#         "next_milestones": [
#             "Complete Facebook OAuth2 integration",
#             "Implement token refresh mechanism"
#         ]
#     }
# }

# CEO_FINAL_REPORT_EXAMPLE = {
#     "final_report": {
#         "task_overview": "Successfully implemented new authentication system with social login capabilities",
#         "overall_status": "successful",
#         "department_summaries": [
#             {
#                 "department": "Engineering",
#                 "contribution": "Implemented secure OAuth2 integrations and token management",
#                 "performance": "Excellent - completed all technical requirements"
#             },
#             {
#                 "department": "Design",
#                 "contribution": "Created intuitive UI/UX for authentication flows",
#                 "performance": "Good - delivered all required designs on time"
#             }
#         ],
#         "final_results": "New authentication system is live and processing 1000+ logins per day",
#         "recommendations": [
#             "Add more social login providers",
#             "Implement biometric authentication",
#             "Add two-factor authentication"
#         ],
#         "next_steps": "Monitor system performance and gather user feedback for improvements"
#     }
# }


class AutoCorp:
    def __init__(
        self,
        name: str = "AutoCorp",
        description: str = "A company that uses agents to automate tasks",
        departments: List[Department] = [],
        ceo: Agent = None,
    ):
        self.name = name
        self.description = description
        self.departments = departments
        self.ceo = ceo
        self.conversation = Conversation()

        # Check if the CEO and departments are set
        self.reliability_check()

        # Add departments to conversation
        self.add_departments_to_conversation()

        # Initialize the CEO agent
        self.initialize_ceo_agent()

        # Initialize the department leaders
        self.setup_department_leaders()

        # Initialize the department employees
        self.department_employees_initialize()

    def initialize_ceo_agent(self):
        self.ceo.tools_list_dictionary = [
            CEO_SCHEMA,
            CEO_FINAL_REPORT_SCHEMA,
        ]

    def setup_department_leaders(self):
        self.department_leader_initialize()
        self.initialize_department_leaders()

    def department_leader_initialize(self):
        """Initialize each department leader with their department's context."""

        for department in self.departments:
            # Create a context dictionary for the department
            department_context = {
                "name": department.name,
                "description": department.description,
                "employees": list_all_agents(
                    department.employees,
                    self.conversation,
                    department.name,
                    False,
                ),
            }

            # Convert the context to a string
            context_str = any_to_str(department_context)

            # TODO: Add the department leader's tools and context
            department.leader.system_prompt += f"""
            You are the leader of the {department.name} department.
            
            Department Context:
            {context_str}

            Your role is to:
            1. Break down tasks into subtasks
            2. Assign subtasks to appropriate employees
            3. Track progress and manage blockers
            4. Report back to the CEO

            Use the provided tools to manage your department effectively.
            """

    def department_employees_initialize(self):
        """Initialize each department leader with their department's context."""

        for department in self.departments:
            # Create a context dictionary for the department
            department_context = {
                "name": department.name,
                "description": department.description,
                "employees": list_all_agents(
                    department.employees,
                    self.conversation,
                    department.name,
                    False,
                ),
                "leader": department.leader_name,
            }

            print(department_context)

            # Convert the context to a string
            context_str = any_to_str(department_context)

            # Set the department leader's tools and context
            department.employees.system_prompt += f"""
            You are an employee of the {department.name} department.
            
            Department Context:
            {context_str}

            Your role is to:
            1. Break down tasks into subtasks
            2. Assign subtasks to appropriate employees
            3. Track progress and manage blockers
            4. Report back to the CEO

            Use the provided tools to manage your department effectively.
            """

    def initialize_department_leaders(self):
        # Use list comprehension for faster initialization
        [
            setattr(
                dept.leader,
                "tools_list_dictionary",
                [DEPARTMENT_LEADER_SCHEMA],
            )
            for dept in self.departments
        ]

    def reliability_check(self):
        if self.ceo is None:
            raise ValueError("CEO is not set")

        if self.departments is None:
            raise ValueError("No departments are set")

        if len(self.departments) == 0:
            raise ValueError("No departments are set")

    def add_departments_to_conversation(self):
        # Batch process departments using list comprehension
        messages = [
            {
                "role": "System",
                "content": f"Team: {dept.name}\nDescription: {dept.description}\nLeader: {dept.leader_name}\nAgents: {list_all_agents(dept.employees, self.conversation, dept.name, False)}",
            }
            for dept in self.departments
        ]
        self.conversation.batch_add(messages)

    # def add_department(self, department: Department):
    #     self.departments.append(department)

    # def add_employee(self, employee: Union[Agent, Callable]):
    #     self.departments[-1].employees.append(employee)

    # def add_ceo(self, ceo: Agent):
    #     self.ceo = ceo

    # def add_employee_to_department(
    #     self, employee: Union[Agent, Callable], department: Department
    # ):
    #     department.employees.append(employee)

    # def add_leader_to_department(
    #     self, leader: Agent, department: Department
    # ):
    #     department.leader = leader

    # def add_department_to_auto_corp(self, department: Department):
    #     self.departments.append(department)

    # def add_ceo_to_auto_corp(self, ceo: Agent):
    #     self.ceo = ceo

    # def add_employee_to_ceo(self, employee: Union[Agent, Callable]):
    #     self.ceo.employees.append(employee)

    def run(self, task: str):
        self.ceo_to_department_leaders(task)

        # Then the department leaders to employees

    def ceo_to_department_leaders(self, task: str):
        orders = self.ceo.run(
            f"History: {self.conversation.get_str()}\n Your Current Task: {task}"
        )

        orders = str_to_dict(orders)

        for department in orders["tasks"]["selected_departments"]:
            department_leader = self.departments[department].leader

            # Get the department leader to break down the task
            outputs = department_leader.run(
                orders["tasks"]["selected_leaders"]
            )

            # Add the department leader's response to the conversation
            self.conversation.add(
                role=f"{department_leader.name} from {department}",
                content=outputs,
            )
