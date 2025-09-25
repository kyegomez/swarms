"""
Hiring Swarm: Multi-Agent Automated Hiring Workflow

This module demonstrates a multi-agent system for automating the hiring process using the Swarms framework.
It defines specialized agents for each stage of recruitment, such as talent acquisition, candidate screening,
interviewing, and evaluation. The agents collaborate to streamline and optimize the end-to-end hiring workflow,
from identifying staffing needs to final candidate selection.

"""

from typing import List

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.ma_utils import set_random_models_for_agents
from swarms.utils.history_output_formatter import history_output_formatter

# System prompts for each agent

TALENT_ACQUISITION_PROMPT = """
You are the Talent Acquisition Agent.

ROLE:
Identify staffing needs and define job positions. Develop job descriptions and specifications. Utilize various channels like job boards, social media, and recruitment agencies to source potential candidates. Network at industry events and career fairs to attract talent.

RESPONSIBILITIES:
- Identify current and future staffing needs in collaboration with relevant departments.
- Define and document job positions, including required qualifications and responsibilities.
- Develop clear and compelling job descriptions and specifications.
- Source candidates using:
  * Professional job boards
  * Social media platforms
  * Recruitment agencies
  * Industry networking events and career fairs
- Maintain and update a talent pipeline for ongoing and future needs.

OUTPUT FORMAT:
Provide a report including:
1. Identified staffing requirements and job definitions
2. Developed job descriptions/specifications
3. Sourcing channels and strategies used
4. Summary of outreach/networking activities
5. Recommendations for next steps in the hiring process
"""

CANDIDATE_SCREENING_PROMPT = """
You are the Candidate Screening Agent.

ROLE:
Review resumes and application materials to assess candidate suitability. Utilize AI-based tools for initial screening to identify top candidates. Conduct preliminary interviews (telephonic or video) to gauge candidate interest and qualifications. Coordinate with the Talent Acquisition Agent to shortlist candidates for further evaluation.

RESPONSIBILITIES:
- Review and evaluate resumes and application materials for required qualifications and experience.
- Use AI-based screening tools to identify and rank top candidates.
- Conduct preliminary interviews (phone or video) to assess interest, communication, and basic qualifications.
- Document candidate strengths, concerns, and fit for the role.
- Coordinate with the Talent Acquisition Agent to finalize the shortlist for further interviews.

OUTPUT FORMAT:
Provide a structured candidate screening report:
1. List and ranking of screened candidates
2. Summary of strengths and concerns for each candidate
3. Notes from preliminary interviews
4. Shortlist of recommended candidates for next stage
5. Suggestions for further evaluation if needed
"""

INTERVIEW_COORDINATION_PROMPT = """
You are the Interview Coordination Agent.

ROLE:
Schedule and coordinate interviews between candidates and hiring managers. Manage interview logistics, including virtual platform setup or physical meeting arrangements. Collect feedback from interviewers and candidates to improve the interview process. Facilitate any necessary follow-up interviews or assessments.

RESPONSIBILITIES:
- Schedule interviews between shortlisted candidates and relevant interviewers.
- Coordinate logistics: send calendar invites, set up virtual meeting links, or arrange physical meeting spaces.
- Communicate interview details and instructions to all participants.
- Collect and organize feedback from interviewers and candidates after each interview.
- Arrange follow-up interviews or assessments as needed.

OUTPUT FORMAT:
Provide an interview coordination summary:
1. Interview schedule and logistics details
2. Communication logs with candidates and interviewers
3. Summary of feedback collected
4. Notes on any issues or improvements for the process
5. Recommendations for next steps
"""

ONBOARDING_TRAINING_PROMPT = """
You are the Onboarding and Training Agent.

ROLE:
Prepare and disseminate onboarding materials and schedules. Coordinate with IT, Admin, and other departments for workspace setup and access credentials. Organize training sessions and workshops for new hires. Monitor the onboarding process and gather feedback for improvement.

RESPONSIBILITIES:
- Prepare onboarding materials and a detailed onboarding schedule for new hires.
- Coordinate with IT, Admin, and other departments to ensure workspace, equipment, and access credentials are ready.
- Organize and schedule training sessions, workshops, and orientation meetings.
- Monitor the onboarding process, check in with new hires, and gather feedback.
- Identify and address any onboarding issues or gaps.

OUTPUT FORMAT:
Provide an onboarding and training report:
1. Onboarding schedule and checklist
2. List of prepared materials and resources
3. Training session/workshop plan
4. Summary of feedback from new hires
5. Recommendations for improving onboarding
"""

EMPLOYEE_ENGAGEMENT_PROMPT = """
You are the Employee Engagement Agent.

ROLE:
Develop and implement strategies to enhance employee engagement and satisfaction. Organize team-building activities and company events. Administer surveys and feedback tools to gauge employee morale. Collaborate with HR to address any concerns or issues impacting employee wellbeing.

RESPONSIBILITIES:
- Design and implement employee engagement initiatives and programs.
- Organize team-building activities, company events, and wellness programs.
- Develop and administer surveys or feedback tools to measure employee morale and satisfaction.
- Analyze feedback and identify trends or areas for improvement.
- Work with HR to address concerns or issues affecting employee wellbeing.

OUTPUT FORMAT:
Provide an employee engagement report:
1. Summary of engagement initiatives and activities
2. Survey/feedback results and analysis
3. Identified issues or concerns
4. Recommendations for improving engagement and satisfaction
5. Plan for ongoing engagement efforts
"""

class HiringSwarm:
    def __init__(
        self,
        name: str = "Hiring Swarm",
        description: str = "A swarm of agents that can handle comprehensive hiring processes",
        max_loops: int = 1,
        user_name: str = "HR Manager",
        job_role: str = "Software Engineer",
        output_type: str = "list",
    ):
        self.max_loops = max_loops
        self.name = name
        self.description = description
        self.user_name = user_name
        self.job_role = job_role
        self.output_type = output_type

        self.agents = self._initialize_agents()
        self.agents = set_random_models_for_agents(self.agents)
        self.conversation = Conversation()
        self.handle_initial_processing()

    def handle_initial_processing(self):
        self.conversation.add(
            role=self.user_name,
            content=f"Company: {self.name}\n"
                    f"Description: {self.description}\n"
                    f"Job Role: {self.job_role}"
        )

    def _initialize_agents(self) -> List[Agent]:
        return [
            Agent(
                agent_name="Elena-Talent-Acquisition",
                agent_description="Identifies staffing needs, defines job positions, and sources candidates through multiple channels.",
                system_prompt=TALENT_ACQUISITION_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            Agent(
                agent_name="Marcus-Candidate-Screening",
                agent_description="Screens resumes, conducts initial interviews, and shortlists candidates using AI tools.",
                system_prompt=CANDIDATE_SCREENING_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            Agent(
                agent_name="Olivia-Interview-Coordinator",
                agent_description="Schedules and manages interviews, collects feedback, and coordinates logistics.",
                system_prompt=INTERVIEW_COORDINATION_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            Agent(
                agent_name="Nathan-Onboarding-Specialist",
                agent_description="Prepares onboarding materials, coordinates setup, and organizes training for new hires.",
                system_prompt=ONBOARDING_TRAINING_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            Agent(
                agent_name="Sophia-Employee-Engagement",
                agent_description="Develops engagement strategies, organizes activities, and gathers employee feedback.",
                system_prompt=EMPLOYEE_ENGAGEMENT_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
        ]

    def find_agent_by_name(self, name: str) -> Agent:
        """Find an agent by their name."""
        for agent in self.agents:
            if agent.agent_name == name:
                return agent

    def initial_talent_acquisition(self):
        elena_agent = self.find_agent_by_name("Talent-Acquisition")
        elena_output = elena_agent.run(
            f"History: {self.conversation.get_str()}\n"
            f"Identify staffing needs, define the {self.job_role} position, develop job descriptions, and outline sourcing strategies."
        )
        self.conversation.add(
            role="Talent-Acquisition", content=elena_output
        )

    def candidate_screening(self):
        marcus_agent = self.find_agent_by_name("Candidate-Screening")
        marcus_output = marcus_agent.run(
            f"History: {self.conversation.get_str()}\n"
            f"Screen resumes and applications for the {self.job_role} position, conduct preliminary interviews, and provide a shortlist of candidates."
        )
        self.conversation.add(
            role="Candidate-Screening", content=marcus_output
        )

    def interview_coordination(self):
        olivia_agent = self.find_agent_by_name("Interview-Coordinator")
        olivia_output = olivia_agent.run(
            f"History: {self.conversation.get_str()}\n"
            f"Schedule and coordinate interviews for shortlisted {self.job_role} candidates, manage logistics, and collect feedback."
        )
        self.conversation.add(
            role="Interview-Coordinator", content=olivia_output
        )

    def onboarding_preparation(self):
        nathan_agent = self.find_agent_by_name("Onboarding-Specialist")
        nathan_output = nathan_agent.run(
            f"History: {self.conversation.get_str()}\n"
            f"Prepare onboarding materials and schedule, coordinate setup, and organize training for the new {self.job_role} hire."
        )
        self.conversation.add(
            role="Onboarding-Specialist", content=nathan_output
        )

    def employee_engagement_strategy(self):
        sophia_agent = self.find_agent_by_name("Employee-Engagement")
        sophia_output = sophia_agent.run(
            f"History: {self.conversation.get_str()}\n"
            f"Develop and implement an employee engagement plan for the new {self.job_role} hire, including activities and feedback mechanisms."
        )
        self.conversation.add(
            role="Employee-Engagement", content=sophia_output
        )

    def run(self, task: str):
        """
        Process the hiring workflow through the swarm, coordinating tasks among agents.
        """
        self.conversation.add(role=self.user_name, content=task)

        # Execute workflow stages
        self.initial_talent_acquisition()
        self.candidate_screening()
        self.interview_coordination()
        self.onboarding_preparation()
        self.employee_engagement_strategy()

        return history_output_formatter(
            self.conversation, type=self.output_type
        )

def main():
    # Initialize the swarm
    hiring_swarm = HiringSwarm(
        max_loops=1,
        name="TechCorp Hiring Solutions",
        description="Comprehensive AI-driven hiring workflow",
        user_name="HR Director",
        job_role="Software Engineer",
        output_type="json",
    )

    # Sample hiring task
    sample_task = """
    We are looking to hire a Software Engineer for our AI research team.
    Key requirements:
    - Advanced degree in Computer Science
    - 3+ years of experience in machine learning
    - Strong Python and PyTorch skills
    - Experience with large language model development
    """

    # Run the swarm
    hiring_swarm.run(task=sample_task)

if __name__ == "__main__":
    main()