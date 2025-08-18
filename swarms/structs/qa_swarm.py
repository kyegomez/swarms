from swarms.structs.agent import Agent
from typing import List
from swarms.structs.conversation import Conversation
import uuid
import random
from loguru import logger
from typing import Optional
from swarms.security import SwarmShieldIntegration, ShieldConfig


class QASwarm:
    """
    A Question and Answer swarm system where random agents ask questions to speaker agents.

    This system allows for dynamic Q&A sessions where:
    - Multiple agents can act as questioners
    - One or multiple agents can act as speakers/responders
    - Questions are asked randomly by different agents
    - The conversation is tracked and managed
    - Agents are showcased to each other with detailed information
    """

    def __init__(
        self,
        name: str = "QandA",
        description: str = "Question and Answer Swarm System",
        agents: List[Agent] = None,
        speaker_agents: List[Agent] = None,
        id: str = str(uuid.uuid4()),
        max_loops: int = 5,
        show_dashboard: bool = True,
        speaker_agent: Agent = None,
        showcase_agents: bool = True,
        shield_config: Optional[ShieldConfig] = None,
        enable_security: bool = True,
        security_level: str = "standard",
        **kwargs,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.max_loops = max_loops
        self.show_dashboard = show_dashboard
        self.agents = agents or []
        self.speaker_agents = speaker_agents or []
        self.kwargs = kwargs
        self.speaker_agent = speaker_agent
        self.showcase_agents = showcase_agents

        # Initialize SwarmShield integration
        self._initialize_swarm_shield(shield_config, enable_security, security_level)

        self.conversation = Conversation()

        # Validate setup
        self._validate_setup()

    def _initialize_swarm_shield(
        self, 
        shield_config: Optional[ShieldConfig] = None,
        enable_security: bool = True,
        security_level: str = "standard"
    ) -> None:
        """Initialize SwarmShield integration for security features."""
        self.enable_security = enable_security
        self.security_level = security_level
        
        if enable_security:
            if shield_config is None:
                shield_config = ShieldConfig.get_security_level(security_level)
            
            self.swarm_shield = SwarmShieldIntegration(shield_config)
        else:
            self.swarm_shield = None

    # Security methods
    def validate_task_with_shield(self, task: str) -> str:
        """Validate and sanitize task input using SwarmShield."""
        if self.swarm_shield:
            return self.swarm_shield.validate_and_protect_input(task)
        return task

    def validate_agent_config_with_shield(self, agent_config: dict) -> dict:
        """Validate agent configuration using SwarmShield."""
        if self.swarm_shield:
            return self.swarm_shield.validate_and_protect_input(str(agent_config))
        return agent_config

    def process_agent_communication_with_shield(self, message: str, agent_name: str) -> str:
        """Process agent communication through SwarmShield security."""
        if self.swarm_shield:
            return self.swarm_shield.process_agent_communication(message, agent_name)
        return message

    def check_rate_limit_with_shield(self, agent_name: str) -> bool:
        """Check rate limits for an agent using SwarmShield."""
        if self.swarm_shield:
            return self.swarm_shield.check_rate_limit(agent_name)
        return True

    def add_secure_message(self, message: str, agent_name: str) -> None:
        """Add a message to secure conversation history."""
        if self.swarm_shield:
            self.swarm_shield.add_secure_message(message, agent_name)

    def get_secure_messages(self) -> List[dict]:
        """Get secure conversation messages."""
        if self.swarm_shield:
            return self.swarm_shield.get_secure_messages()
        return []

    def get_security_stats(self) -> dict:
        """Get security statistics and metrics."""
        if self.swarm_shield:
            return self.swarm_shield.get_security_stats()
        return {"security_enabled": False}

    def update_shield_config(self, new_config: ShieldConfig) -> None:
        """Update SwarmShield configuration."""
        if self.swarm_shield:
            self.swarm_shield.update_config(new_config)

    def enable_security(self) -> None:
        """Enable SwarmShield security features."""
        if not self.swarm_shield:
            self._initialize_swarm_shield(enable_security=True, security_level=self.security_level)

    def disable_security(self) -> None:
        """Disable SwarmShield security features."""
        self.swarm_shield = None
        self.enable_security = False

    def cleanup_security(self) -> None:
        """Clean up SwarmShield resources."""
        if self.swarm_shield:
            self.swarm_shield.cleanup()

    def _validate_setup(self):
        """Validate that the Q&A system is properly configured."""
        if not self.agents:
            logger.warning(
                "No questioner agents provided. Add agents using add_agent() method."
            )

        if not self.speaker_agents and not self.speaker_agent:
            logger.warning(
                "No speaker agents provided. Add speaker agents using add_speaker_agent() method."
            )

        if (
            not self.agents
            and not self.speaker_agents
            and not self.speaker_agent
        ):
            raise ValueError(
                "At least one agent (questioner or speaker) must be provided."
            )

    def add_agent(self, agent: Agent):
        """Add a questioner agent to the swarm."""
        self.agents.append(agent)
        logger.info(f"Added questioner agent: {agent.agent_name}")

    def add_speaker_agent(self, agent: Agent):
        """Add a speaker agent to the swarm."""
        if self.speaker_agents is None:
            self.speaker_agents = []
        self.speaker_agents.append(agent)
        logger.info(f"Added speaker agent: {agent.agent_name}")

    def get_agent_info(self, agent: Agent) -> dict:
        """Extract key information about an agent for showcasing."""
        info = {
            "name": getattr(agent, "agent_name", "Unknown Agent"),
            "description": getattr(
                agent, "agent_description", "No description available"
            ),
            "role": getattr(agent, "role", "worker"),
        }

        # Get system prompt preview (first 50 characters)
        system_prompt = getattr(agent, "system_prompt", "")
        if system_prompt:
            info["system_prompt_preview"] = (
                system_prompt[:50] + "..."
                if len(system_prompt) > 50
                else system_prompt
            )
        else:
            info["system_prompt_preview"] = (
                "No system prompt available"
            )

        return info

    def showcase_speaker_to_questioner(
        self, questioner: Agent, speaker: Agent
    ) -> str:
        """Create a showcase prompt introducing the speaker agent to the questioner."""
        speaker_info = self.get_agent_info(speaker)

        showcase_prompt = f"""
        You are about to ask a question to a specialized agent. Here's what you need to know about them:

        **Speaker Agent Information:**
        - **Name**: {speaker_info['name']}
        - **Role**: {speaker_info['role']}
        - **Description**: {speaker_info['description']}
        - **System Prompt Preview**: {speaker_info['system_prompt_preview']}

        Please craft a thoughtful, relevant question that takes into account this agent's expertise and background. 
        Your question should be specific and demonstrate that you understand their role and capabilities.
        """
        return showcase_prompt

    def showcase_questioner_to_speaker(
        self, speaker: Agent, questioner: Agent
    ) -> str:
        """Create a showcase prompt introducing the questioner agent to the speaker."""
        questioner_info = self.get_agent_info(questioner)

        showcase_prompt = f"""
You are about to answer a question from another agent. Here's what you need to know about them:

**Questioner Agent Information:**
- **Name**: {questioner_info['name']}
- **Role**: {questioner_info['role']}
- **Description**: {questioner_info['description']}
- **System Prompt Preview**: {questioner_info['system_prompt_preview']}

Please provide a comprehensive answer that demonstrates your expertise and addresses their question thoroughly.
Consider their background and role when formulating your response.
"""
        return showcase_prompt

    def random_select_agent(self, agents: List[Agent]) -> Agent:
        """Randomly select an agent from the list."""
        if not agents:
            raise ValueError("No agents available for selection")
        return random.choice(agents)

    def get_current_speaker(self) -> Agent:
        """Get the current speaker agent (either from speaker_agents list or single speaker_agent)."""
        if self.speaker_agent:
            return self.speaker_agent
        elif self.speaker_agents:
            return self.random_select_agent(self.speaker_agents)
        else:
            raise ValueError("No speaker agent available")

    def run(
        self, task: str, img: Optional[str] = None, *args, **kwargs
    ):
        """Run the Q&A session with agent showcasing."""
        self.conversation.add(role="user", content=task)

        # Get current speaker
        current_speaker = self.get_current_speaker()

        # Select a random questioner
        questioner = self.random_select_agent(self.agents)

        # Showcase agents to each other if enabled
        if self.showcase_agents:
            # Showcase speaker to questioner
            speaker_showcase = self.showcase_speaker_to_questioner(
                questioner, current_speaker
            )
            questioner_task = f"{speaker_showcase}\n\nNow ask a question about: {task}"

            # Showcase questioner to speaker
            questioner_showcase = self.showcase_questioner_to_speaker(
                current_speaker, questioner
            )
        else:
            questioner_task = f"Ask a question about {task} to {current_speaker.agent_name}"

        # Generate question
        question = questioner.run(
            task=questioner_task,
            img=img,
            *args,
            **kwargs,
        )

        self.conversation.add(
            role=questioner.agent_name, content=question
        )

        # Prepare answer task with showcasing if enabled
        if self.showcase_agents:
            answer_task = f"{questioner_showcase}\n\nAnswer this question from {questioner.agent_name}: {question}"
        else:
            answer_task = f"Answer the question '{question}' from {questioner.agent_name}"

        # Generate answer
        answer = current_speaker.run(
            task=answer_task,
            img=img,
            *args,
            **kwargs,
        )

        self.conversation.add(
            role=current_speaker.agent_name, content=answer
        )

        return answer

    def run_multi_round(
        self,
        task: str,
        rounds: int = 3,
        img: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Run multiple rounds of Q&A with different questioners."""
        results = []

        for round_num in range(rounds):
            logger.info(
                f"Starting Q&A round {round_num + 1}/{rounds}"
            )

            round_result = self.run(task, img, *args, **kwargs)
            results.append(
                {"round": round_num + 1, "result": round_result}
            )

        return results

    def get_conversation_history(self):
        """Get the conversation history."""
        return self.conversation.get_history()

    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation = Conversation()
        logger.info("Conversation history cleared")
