import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from swarms.workers.worker import Worker


@dataclass
class GroupChat:
    """A group chat with multiple participants with a list of agents and a max number of rounds"""

    agents: List[Worker]
    messages: List[Dict]
    max_rounds: int = 10
    admin_name: str = "Admin" #admin agent

    @property
    def agent_names(self) -> List[str]:
        """returns the names of the agents in the group chat"""
        return [agent.ai_name for agent in self.agents]
    
    def reset(self):
        self.messages.clear()
    
    def agent_by_name(self, name: str) -> Worker:
        """Find the next speaker baed on the message"""
        return self.agents[self.agent_names.index(name)]
    
    def next_agent(self, agent: Worker) -> Worker:
        """Returns the next agent in the list"""
        return self.agents[
            (self.agents_names.index(agent.ai_name) + 1) % len(self.agents)
        ]
    
    def select_speaker_msg(self):
        """Return the message to select the next speaker"""

        return f"""
        You are in a role play game the following rules are available:
        {self.__participant_roles()}.

        Read the following conversation then select the next role from {self.agent_names}
        to play and only return the role
        """
    
    def select_speaker(
        self,
        last_speaker: Worker,
        selector: Worker,
    ):
        """Selects the next speaker"""
        selector.update_system_message(self.select_speaker_msg())

        final, name = selector.run(
            self.messages + [
                {
                    "role": "system",
                    "context": f"Read the above conversation. Then select the next role from {self.agent_names} to play. Only return the role.",
                }
            ]
        )
        if not final:
            return self.next_agent(last_speaker)
        try:
            return self.agent_by_name(name)
        except ValueError:
            return self.next_agent(last_speaker)
        
    def _participant_roles(self):
        return "\n".join(
            [f"{agent.ai_name}: {agent.system_message}" for agent in self.agents] 
        )
    

class GroupChatManager(Worker):
    def __init__(
        self,
        groupchat: GroupChat,
        name: Optional[str] = "chat_manager",
        #unlimited auto reply
        max_consecutive_auto_reply: Optional[int] = sys.maxsize,
        human_input_mode: Optional[str] = "NEVER",
        system_message: Optional[str] = "Group chat manager",
        **kwargs
    ):
        super().__init__(
            name=name,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            system_message=system_message,
            **kwargs
        )
        self.register_reply(
            Worker,
            GroupChatManager.run_chat,
            config=groupchat,
            reset_config=GroupChat.reset
        )

    def run(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Worker] = None,
        config: Optional[GroupChat] = None,
    ) -> Union[str, Dict, None]:
        #run
        if messages is None:
            messages = []
        
        message = messages[-1]
        speaker = sender
        groupchat = config

        for i in range(groupchat.max_rounds):
            if message["role"] != "function":
                message["name"]= speaker.ai_name
            
            groupchat.messages.append(message)

            #broadcast the message to all agents except the speaker
            for agent in groupchat.agents:
                if agent != speaker:
                    self.send(
                        message,
                        agent,
                        request_reply=False,
                        silent=True,
                    )
            if i == groupchat.max_rounds - 1:
                break

            try:
                #select next speaker
                speaker = groupchat.select_speaker(speaker, self)
                #let the speaker speak
                reply = speaker.generate_reply(sender=self)
            
            except KeyboardInterrupt:
                #let the admin speak if interrupted
                if groupchat.admin_name in groupchat.agent_names:
                    #admin agent is a particpant
                    speaker = groupchat.agent_by_name(groupchat.admin_name)
                    reply = speaker.generate_reply(sender=self)
                else:
                    #admin agent is not found in particpants
                    raise
            if reply is None:
                break

            #speaker sends message without requesting a reply
            speaker.send(
                reply,
                self,
                request_reply=False
            )
            message = self.last_message(speaker)
            message = self.last_messge(speaker)
        return True, None