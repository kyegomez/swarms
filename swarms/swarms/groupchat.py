import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from swarms.workers.worker import Worker


@dataclass
class GroupChat:
    """A group chat with multiple participants with a list of workers and a max number of rounds"""

    workers: List[Worker]
    messages: List[Dict]
    max_rounds: int = 10
    admin_name: str = "Admin"  # admin worker

    @property
    def worker_names(self) -> List[str]:
        """returns the names of the workers in the group chat"""
        return [worker.ai_name for worker in self.workers]

    def reset(self):
        self.messages.clear()

    def worker_by_name(self, name: str) -> Worker:
        """Find the next speaker baed on the message"""
        return self.workers[self.worker_names.index(name)]

    def next_worker(self, worker: Worker) -> Worker:
        """Returns the next worker in the list"""
        return self.workers[
            (self.workers_names.index(worker.ai_name) + 1) % len(self.workers)
        ]

    def select_speaker_msg(self):
        """Return the message to select the next speaker"""

        return f"""
        You are in a role play game the following rules are available:
        {self.__participant_roles()}.

        Read the following conversation then select the next role from {self.worker_names}
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
            self.messages
            + [
                {
                    "role": "system",
                    "context": f"Read the above conversation. Then select the next role from {self.worker_names} to play. Only return the role.",
                }
            ]
        )
        if not final:
            return self.next_worker(last_speaker)
        try:
            return self.worker_by_name(name)
        except ValueError:
            return self.next_worker(last_speaker)

    def _participant_roles(self):
        return "\n".join(
            [f"{worker.ai_name}: {worker.system_message}" for worker in self.workers]
        )


class GroupChatManager(Worker):
    def __init__(
        self,
        groupchat: GroupChat,
        ai_name: Optional[str] = "chat_manager",
        max_consecutive_auto_reply: Optional[int] = sys.maxsize,
        human_input_mode: Optional[str] = "NEVER",
        system_message: Optional[str] = "Group chat manager",
        **kwargs,
    ):
        super().__init__(
            ai_name=ai_name,
            # max_consecutive_auto_reply=max_consecutive_auto_reply,
            # human_input_mode=human_input_mode,
            # system_message=system_message,
            **kwargs,
        )
        self.register_reply(
            Worker, GroupChatManager.run, config=groupchat, reset_config=GroupChat.reset
        )

    def run(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Worker] = None,
        config: Optional[GroupChat] = None,
    ) -> Union[str, Dict, None]:
        # run
        if messages is None:
            messages = []

        message = messages[-1]
        speaker = sender
        groupchat = config

        for i in range(groupchat.max_rounds):
            if message["role"] != "function":
                message["name"] = speaker.ai_name

            groupchat.messages.append(message)

            # broadcast the message to all workers except the speaker
            for worker in groupchat.workers:
                if worker != speaker:
                    self.send(
                        message,
                        worker,
                        request_reply=False,
                        silent=True,
                    )
            if i == groupchat.max_rounds - 1:
                break

            try:
                # select next speaker
                speaker = groupchat.select_speaker(speaker, self)
                # let the speaker speak
                reply = speaker.generate_reply(sender=self)

            except KeyboardInterrupt:
                # let the admin speak if interrupted
                if groupchat.admin_name in groupchat.worker_names:
                    # admin worker is a particpant
                    speaker = groupchat.worker_by_name(groupchat.admin_name)
                    reply = speaker.generate_reply(sender=self)
                else:
                    # admin worker is not found in particpants
                    raise
            if reply is None:
                break

            # speaker sends message without requesting a reply
            speaker.send(reply, self, request_reply=False)
            message = self.last_message(speaker)
            message = self.last_messge(speaker)
        return True, None
