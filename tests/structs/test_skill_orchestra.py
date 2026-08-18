import pytest
from unittest.mock import MagicMock
from examples.multi_agent.skill_orchestra_examples.skill_orchestra import (
    SkillOrchestra,
    SkillHandbook,
    SkillDefinition,
    AgentProfile,
    AgentSkillProfile,
    TaskSkillInference,
    InferredSkill,
)


def create_test_handbook():
    return SkillHandbook(
        skills=[
            SkillDefinition(
                name="coding",
                description="Writing code",
            )
        ],
        agent_profiles=[
            AgentProfile(
                agent_name="Agent1",
                skill_profiles=[
                    AgentSkillProfile(
                        skill_name="coding",
                        competence=0.9,
                    )
                ],
            ),
            AgentProfile(
                agent_name="Agent2",
                skill_profiles=[
                    AgentSkillProfile(
                        skill_name="coding",
                        competence=0.8,
                    )
                ],
            ),
        ],
    )


def test_single_agent_imgs_and_kwargs_forwarding(monkeypatch):
    agent1 = MagicMock()
    agent1.agent_name = "Agent1"
    agent1.run.return_value = "Agent1 response"

    handbook = create_test_handbook()
    orchestra = SkillOrchestra(
        agents=[agent1],
        skill_handbook=handbook,
        top_k_agents=1,
        learning_enabled=False,
        autosave=False,
    )

    fake_inference = TaskSkillInference(
        task_summary="coding task",
        required_skills=[
            InferredSkill(
                skill_name="coding",
                importance=1.0,
                reasoning="need code",
            )
        ],
    )
    monkeypatch.setattr(
        orchestra, "_infer_task_skills", lambda task: fake_inference
    )

    custom_cb = lambda x: None
    result = orchestra.run(
        "Write python code",
        imgs=["img1.png", "img2.png"],
        custom_param="custom_val",
        streaming_callback=custom_cb,
    )

    assert agent1.run.called
    call_kwargs = agent1.run.call_args.kwargs
    assert call_kwargs["task"] == "Write python code"
    assert call_kwargs["imgs"] == ["img1.png", "img2.png"]
    assert call_kwargs["custom_param"] == "custom_val"
    assert call_kwargs["streaming_callback"] == custom_cb


def test_multi_agent_imgs_and_kwargs_forwarding(monkeypatch):
    agent1 = MagicMock()
    agent1.agent_name = "Agent1"
    agent1.run.return_value = "Agent1 response"

    agent2 = MagicMock()
    agent2.agent_name = "Agent2"
    agent2.run.return_value = "Agent2 response"

    handbook = create_test_handbook()
    orchestra = SkillOrchestra(
        agents=[agent1, agent2],
        skill_handbook=handbook,
        top_k_agents=2,
        learning_enabled=False,
        autosave=False,
    )

    fake_inference = TaskSkillInference(
        task_summary="coding task",
        required_skills=[
            InferredSkill(
                skill_name="coding",
                importance=1.0,
                reasoning="need code",
            )
        ],
    )
    monkeypatch.setattr(
        orchestra, "_infer_task_skills", lambda task: fake_inference
    )

    result = orchestra.run(
        "Parallel task",
        imgs=["imgA.png", "imgB.png"],
        extra_flag=True,
    )

    assert agent1.run.called
    assert agent2.run.called

    for ag in (agent1, agent2):
        call_kwargs = ag.run.call_args.kwargs
        assert call_kwargs["task"] == "Parallel task"
        assert call_kwargs["imgs"] == ["imgA.png", "imgB.png"]
        assert call_kwargs["extra_flag"] is True


def test_existing_task_only_and_img_only(monkeypatch):
    agent1 = MagicMock()
    agent1.agent_name = "Agent1"
    agent1.run.return_value = "Agent1 response"

    handbook = create_test_handbook()
    orchestra = SkillOrchestra(
        agents=[agent1],
        skill_handbook=handbook,
        top_k_agents=1,
        learning_enabled=False,
        autosave=False,
    )

    fake_inference = TaskSkillInference(
        task_summary="task",
        required_skills=[
            InferredSkill(
                skill_name="coding",
                importance=1.0,
                reasoning="need code",
            )
        ],
    )
    monkeypatch.setattr(
        orchestra, "_infer_task_skills", lambda task: fake_inference
    )

    # Task only
    orchestra.run("Simple task")
    assert agent1.run.call_args.kwargs["task"] == "Simple task"
    assert agent1.run.call_args.kwargs["img"] is None
    assert agent1.run.call_args.kwargs["imgs"] is None

    # Img only
    orchestra.run("Task with img", img="single.png")
    assert agent1.run.call_args.kwargs["task"] == "Task with img"
    assert agent1.run.call_args.kwargs["img"] == "single.png"
    assert agent1.run.call_args.kwargs["imgs"] is None
