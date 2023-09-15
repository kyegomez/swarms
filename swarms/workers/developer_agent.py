import enum
import os
from pathlib import Path
import sys
import time
import shutil
import argparse
import asyncio
import re
from typing import List, Optional, Callable, Any

import openai
from openai_function_call import openai_function
from tenacity import retry, stop_after_attempt, wait_random_exponential
import logging

from smol_dev.prompts import plan, specify_file_paths, generate_code_sync
from smol_dev.utils import generate_folder, write_file

from agent_protocol import Agent, Step, Task




class DeveloperAgent:
    class StepTypes(str, enum.Enum):
        PLAN = "plan"
        SPECIFY_FILE_PATHS = "specify_file_paths"
        GENERATE_CODE = "generate_code"

    async def _generate_shared_deps(step: Step) -> Step:
        task = await Agent.db.get_task(step.task_id)
        shared_deps = plan(task.input)
        await Agent.db.create_step(
            step.task_id,
            DeveloperAgent.StepTypes.SPECIFY_FILE_PATHS,
            additional_properties={
                "shared_deps": shared_deps,
            },
        )
        step.output = shared_deps
        return step

    async def _generate_file_paths(task: Task, step: Step) -> Step:
        shared_deps = step.additional_properties["shared_deps"]
        file_paths = specify_file_paths(task.input, shared_deps)
        for file_path in file_paths[:-1]:
            await Agent.db.create_step(
                task.task_id,
                f"Generate code for {file_path}",
                additional_properties={
                    "shared_deps": shared_deps,
                    "file_path": file_paths[-1],
                },
            )

        await Agent.db.create_step(
            task.task_id,
            f"Generate code for {file_paths[-1]}",
            is_last=True,
            additional_properties={
                "shared_deps": shared_deps,
                "file_path": file_paths[-1],
            },
        )

        step.output = f"File paths are: {str(file_paths)}"
        return step

    async def _generate_code(task: Task, step: Step) -> Step:
        shared_deps = step.additional_properties["shared_deps"]
        file_path = step.additional_properties["file_path"]

        code = await generate_code(task.input, shared_deps, file_path)
        step.output = code

        write_file(os.path.join(Agent.get_workspace(task.task_id), file_path), code)
        path = Path("./" + file_path)
        await Agent.db.create_artifact(
            task_id=task.task_id,
            step_id=step.step_id,
            relative_path=str(path.parent),
            file_name=path.name,
        )

        return step

    async def task_handler(task: Task) -> None:
        if not task.input:
            raise Exception("No task prompt")
        await Agent.db.create_step(task.task_id, DeveloperAgent.StepTypes.PLAN)

    async def step_handler(step: Step):
        task = await Agent.db.get_task(step.task_id)
        if step.name == DeveloperAgent.StepTypes.PLAN:
            return await DeveloperAgent._generate_shared_deps(step)
        elif step.name == DeveloperAgent.StepTypes.SPECIFY_FILE_PATHS:
            return await DeveloperAgent._generate_file_paths(task, step)
        else:
            return await DeveloperAgent._generate_code(task, step)

    @classmethod
    def setup_agent(cls, task_handler, step_handler):
        # Setup agent here
        pass

    @staticmethod
    def generate_folder(folder_path: str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)

    @staticmethod
    def write_file(file_path: str, content: str):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, "w") as f:
            f.write(content)

    @staticmethod
    def main(prompt, generate_folder_path="generated", debug=False, model: str = 'gpt-4-0613'):
        DeveloperAgent.generate_folder(generate_folder_path)

        if debug:
            print("--------shared_deps---------")
        with open(f"{generate_folder_path}/shared_deps.md", "wb") as f:
            start_time = time.time()
            def stream_handler(chunk):
                f.write(chunk)
                if debug:
                    end_time = time.time()
                    sys.stdout.write("\r \033[93mChars streamed\033[0m: {}. \033[93mChars per second\033[0m: {:.2f}".format(stream_handler.count, stream_handler.count / (end_time - start_time)))
                    sys.stdout.flush()
                    stream_handler.count += len(chunk)
            stream_handler.count = 0
            stream_handler.onComplete = lambda x: sys.stdout.write("\033[0m\n")

            shared_deps = plan(prompt, stream_handler, model=model)
        if debug:
            print(shared_deps)
        DeveloperAgent.write_file(f"{generate_folder_path}/shared_deps.md", shared_deps)
        if debug:
            print("--------shared_deps---------")

        if debug:
            print("--------specify_filePaths---------")
        file_paths = specify_file_paths(prompt, shared_deps, model=model)
        if debug:
            print(file_paths)
        if debug:
            print("--------file_paths---------")

        for file_path in file_paths:
            file_path = f"{generate_folder_path}/{file_path}"
            if debug:
                print(f"--------generate_code: {file_path} ---------")

            start_time = time.time()
            def stream_handler(chunk):
                if debug:
                    end_time = time.time()
                    sys.stdout.write("\r \033[93mChars streamed\033[0m: {}. \033[93mChars per second\033[0m: {:.2f}".format(stream_handler.count, stream_handler.count / (end_time - start_time)))
                    sys.stdout.flush()
                    stream_handler.count += len(chunk)
            stream_handler.count = 0
            stream_handler.onComplete = lambda x: sys.stdout.write("\033[0m\n")
            code = generate_code_sync(prompt, shared_deps, file_path, stream_handler, model=model)
            if debug:
                print(code)
            if debug:
                print(f"--------generate_code: {file_path} ---------")
            DeveloperAgent.write_file(file_path, code)
            
        print("--------Smol Dev done!---------")

if __name__ == "__main__":
    prompt = """
  a simple JavaScript/HTML/CSS/Canvas app that is a one-player game of PONG. 
  The left paddle is controlled by the player, following where the mouse goes.
  The right paddle is controlled by a simple AI algorithm, which slowly moves the paddle toward the ball at every frame, with some probability of error.
  Make the canvas a 400 x 400 black square and center it in the app.
  Make the paddles 100px long, yellow, and the ball small and red.
  Make sure to render the paddles and name them so they can be controlled in JavaScript.
  Implement the collision detection and scoring as well.
  Every time the ball bounces off a paddle, the ball should move faster.
  It is meant to run in the Chrome browser, so don't use anything that is not supported by Chrome, and don't use the import and export keywords.
  """

    if len(sys.argv) == 2:
        prompt = sys.argv[1]
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--prompt", type=str, required=True, help="Prompt for the app to be created.")
        parser.add_argument("--generate_folder_path", type=str, default="generated", help="Path of the folder for generated code.")
        parser.add_argument("--debug", type=bool, default=False, help="Enable or disable debug mode.")
        args = parser.parse_args()
        if args.prompt:
            prompt = args.prompt

    print(prompt)

    DeveloperAgent.main(prompt=prompt, generate_folder_path=args.generate_folder_path, debug=args.debug)
