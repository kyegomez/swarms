import os
from celery import Celery
from celery.result import AsyncResult

from api.container import agent_manager

# from env import settings

celery_broker = os.environ["CELERY_BROKER_URL"]


celery_app = Celery(__name__)
celery_app.conf.broker_url = celery_broker
celery_app.conf.result_backend = celery_broker
celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    accept_content=["json"],  # Ignore other content
    result_serializer="json",
    enable_utc=True,
)


@celery_app.task(name="task_execute", bind=True)
def task_execute(self, session: str, prompt: str):
    executor = agent_manager.create_executor(session, self)
    response = executor({"input": prompt})
    result = {"output": response["output"]}

    previous = AsyncResult(self.request.id)
    if previous and previous.info:
        result.update(previous.info)

    return result


def get_task_result(task_id):
    return AsyncResult(task_id)


def start_worker():
    celery_app.worker_main(
        [
            "worker",
            "--loglevel=INFO",
        ]
    )
