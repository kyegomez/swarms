import os
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()


class DotEnv(TypedDict):
    OPENAI_API_KEY: str

    EVAL_PORT: int
    SERVER: str

    CELERY_BROKER_URL: str
    USE_GPU: bool  # optional
    PLAYGROUND_DIR: str  # optional
    LOG_LEVEL: str  # optional
    BOT_NAME: str  # optional
    AWS_ACCESS_KEY_ID: str  # optional
    AWS_SECRET_ACCESS_KEY: str  # optional
    AWS_REGION: str  # optional
    AWS_S3_BUCKET: str  # optional
    WINEDB_HOST: str  # optional
    WINEDB_PASSWORD: str  # optional
    BING_SEARCH_URL: str  # optional
    BING_SUBSCRIPTION_KEY: str  # optional
    SERPAPI_API_KEY: str  # optional


EVAL_PORT = int(os.getenv("EVAL_PORT", 8000))
settings: DotEnv = {
    "EVAL_PORT": EVAL_PORT,
    "MODEL_NAME": os.getenv("MODEL_NAME", "gpt-4"),
    "CELERY_BROKER_URL": os.getenv("CELERY_BROKER_URL", "redis://localhost:6379"),
    "SERVER": os.getenv("SERVER", f"http://localhost:{EVAL_PORT}"),
    "USE_GPU": os.getenv("USE_GPU", "False").lower() == "true",
    "PLAYGROUND_DIR": os.getenv("PLAYGROUND_DIR", "playground"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    "BOT_NAME": os.getenv("BOT_NAME", "Orca"),
    "WINEDB_HOST": os.getenv("WINEDB_HOST"),
    "WINEDB_PASSWORD": os.getenv("WINEDB_PASSWORD"),
    "BING_SEARCH_URL": os.getenv("BING_SEARCH_URL"),
    "BING_SUBSCRIPTION_KEY": os.getenv("BING_SUBSCRIPTION_KEY"),
    "SERPAPI_API_KEY": os.getenv("SERPAPI_API_KEY"),
}