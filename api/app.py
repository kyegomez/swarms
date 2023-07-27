import logging
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi_cache.decorator import cache
from fastapi_cache.coder import JsonCoder

from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from aioredis import Redis

from pydantic import BaseModel
from swarms.swarms import swarm
from fastapi_limiter import FastAPILimiter

from fastapi_limiter.depends import RateLimiter
from dotenv import load_dotenv

load_dotenv()

class SwarmInput(BaseModel):
    api_key: str
    objective: str

app = FastAPI()

@app.on_event("startup")
async def startup():
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    redis = await Redis.create(redis_host, redis_port)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache", coder=JsonCoder())
    await FastAPILimiter.init(f"redis://{redis_host}:{redis_port}")

@app.post("/chat", dependencies=[Depends(RateLimiter(times=2, minutes=1))])
@cache(expire=60)  # Cache results for 1 minute
async def run(swarm_input: SwarmInput):
    try:
        results = swarm(swarm_input.api_key, swarm_input.objective)
        if not results:
            raise HTTPException(status_code=500, detail="Failed to run swarms")
        return {"results": results}
    except ValueError as ve:
        logging.error("A ValueError occurred", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        logging.error("An error occurred", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
