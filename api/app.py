from fastapi import FastAPI, HTTPException, Depends
from fastapi_cache.decorator import cache
from fastapi_cache.coder import JsonCoder
from fastapi_cache import FastAPICache
from aioredis import Redis
from pydantic import BaseModel
from swarms.swarms import swarm
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

class SwarmInput(BaseModel):
    api_key: str
    objective: str

app = FastAPI()

@app.on_event("startup")
async def startup():
    redis = Redis(host="localhost", port=6379)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache", coder=JsonCoder())
    await FastAPILimiter.init("redis://localhost:6379")

@app.post("/chat", dependencies=[Depends(RateLimiter(times=2, minutes=1))])
@cache(expire=60)  # Cache results for 1 minute
async def run_swarms(swarm_input: SwarmInput):
    try:
        results = swarm(swarm_input.api_key, swarm_input.objective)
        if not results:
            raise HTTPException(status_code=500, detail="Failed to run swarms")
        return {"results": results}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))