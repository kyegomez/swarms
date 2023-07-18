from fastapi import FastAPI, HTTPException
from swarms.swarms import swarm
from pydantic import BaseModel

app = FastAPI()

class SwarmInput(BaseModel):
    api_key: set
    objective: str


@app.post("/chat")
async def run_swarms(swarm_input: SwarmInput):
    try:
        results = swarm(swarm_input.api_key, swarm_input.objective)
        if not results:
            raise HTTPException(status_code=500, detaile="Failed to run swarms")
        return {"results": results}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
