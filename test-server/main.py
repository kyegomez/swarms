from fastapi import FastAPI
from routers import openai, generic


app = FastAPI()

# This mocks the openai router
app.include_router(openai.router, prefix="/openai")

# This is for showing what you need to mock
# You can check the log in console or in app.log
app.include_router(generic.router)


# This is here for debugging only
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
