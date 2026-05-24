from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import run_agent

app = FastAPI(title="doc-qa-agent")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    plan: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    return run_agent(req.question)
