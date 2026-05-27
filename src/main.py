import subprocess
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agent import run_agent

app = FastAPI(title="doc-qa-agent")

_STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    plan: list[dict]


@app.get("/")
def index():
    return FileResponse(str(_STATIC / "index.html"))


@app.get("/open")
def open_file(path: str = Query(...)):
    p = Path(path).resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    subprocess.Popen(["open", str(p)])
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    return run_agent(req.question)
