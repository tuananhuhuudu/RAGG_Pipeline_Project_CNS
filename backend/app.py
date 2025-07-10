# api_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend.orchestrator.tool_selector import ToolSelector
from backend.config import (
    llm
)
import os

# Init FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Hoặc thay * bằng ["http://localhost:3000"] nếu dùng React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init tools
from backend.tool import CCCD
tools = {
    "CCCD": CCCD.CCCD_RAGG
}

# Init ToolSelector
selector = ToolSelector(llm=llm, tools=tools)

# Request model
class QuestionRequest(BaseModel):
    question: str

# Endpoint: hỏi đáp
@app.post("/ask")
def ask_question(req: QuestionRequest):
    try:
        answer = selector.call(req.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
# Endpoint: reset chủ đề
@app.post("/reset")
def reset_topic():
    selector.current_function = None
    return {"message": "Đã đặt lại chủ đề."}
