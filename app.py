# # api_server.py

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from utils.tool_functions import ToolSelector
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# import os

# # Load env
# load_dotenv()

# # Init FastAPI
# app = FastAPI()

# # Init LLM
# llm = ChatGroq(
#     groq_api_key=os.getenv("GROQ_API_KEY"),
#     model_name="Gemma2-9b-It"
# )

# # Init tools
# from tool import CCCD
# tools = {
#     "CCCD": CCCD.CCCD_RAGG
# }

# # Init ToolSelector
# selector = ToolSelector(llm=llm, tools=tools)

# # Request model
# class QuestionRequest(BaseModel):
#     question: str

# # Endpoint: hỏi đáp
# @app.post("/ask")
# def ask_question(req: QuestionRequest):
#     try:
#         answer = selector.call(req.question)
#         return {"answer": answer}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
    
# # Endpoint: reset chủ đề
# @app.post("/reset")
# def reset_topic():
#     selector.current_function = None
#     return {"message": "Đã đặt lại chủ đề."}
