from fastapi import APIRouter , HTTPException
from backend.api.schemas import QuestionRequest 
from backend.api import services
router = APIRouter()

@router.post("/ask")
def ask_question(req : QuestionRequest):
    try : 
        answer = services.ask_question(req.question)
        return {"answer": answer}
    except Exception as e : 
        raise HTTPException(status_code=400 , detail= str(e))

@router.post("/reset")
def reset_topic():
    services.reset_topic()
    return {"message": "Đã đặt lại chủ đề."}