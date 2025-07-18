from backend.api.schemas import QuestionRequest 
from backend.orchestrator.tool_selector import ToolSelector 
from backend.tool import CCCD
from backend.config import llm 

tools = {
    "CCCD" : CCCD.CCCD_RAGG , 
}
    
selector = ToolSelector(llm = llm , tools= tools)

def ask_question(question: str) -> str:
    answer = selector.call(question=question)
    return answer


def reset_topic() -> None : 
    selector.current_function = None 
    return selector.current_function 


