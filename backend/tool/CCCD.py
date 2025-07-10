import os 
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".." , "..")))
from backend.ragg import RagMini
from retriever import Retriever
from llm.models import chat_llm_with_ragg 

name = "CCCD"

class CCCD(RagMini):
    def __init__(self):
        self.retriever = Retriever(name)
        self.description = "Cung cấp thông tin về CCCD mà bạn cần biết"
        
    def get_document_relevant(self, query):
        docs = self.retriever.retriever.invoke(
            query
        )
        
        return "\n".join([doc.page_content for doc in docs])
    
    def invoke(self , question : str) -> str : 
        context = self.get_document_relevant(question)
        response = chat_llm_with_ragg( 
            task = "search_information",
            params = {"input" : question , "context" : context}
        )
        
        return response

CCCD_RAGG = CCCD()

if __name__ == "__main__":
    question = "CCCD là gì và dùng để làm gì?"
    result = CCCD_RAGG.invoke(question)
    print("Câu hỏi:", question)
    print("Trợ lý trả lời:", result)
