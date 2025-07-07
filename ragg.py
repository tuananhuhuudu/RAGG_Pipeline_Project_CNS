from abc import ABC , abstractmethod
from pydantic import BaseModel

class RagMini(ABC , BaseModel):
    retriever : str 
    description : str 
    messages : str = "Bạn là 1 con chatbot . Nhiệm vụ của bạn là tư vấn , hướng dẫn và trả lời câu hỏi người dùng dựa trên thông tin mà bạn đã được cung cấp"
    
    @abstractmethod
    
    def get_document_relevant(self , query : str) -> str:
        raise NotImplementedError("Bạn phải override hàm get_document_relevant()")
    
    def invoke(self , messages : str) -> str : 
        raise NotImplementedError("Bạn phải override hàm invoke()")
        
    
    
    
        
    
    
    
        
    
    
    