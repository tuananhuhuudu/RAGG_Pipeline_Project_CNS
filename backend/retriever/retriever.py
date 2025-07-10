import os 

import faiss 
from langchain_community.vectorstores import FAISS 
from langchain_community.docstore.in_memory import InMemoryDocstore 
from langchain_core.vectorstores import VectorStoreRetriever 
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import (
    embeddings , 
    VECTORSTORE_PATH , 
    DATA_PATH
)
from backend.data_loader import data_loader

class Retriever : 
    def __init__(self , name , data_level = "folder"):
        self.vectorstore_path = VECTORSTORE_PATH[name]
        self.data_path = DATA_PATH[name]
        self.data_level = data_level 
        
        self.vectorstore = None 
        self.retriever = None 
        
        if os.path.exists(os.path.join(self.vectorstore_path , "index.faiss")):
            self.load()
        else : 
            self.build()
    def build(self):
        if self.data_level == "folder": 
            documents = data_loader.data_loader.load_folder(self.data_path)
        elif self.data_level == "multi_folders":
            documents = data_loader.data_loader.load_folder(self.data_path)
        
        index = faiss.IndexFlatL2(384)
        vectorstore = FAISS(
            embedding_function= embeddings , 
            index= index , 
            index_to_docstore_id= {},
            docstore= InMemoryDocstore()
        )
        vectorstore.add_documents(documents)
        vectorstore.save_local(self.vectorstore_path)       
        self.vectorstore = vectorstore 
        self.retriever = VectorStoreRetriever(vectorstore=vectorstore)
    
    def load(self):
        vectorstore = FAISS.load_local(
            folder_path= self.vectorstore_path , 
            embeddings= embeddings , 
            allow_dangerous_deserialization= True
        )
        
        self.vectorstore = vectorstore 
        self.retriever = VectorStoreRetriever(vectorstore=vectorstore)
        
## Test        
if __name__ == "__main__":
    a = Retriever(name = "CCCD")
    query = "Công dân đến cơ quan nào để cấp lại CCCD"
    
    answer = a.retriever.get_relevant_documents(query= query)
    for i , doc in enumerate(answer):
        print(f"--- Document {i} ---")
        print(doc.page_content)
        print(doc.metadata)