import os 

import faiss 
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.vectorstores import VectorStoreRetriever

# from core.models import embeddings 
from utils.data_loader import data_loader
## Test
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv 
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
class Retriever:
    def __init__(self , name : str , data_level = "folder"):
        from paths import VECTOR_STORE_PATHS , TXT_DATA_PATHS
        
        self.data_path = TXT_DATA_PATHS[name]
        self.vector_store_path = VECTOR_STORE_PATHS[name]
        self.data_level = data_level
        
        self.vector_store = None 
        self.retriever = None 
        
        # Kiểm tra vectorstore đã tồn tại chưa 
        if os.path.exists(os.path.join(self.vector_store_path , "index.faiss")):
            self.load()
        else :
            self.build()
    def build(self):
        # Load data
        if self.data_level == "folder":
            texts = data_loader.load_folder(self.data_path)
        elif self.data_level == "multi_folders":
            texts = data_loader.load_data(self.data_path)
        index = faiss.IndexFlatL2(384)
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        
        vector_store.add_documents(texts)
        vector_store.save_local(self.vector_store_path)
        self.vector_store = vector_store
        self.retriever = VectorStoreRetriever(vectorstore=vector_store)
        
    def load(self):
        vector_store = FAISS.load_local(
            folder_path=self.vector_store_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        self.vector_store = vector_store
        self.retriever = VectorStoreRetriever(vectorstore=vector_store)
        
if __name__ == "__main__":
    a = Retriever("chi_tieu")
    query = "Tuấn Anh"
    docs = a.retriever.get_relevant_documents(query)

    for i, doc in enumerate(docs, 1):
        print(f"--- Document {i} ---")
        print(doc.page_content)
        print(doc.metadata)