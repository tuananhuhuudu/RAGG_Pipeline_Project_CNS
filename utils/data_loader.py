import os 

from langchain_community.document_loaders.text import TextLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from tqdm import tqdm
data_path = os.path.join(os.getcwd() , "documents" , "data")

class DataLoader : 
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000 , 
            chunk_overlap = 200 , 
            is_separator_regex= False 
        )
    def load_file(self , file_data):
        data =  []
        docs = TextLoader(file_path = file_data , encoding= "utf-8")
        documents = docs.load()
        for doc in documents:
            data.append(doc.page_content)
        
        texts = self.text_splitter.create_documents(data)
        
        return texts 
    
    def load_folder(self , folder_path):
        texts = []
        for file in tqdm(os.listdir(folder_path), desc = "Loading...."):
            if file.endswith(".txt"):
                txt_file_path = os.path.join(folder_path , file)
                documents = self.load_file(txt_file_path)
                texts += documents
        return texts 
    
    def load_dir(self , data_path):
        texts = []
        for folder in os.scandir(data_path):
            if folder.is_dir():
                documents = self.load_folder(folder.path)
                texts += documents
        return texts
    
# if __name__ == "__main__":
#     a = DataLoader()
#     data_path = os.path.join(os.getcwd() , "documents" , "data")
#     text = a.load_dir(data_path)
#     print(text)

data_loader = DataLoader()