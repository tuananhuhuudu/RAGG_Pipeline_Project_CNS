# import os 
# data_path = os.path.join(os.getcwd() , ".." , "ragg")
# print(data_path)
# data_path = os.path.abspath(data_path)
# print(data_path)

import os 
print(os.getcwd())
data_path = os.path.join(os.getcwd() , "ragg" , "data")

VECTORSTORE_PATH =  {
    "CCCD" : os.path.join(data_path , "vectorstore" , "CCCD")
}
DATA_PATH = {
    "CCCD" : os.path.join(data_path , "data_txt" , "CCCD_Guide")
}
print("Exists:", os.path.exists(DATA_PATH["CCCD"]))