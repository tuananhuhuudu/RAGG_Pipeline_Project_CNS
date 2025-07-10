import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from backend.config import (
    llm , 
    embeddings
)


# Prompt template của bạn giữ nguyên
def prompt_template(task):
    if task == "search_information":
        prompt_system = (
            """
            Bạn  là 1 trợ lý ảo . Bạn có nhiệm vụ là cung cấp và trả lời câu hỏi người dùng ,
            Bối cảnh:
            - Bạn đang hỗ trợ mọi người trong các thủ tục hành chính
            - Thông tin bạn cung cấp phải chính xác và dựa trên dữ liệu đã được xác nhận và phải cho người dùng cảm thấy dễ hiểu không có lan man , dài.
            - Nếu câu hỏi liên quan đến các thủ tục pháp luật, hành chính cá nhân và giấy tờ công dân (như CCCD, giấy khai sinh, hộ chiếu, đăng ký cư trú,...), cho người dùng cảm thấy dễ hiểu không có lan man , dài., hướng dẫn từng bước và nêu quy định pháp luật áp dụng.

            Nhớ rằng : 
            - Hãy dựa trên thông tin được cung cấp để trả lời câu hỏi 
            - Nếu không thể trả lời câu hỏi dựa trên thông tin được cung cấp hoặc không thấy thông tin ,
            bạn hãy trả lời rằng : hmm...Hiện nay tôi không có thông tin và đủ căn cứ để trả lời câu hỏi trên! , hoặc là bạn có thể vui lòng cung cấp câu hỏi rõ ràng hơn 
            """
        )
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_system),
                ("system", "Thông tin cung cấp:\n{context}"),
                ("human", "{input}")
            ]
        )
    else:
        prompt_template = ChatPromptTemplate.from_template(
        """
        Bạn là một trợ lý ảo thông minh. Nhiệm vụ của bạn là chọn function phù hợp nhất để xử lý câu hỏi của người dùng.
        Dưới đây là danh sách các function có sẵn và mô tả của chúng:
        {tool_description}

        Câu hỏi:
        {question}

        Dựa trên câu hỏi của người dùng, bạn cần chọn function phù hợp nhất.
        Nếu cảm thấy câu hỏi không liên quan đến các function mà bạn có 
        Hãy trả lời rằng (hmm câu hỏi không rõ rằng or thông tin này tôi chưa được cập nhật)
        Trả lời chỉ với tên function mà bạn chọn, không cần giải thích gì thêm.
        """
    )
    return prompt_template

def chat_llm_with_ragg(task: str, params={}):
    prompt = prompt_template(task)
    formatted_messages = prompt.format_messages(**params)
    response = llm.invoke(formatted_messages)

    return response.content

if __name__ == "__main__":
    response = chat_llm_with_ragg(
        task="search_information",
        params={
            "context": """
                Theo Điều 23 Luật Cư trú 2020, hồ sơ đăng ký tạm trú gồm:
                - Tờ khai
                - Căn cước công dân (CCCD)
                - Giấy tờ chứng minh chỗ ở hợp pháp
            """,
            "input": "Tôi cần làm hồ sơ đăng ký tạm trú"
        }
    )
    
    print("Assistant:", response)
