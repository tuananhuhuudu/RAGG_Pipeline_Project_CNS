import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load biến môi trường
load_dotenv()

# Khởi tạo Embeddings
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Khởi tạo LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="Gemma2-9b-It"
)

# Prompt template của bạn giữ nguyên
def prompt_template(task):
    if task == "search_information":
        prompt_system = (
            """
            Bạn  là 1 trợ lý ảo . Bạn có nhiệm vụ là cung cấp và trả lời câu hỏi người dùng ,
            có thể tham chiếu đến các nội dung trước đó và trả lời 1 cách chính xác nhất cho 
            câu hỏi mới 
            Bối cảnh:
            - Bạn đang hỗ trợ mọi người trong các thủ tục hành chính
            - Thông tin bạn cung cấp phải chính xác và dựa trên dữ liệu đã được xác nhận.
            - Nếu câu hỏi liên quan đến các thủ tục pháp luật, hành chính cá nhân và giấy tờ công dân (như CCCD, giấy khai sinh, hộ chiếu, đăng ký cư trú,...), hãy trả lời chi tiết, hướng dẫn từng bước và nêu quy định pháp luật áp dụng.

            Nhớ rằng : 
            - Hãy dựa trên thông tin được cung cấp để trả lời câu hỏi 
            - Nếu không thể trả lời câu hỏi dựa trên thông tin được cung cấp hoặc không thấy thông tin ,
            bạn hãy trả lời rằng : hmm...Hiện nay tôi không có thông tin và đủ căn cứ để trả lời câu hỏi trên!
            """
        )
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_system),
                ("system", "Thông tin cung cấp:\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
    else:
        prompt_template = ChatPromptTemplate.from_template(
            """
            Bạn là một trợ lý ảo thông minh. Nhiệm vụ của bạn là chọn function phù hợp nhất để xử lý câu hỏi của người dùng.
            Dưới đây là danh sách các function có sẵn và mô tả của chúng:
            {functions_description}

            Câu hỏi:
            {question}

            Dựa trên câu hỏi của người dùng, bạn cần chọn function phù hợp nhất.
            Trả lời chỉ với tên function mà bạn chọn, không cần giải thích gì thêm.
            """
        )
    return prompt_template

def chat_llm_with_ragg(messages: str, task: str, params={}):
    """
    messages: session_id (str) để lưu lịch sử hội thoại
    task: tên tác vụ
    params: dict chứa context, input,...
    """
    prompt = prompt_template(task)
    chain = prompt | llm

    runnable = RunnableWithMessageHistory(
        chain,
        lambda session_id: InMemoryChatMessageHistory(),
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    # Gọi model
    response = runnable.invoke(
        params,
        config={"configurable": {"session_id": messages}}
    )

    return response.content






if __name__ == "__main__":
    session_id = "user_001"

    result1 = chat_llm_with_ragg(
        messages=session_id,
        task="search_information",
        params={
            "context": """
                Theo Điều 23 Luật Cư trú 2020, hồ sơ đăng ký tạm trú gồm:
                - Tờ khai
                - CCCD
                - Giấy tờ chứng minh chỗ ở hợp pháp
            """,
            "input": "Tôi cần làm hồ sơ đăng ký tạm trú"
        }
    )
    print("Assistant:", result1)

    result2 = chat_llm_with_ragg(
        messages=session_id,
        task="search_information",
        params={
            "context": """
                Thời hạn giải quyết hồ sơ đăng ký tạm trú là 3 ngày làm việc.
            """,
            "input": "Thời hạn giải quyết hồ sơ là bao lâu?"
        }
    )
    print("Assistant:", result2)
