from utils.tool_functions import ToolSelector
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from tool import CCCD
from llm.models import chat_llm_with_ragg

# Load biến môi trường
load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="Gemma2-9b-It"
)

tools = {
    "CCCD": CCCD.CCCD_RAGG
}

if __name__ == "__main__":
    selector = ToolSelector(llm=llm, tools=tools)

    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Assistant: Tạm biệt!")
            break
        if user_input.lower() in ["reset", "đổi chủ đề"]:
            selector.current_function = None
            print("Assistant: Đã đặt lại chủ đề.")
            continue

        try:
            answer = selector.call(user_input)
            print("Trợ lý:", answer)
        except Exception as e:
            print("Lỗi:", str(e))