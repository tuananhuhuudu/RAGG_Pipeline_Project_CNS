import os

from backend.orchestrator.tool_selector import ToolSelector
from backend.tool import CCCD

from backend.config import (
    llm
)
from backend.llm.models import chat_llm_with_ragg

from langchain_core.prompts import ChatPromptTemplate

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