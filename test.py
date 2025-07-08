from tool import (
    CCCD,
)
from llm.models import chat_llm_with_ragg

tool = {
    "CCCD": CCCD.CCCD_RAGG
}

tool_description = "\n".join(
    [f"{name}: {obj.description}" for name, obj in tool.items()]
)

def tool_calling(question: str) -> str:
    response = chat_llm_with_ragg(
        task="function_calling",
        params={
            "question": question,
            "tool_description": tool_description
        }
    ).strip() 

    if response not in tool.keys():
        raise ValueError(f"Không tìm thấy function: {response}")

    print(f"Chọn function: {response}")

    return tool[response].invoke(question=question)

if __name__ == "__main__":
    answer = tool_calling(question="Quy trình làm lại CCCD")
    print("Trợ lý trả lời:", answer)
