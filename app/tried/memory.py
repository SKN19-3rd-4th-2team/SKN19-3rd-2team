# app/nodes/memory.py

from app.state import State

MAX_MESSAGES = 30  # 예시: 너무 길어지면 앞부분부터 줄이기


def memory_node(state: State) -> State:
    """
    v1: messages 길이가 너무 길어지면 앞부분 일부를 잘라내는 단순 버전.
    v2에서는 LLM을 사용해서 short_summary를 생성하도록 확장할 예정.
    """
    messages = state.get("messages", [])

    if len(messages) > MAX_MESSAGES:
        # 나중엔 여기서 LLM으로 요약해서 short_summary에 넣고,
        # 오래된 메시지는 더 과감하게 줄이는 방향으로 확장 가능.
        trimmed = messages[-MAX_MESSAGES:]
        return {
            **state,
            "messages": trimmed,
        }

    return state