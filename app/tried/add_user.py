# app/nodes/add_user.py
from langgraph.graph import MessagesState


from typing import Dict, Any
from langgraph.graph import MessagesState
from app.state import State



def add_user_node(state: State, user_input: str) -> State:
    """
    새 user 발화를 state.messages에 추가하는 노드.
    LangGraph에선 보통 state만 인자로 받게 만들지만,
    여기선 개념을 보기 쉽게 user_input을 분리해서 표시.
    실제 graph에서는 state 안에 이미 user 메시지가 들어있는 형태로 호출할 수도 있음.
    """
    messages = state.get("messages", [])
    messages.append({"role": "user", "content": user_input})

    # 턴 카운트 증가
    meta = state.get("meta", {})
    meta["turn_index"] = meta.get("turn_index", 0) + 1

    return {
        **state,
        "messages": messages,
        "meta": meta,
    }