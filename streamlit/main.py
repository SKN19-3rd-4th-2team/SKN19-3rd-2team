# app/main.py

import os

from dotenv import load_dotenv
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from total_tools import (
    tool_search_ipc_code_with_description,
    tool_search_ipc_description_from_code,
    tool_search_patent_with_description,
    tool_search_detail_patent_by_id,
)
from SYSTEM_PROMPT import SYSTEM_PROMPT

# ==========================================
# 1. 환경 변수 & LLM 설정
# ==========================================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 가 설정되지 않았습니다. .env 를 확인하세요.")

# ==========================================
# 2. 에이전트 생성 (ReAct + 메모리)
# ==========================================

tools = [
    tool_search_patent_with_description,
    tool_search_ipc_code_with_description,
    tool_search_ipc_description_from_code,
    tool_search_detail_patent_by_id,
]

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    api_key=OPENAI_API_KEY,
)

memory = MemorySaver()

llm_with_tools = llm.bind_tools(tools)


def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    # 기존 메시지 리스트에 새로운 응답을 추가하여 반환
    return {"messages": [response]}


# 3. 조건부 엣지 정의 (도구 사용 여부 판단)
def should_continue(state: MessagesState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    # LLM이 도구 호출을 요청했으면 'tools' 노드로 이동
    if last_message.tool_calls:
        return "tools"
    # 그렇지 않으면 종료
    return END


# 4. 그래프(Workflow) 구성
workflow = StateGraph(MessagesState)

# 노드 추가 (기존 app.py와의 호환성을 위해 이름을 'agent', 'tools'로 지정)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# 엣지 연결
workflow.add_edge(START, "agent")  # 시작 -> 에이전트

# 조건부 엣지: agent 결과에 따라 tools로 갈지, 끝낼지 결정
workflow.add_conditional_edges(
    "agent", should_continue, ["tools", END]  # 이동 가능한 경로 명시
)

# 도구 실행 후 다시 에이전트로 돌아옴 (ReAct 루프)
workflow.add_edge("tools", "agent")
# 5. 컴파일 (메모리 체크포인터 포함)
# app.py에서는 이 agent_executor를 그대로 사용하면 됩니다.
agent_executor = workflow.compile(checkpointer=memory)
# LangGraph 메모리: thread_id 기준으로 대화 컨텍스트를 저장

# ==========================================
# 3. 대화 함수 (스트리밍 + 툴 호출 로그)
# ==========================================


def chat_with_memory(user_input: str, thread_id: str = "default-thread") -> None:
    """
    한 턴의 사용자 입력에 대해 ReAct 에이전트를 실행하고,
    에이전트의 '생각 / 도구 호출 / 최종 답변'을 단계별로 콘솔에 출력합니다.
    """
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n\n=== 사용자({thread_id}) 입력 ===")
    print(user_input)
    print("================================\n")

    # 매 턴마다 system + user 메시지를 넣어줌
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ]

    step_idx = 0

    for event in agent_executor.stream({"messages": messages}, config=config):
        # event 예시: {"agent": {"messages": [...]}} 또는 {"tools": {"messages": [...]}}
        for node_name, value in event.items():
            messages_in_node = value.get("messages", [])
            if not messages_in_node:
                continue

            last_message = messages_in_node[-1]

            # 1) 에이전트 노드 (LLM)
            if node_name == "agent":
                step_idx += 1
                tool_calls = getattr(last_message, "tool_calls", None) or []

                # (A) 이번 step에서 도구를 호출하려는 경우
                if tool_calls:
                    tool_names = [tc.get("name", "UNKNOWN_TOOL") for tc in tool_calls]
                    print(f"[Step {step_idx}][Agent] 다음 도구 호출 예정: {tool_names}")

                # (B) 최종 자연어 답변 (tool_calls 없이 content만 있는 경우)
                elif last_message.content:
                    print(
                        f"[Step {step_idx}][Agent 최종 답변]\n{last_message.content}\n"
                    )

            # 2) 툴 노드
            elif node_name == "tools":
                # 툴 메시지는 ToolMessage 형태로 들어옴
                tool_name = getattr(last_message, "name", None) or getattr(
                    last_message, "tool", "unknown_tool"
                )
                content_str = str(last_message.content)
                preview = content_str[:120].replace("\n", " ")

                print(
                    f"[Tool 결과 수신] tool='{tool_name}' "
                    f"(내용 길이: {len(content_str)}자, 미리보기: {preview}...)"
                )
