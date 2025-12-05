import os
import uuid
import streamlit as st
from typing import Generator, Dict, Any, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from total_tools import (
    tool_search_ipc_code_with_description,
    tool_search_ipc_description_from_code,
    tool_search_detail_patent_by_id,
    tool_search_patent_with_description,
)
from SYSTEM_PROMPT import SYSTEM_PROMPT
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API 키 검증
if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    st.stop()

# ==========================================
# 1. 에이전트 초기화
# ==========================================


@st.cache_resource
def initialize_agent():
    tools = [
        tool_search_ipc_code_with_description,
        tool_search_ipc_description_from_code,
        tool_search_detail_patent_by_id,
        tool_search_patent_with_description,
    ]
    llm = ChatOpenAI(model="gpt-5.1", temperature=0, api_key=OPENAI_API_KEY)
    memory = MemorySaver()
    llm_with_tools = llm.bind_tools(tools)

    # 2. 에이전트 노드 정의 (LLM 호출)
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

    return agent_executor


# ==========================================
# 2. 세션 상태 초기화
# ==========================================


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())


# ==========================================
# 3. 대화 기록 표시
# ==========================================


def display_chat_messages():
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.write(content)


# ==========================================
# 4. 에이전트 응답 스트리밍
# ==========================================


def stream_agent_response(agent_executor, user_input: str, thread_id: str) -> Generator:
    config = {"configurable": {"thread_id": thread_id}}

    messages = [
        SystemMessage(content=SYSTEM_PROMPT, id="system_persona"),
        HumanMessage(content=user_input),
    ]

    # [핵심] 이미 UI에 표시한 도구 호출 ID를 기억하기 위한 집합(Set)
    seen_tool_ids = set()

    try:
        for msg, metadata in agent_executor.stream(
            {"messages": messages}, config=config, stream_mode="messages"
        ):
            curr_node = metadata.get("langgraph_node", "")

            # 1. 텍스트 답변 생성 중 (순수 답변)
            if curr_node == "agent" and msg.content:
                if not msg.tool_calls:
                    yield {"type": "token", "content": msg.content}

            # 2. 도구 호출 요청 감지
            if curr_node == "agent" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # 도구마다 고유 ID와 이름이 있습니다.
                    t_id = tool_call.get("id")
                    t_name = tool_call.get("name")

                    # [수정] 이름이 존재하고, 아직 보여주지 않은(새로운) 도구일 때만 yield
                    if t_name and (t_id not in seen_tool_ids):
                        seen_tool_ids.add(t_id)  # 표시한 ID로 등록
                        yield {"type": "tool_call", "tool_name": t_name}

            # 3. 도구 실행 결과 수신
            if curr_node == "tools":
                content_str = str(msg.content)
                # 도구 결과는 한 번만 오므로 별도 필터링 불필요
                yield {"type": "tool_result", "length": len(content_str)}

    except Exception as e:
        yield {"type": "error", "message": str(e)}


# ==========================================
# 5. 사용자 입력 처리
# ==========================================


def process_user_input(user_input: str, agent_executor):
    # 1. 사용자 메시지 기록 및 표시
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # 2. 에이전트 응답 영역
    with st.chat_message("assistant"):
        # 처음에는 기본 메시지로 시작 (접힌 상태)
        status_container = st.status("요청을 분석하고 있습니다...", expanded=False)

        response_placeholder = st.empty()
        full_response = ""
        last_tool_name = ""

        try:
            stream_gen = stream_agent_response(
                agent_executor, user_input, st.session_state.thread_id
            )

            for event in stream_gen:
                event_type = event.get("type")

                # [A] 텍스트 토큰 (답변 생성 중)
                if event_type == "token":
                    token = event.get("content")
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

                    # 텍스트가 나오기 시작하면 상태창 제목을 '답변 작성 중'으로 변경 (선택사항)
                    # 너무 자주 바뀌는 게 싫으면 이 부분은 주석 처리해도 됩니다.
                    # if last_tool_name:
                    #     status_container.update(label="✍️ 답변을 작성하고 있습니다...", state="running")

                # [B] 도구 호출 시작 -> 제목에 도구 이름 표시!
                elif event_type == "tool_call":
                    tool_name = event.get("tool_name")
                    last_tool_name = tool_name

                    # [핵심] 상태창 제목을 즉시 변경하여 현재 사용하는 도구를 보여줌
                    status_container.update(
                        label=f"🛠️ {tool_name} 도구를 사용 중입니다...", state="running"
                    )

                    # 내부에 로그 기록 (클릭해야 보임)
                    status_container.write(f"▪️ 도구 호출: {tool_name}")

                # [C] 도구 결과 수신 -> 제목에 완료 표시
                elif event_type == "tool_result":
                    length = event.get("length")

                    # [핵심] 해당 도구 완료 상태 표시
                    status_container.update(
                        label=f"✅ {last_tool_name} 처리 완료 ({length}자 수신)",
                        state="running",
                    )

                    # 내부에 로그 기록
                    status_container.write(f"▪️ 결과 수신: {length} 글자")

                # [D] 에러 처리
                elif event_type == "error":
                    status_container.update(
                        label="⚠️ 오류가 발생했습니다", state="error"
                    )
                    st.error(f"오류 내용: {event.get('message')}")
                    return

            # 3. 최종 완료 처리
            response_placeholder.markdown(full_response)

            # 모든 과정 종료 시
            status_container.update(
                label="답변 생성이 완료되었습니다.", state="complete", expanded=False
            )

            # 대화 기록 저장
            if full_response:
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

        except Exception as e:
            st.error(f"처리 중 오류가 발생했습니다: {str(e)}")


# ==========================================
# 6. 메인 UI
# ==========================================


def main():
    # 페이지 설정
    st.set_page_config(
        page_title="컴퓨터 비전 전용 특허 에이전트 챗봇", page_icon="💻", layout="wide"
    )

    # 제목
    st.title("💻 컴퓨터 비전 전용 특허 에이전트 챗봇")
    st.markdown("컴퓨터 비전 관련 특허 분류 및 유사 특허 검색을 도와드립니다.")

    # 초기화
    agent_executor = initialize_agent()
    initialize_session_state()

    # 대화 초기화 (사이드 바)
    with st.sidebar:
        st.header("설정")

        if st.button("🔄 새 대화 시작", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        st.divider()

        st.caption(f"현재 세션 ID: {st.session_state.thread_id[:8]}...")
        st.caption(f"메시지 수: {len(st.session_state.messages)}")

    # 대화 기록 표시
    display_chat_messages()

    # 사용자 입력
    if user_input := st.chat_input("컴퓨터 비전 특허 관련 질문을 입력하세요..."):
        process_user_input(user_input, agent_executor)


# ==========================================
# 7. 실행
# ==========================================

if __name__ == "__main__":
    main()
