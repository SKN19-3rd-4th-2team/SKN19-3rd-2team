import os
import uuid
import streamlit as st
from typing import Generator, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent 
from langgraph.checkpoint.memory import MemorySaver

from total_tools import (
    tool_search_ipc_code_with_description, 
    tool_search_ipc_description_from_code,
    tool_search_patent
)
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API 키 검증
if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    st.stop()

SYSTEM_PROMPT = """
당신은 20년 경력의 'IPC(국제특허분류) 전문 베테랑 변리사'입니다. 
당신의 목표는 사용자의 발명 아이디어나 기술 내용을 분석하여, 특허 서류 작성을 보조하거나 사용자의 아이디어에 관련된 IPC코드등을 제공하고 필요하다면 이에 대한 설명도 제공해야합니다.
또한, 이미 시중에 공개된 유사한 특허에 대한 정보를 검색하고 이를 통해 유사한 특허의 정보를 제공하거나 사용자의 아이디어와의 차별점을 분석해서 제공하여야 합니다.
사용자의 목적에 관한 답변을 해주되 최신 특허정보나 IPC코드에 관한 정보에 관해서는 주어진 도구를 이용하여 검색하여 정보를 얻고 이를 바탕으로 사용자에게 최적화된 답변을 제공하세요.

다음 지침을 반드시 따르십시오:
1. [전문성] 단순히 검색 결과 리스트만 나열하지 마십시오. 각 코드가 왜 사용자의 기술과 관련이 있는지 전문가적 견해(Insight)를 덧붙여 설명하세요.
2. [구조적 설명] IPC 코드를 설명할 때는 가능하다면 섹션(Section) -> 클래스(Class) -> 그룹(Group)의 계층 구조를 이해하기 쉽게 풀어서 설명하세요.
3. [친절하되 명확함] 사용자가 비전문가일 수 있음을 고려하여 전문 용어는 쉽게 풀어서 설명하되, 내용은 정확해야 합니다.
4. [도구 활용] 사용자의 질문이 모호하면, 먼저 아이디어를 구체화하기 위한 질문을 하거나, 주어진 도구를 활용하여 최대한 근접한 기술 분류를 탐색하십시오.
5. [답변 스타일] 문장은 정중하고 논리적인 '변리사' 톤을 유지하세요. (예: "~것으로 판단됩니다.", "~분류가 적합해 보입니다.")
6. [추가 정보 요구] 만약 사용자가 제공한 정보중에서 부족하거나 보충해야하는 부분이 있다면 정보를 요구하세요.(예: "~ 것에 관한 부분이 모호합니다. ~를 의미한 건가요?","~에 관한 부분의 정보가 부족합니다. ~점을 더 이야기해주세요.")
"""

# ==========================================
# 1. 에이전트 초기화 
# ==========================================

@st.cache_resource
def initialize_agent():
    tools = [
        tool_search_ipc_code_with_description, 
        tool_search_ipc_description_from_code,
        tool_search_patent
    ]
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
    memory = MemorySaver()
    
    agent_executor = create_react_agent(
        model=llm,       
        tools=tools, 
        checkpointer=memory
    )
    
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
        elif role == "tool":
            with st.chat_message("assistant"):
                tool_name = message.get("tool_name", "도구")
                st.caption(f"{tool_name} 실행 중...")

# ==========================================
# 4. 에이전트 응답 스트리밍
# ==========================================

def stream_agent_response(agent_executor, user_input: str, thread_id: str) -> Generator:
    config = {"configurable": {"thread_id": thread_id}}
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT, id="system_persona"), 
        HumanMessage(content=user_input)
    ]
    
    final_answer = ""
    tool_calls_info = []
    
    try:
        for event in agent_executor.stream({"messages": messages}, config=config):
            for node_name, value in event.items():
                if "messages" in value:
                    last_message = value["messages"][-1]
                    
                    # 에이전트 노드
                    if node_name == "agent":
                        # 도구 호출
                        if last_message.tool_calls:
                            tool_name = last_message.tool_calls[0]['name']
                            tool_calls_info.append({
                                "role": "tool",
                                "content": f"도구 실행: {tool_name}",
                                "tool_name": tool_name
                            })
                            yield {"type": "tool_call", "tool_name": tool_name}
                        
                        # 최종 답변
                        elif last_message.content:
                            final_answer = last_message.content
                            yield {"type": "answer", "content": final_answer}
                    
                    # 도구 노드
                    elif node_name == "tools":
                        content_length = len(str(last_message.content))
                        yield {"type": "tool_result", "length": content_length}
        
        # 최종 답변 반환
        if final_answer:
            yield {"type": "final", "content": final_answer, "tool_calls": tool_calls_info}
    
    except Exception as e:
        yield {"type": "error", "message": str(e)}

# ==========================================
# 5. 사용자 입력 처리
# ==========================================

def process_user_input(user_input: str, agent_executor):
    # 사용자 메시지 추가
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.write(user_input)
    
    # 에이전트 응답 생성
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        tool_placeholder = st.empty()
        
        final_answer = ""
        tool_calls = []
        
        try:
            with st.spinner("생각 중..."):
                for event in stream_agent_response(agent_executor, user_input, st.session_state.thread_id):
                    event_type = event.get("type")
                    
                    if event_type == "tool_call":
                        tool_name = event.get("tool_name")
                        tool_placeholder.caption(f"{tool_name} 실행 중...")
                        tool_calls.append({
                            "role": "tool",
                            "content": f"도구 실행: {tool_name}",
                            "tool_name": tool_name
                        })
                    
                    elif event_type == "tool_result":
                        length = event.get("length")
                        tool_placeholder.caption(f"데이터 수신 완료 ({length} 글자)")
                    
                    elif event_type == "answer":
                        final_answer = event.get("content")
                        response_placeholder.write(final_answer)
                    
                    elif event_type == "final":
                        final_answer = event.get("content")
                        response_placeholder.write(final_answer)
                        tool_placeholder.empty()
                    
                    elif event_type == "error":
                        error_msg = event.get("message")
                        st.error(f"오류가 발생했습니다: {error_msg}")
                        return
            
            # 에이전트 응답을 세션 상태에 추가
            if final_answer:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer
                })
                
                # 도구 호출 정보 저장 
                for tool_call in tool_calls:
                    st.session_state.messages.append(tool_call)
        
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")

# ==========================================
# 6. 메인 UI
# ==========================================

def main():
    # 페이지 설정
    st.set_page_config(
        page_title="IPC 전문 변리사 챗봇",
        page_icon="⚖️",
        layout="wide"
    )
    
    # 제목
    st.title("⚖️ IPC 전문 변리사 챗봇")
    st.markdown("특허 분류 및 유사 특허 검색을 도와드립니다.")
    
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
    if user_input := st.chat_input("특허 관련 질문을 입력하세요..."):
        process_user_input(user_input, agent_executor)

# ==========================================
# 7. 실행
# ==========================================

if __name__ == "__main__":
    main()
