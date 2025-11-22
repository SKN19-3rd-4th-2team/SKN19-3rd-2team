import os
from typing import List

# LangGraph & LangChain 필수 임포트
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# [중요] checkpointer와 함께 사용하여 ReAct(생각-행동-관찰-생각) 루프를 
# 완벽하게 지원하는 LangGraph 전용 빌더를 사용합니다.
from langgraph.prebuilt import create_react_agent 
from langgraph.checkpoint.memory import MemorySaver

from total_tools import tool_search_ipc_code_with_description, tool_search_ipc_description_from_code,tool_search_patent
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
# 2. 에이전트 생성 (메모리 장착!)
# ==========================================

tools = [tool_search_ipc_code_with_description, tool_search_ipc_description_from_code,tool_search_patent]
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

# ★ 핵심 1: 메모리 저장소 초기화
memory = MemorySaver()

# ★ 핵심 2: create_react_agent 사용
# 이 함수는 도구 실행(tools) 후 자동으로 모델(agent)을 다시 호출하여 답변을 생성하는
# 그래프 구조를 자동으로 만들어줍니다.
agent_executor = create_react_agent(
    model=llm,       
    tools=tools, 
    checkpointer=memory
)

# ==========================================
# 3. 대화 실행 (Thread ID 활용)
# ==========================================

def chat_with_memory(user_input: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n\n--- 사용자({thread_id}): {user_input} ---")
    
    # stream 모드로 실행하며 내부 동작을 감시
    # stream_mode="values"를 사용하면 메시지의 누적 상태를 더 쉽게 볼 수 있지만,
    # 여기서는 기본 updates 모드를 사용하여 단계별 로그를 찍습니다.

    messages = [
        SystemMessage(content=SYSTEM_PROMPT, id="system_persona"), 
        HumanMessage(content=user_input)
    ]
    for event in agent_executor.stream(
        {"messages": messages},
        config=config
    ):
        # event는 {'node_name': {'messages': [...]}} 형태입니다.
        for node_name, value in event.items():
            
            # 메시지가 있는지 확인
            if "messages" in value:
                last_message = value["messages"][-1]
                
                # 1. 에이전트(모델)의 행동
                if node_name == "agent":
                    # (A) 도구 호출 명령이 있는 경우
                    if last_message.tool_calls:
                        tool_name = last_message.tool_calls[0]['name']
                        print(f"[Agent 생각]: '{tool_name}' 도구를 사용해야겠다...")
                    
                    # (B) 최종 답변이 있는 경우 (tool_calls가 없고 content가 있음)
                    elif last_message.content:
                        print(f"[Agent 답변]: {last_message.content}")

                # 2. 도구(Tool)의 실행 결과
                elif node_name == "tools":
                    # 도구 결과는 보통 ToolMessage 형태입니다.
                    # 내용이 너무 길 수 있으니 길이만 출력하거나 앞부분만 보여줍니다.
                    content_preview = str(last_message.content)[:100]
                    print(f"[Tool 결과]: 데이터 수신 완료 ({len(str(last_message.content))} 글자)")
                    # print(f"   ㄴ 내용: {content_preview}...") # 필요시 주석 해제
