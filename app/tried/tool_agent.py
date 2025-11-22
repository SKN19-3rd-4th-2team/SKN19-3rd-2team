# app/nodes/tool_agent.py

from __future__ import annotations
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, BaseMessage, AIMessage

from app.state import State
from app.tools import (
    tool_extract_core_keywords,
    tool_search_patent,
    tool_search_ipc_for_tech,
    tool_describe_ipc_codes,
    tool_other_note,   # 원하면 빼도 됨
)

# ---- 1) 에이전트가 쓸 수 있는 도구 목록 ----
TOOLS = [
    tool_extract_core_keywords,
    tool_search_patent,
    tool_search_ipc_for_tech,
    tool_describe_ipc_codes,
    tool_other_note,
]

# ---- 2) LLM + tools 바인딩 ----
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
)

llm_with_tools = llm.bind_tools(TOOLS)

# ---- 3) 에이전트용 시스템 프롬프트 ----
AGENT_SYSTEM_PROMPT = """
너는 '컴퓨터 비전 관련 특허/IPC 질의응답 시스템'의 메인 에이전트야.

[역할]
- 사용자의 한국어 질문을 읽고,
- 필요하다면 특허 벡터 DB / IPC DB에 접근하는 도구들을 호출해서,
- 특허 검색 / IPC 추천 / IPC 코드 설명 / 일반 질문 등을 처리해.
- 최종적으로는 항상 한국어로, 친절하고 이해하기 쉽게 답해야 한다.

[너가 쓸 수 있는 도구들]

1) tool_extract_core_keywords
- 입력: 기술 설명(초록, 청구항, 발명 아이디어 등) 문자열, max_keywords(선택)
- 출력: 이 기술을 특허 검색에 사용하기 좋은 핵심 키워드 리스트
- 사용 시점:
  - 사용자가 "이런 기술/발명에 대한 비슷한 특허가 있냐"고 물을 때,
  - 특허 검색을 하기 전에, 긴 설명을 요약된 검색 키워드로 뽑고 싶을 때.

2) tool_search_patent
- 입력: query_text (검색 질의), top_k (원하는 결과 개수), exclude_app_nums(선택)
- 출력: 특허 검색 기록(PatentSearchRecord) - 검색 질의와 결과 목록 포함
- 사용 시점:
  - "비슷한 특허를 찾아줘", "유사 특허 상위 5개 보여줘" 같은 요청.
- 보통 흐름:
  - 긴 기술 설명 → 먼저 tool_extract_core_keywords 호출 → 키워드들을 적당히 묶어서 query_text에 사용 → tool_search_patent 호출.

3) tool_search_ipc_for_tech
- 입력: tech_text (기술 설명), top_k
- 출력: 이 기술에 어울리는 IPC 코드 후보 리스트(IPCRequestRecord)
- 사용 시점:
  - "이 발명은 어떤 IPC로 분류하는 게 좋을까?",
  - "이 기술에 어울리는 IPC를 추천해줘" 같은 요청.

4) tool_describe_ipc_codes
- 입력: ipc_codes (예: ["G06T", "G06F", "A61B"])
- 출력: 각 IPC 코드의 제목/설명 리스트(IPCRequestRecord)
- 사용 시점:
  - "IPC G06T가 뭐야?", "IPC D, G에 대해 설명해줘" 같은 요청.

5) tool_other_note
- 입력: summary, detail(선택)
- 출력: 기타 메모 레코드(OtherNoteRecord)
- 사용 시점:
  - 특허/IPC DB를 쓰지 않는 일반적인 질문이지만,
  - 나중에 참고할 메모로 남겨두고 싶을 때 (필수는 아님).

[도구 사용 전략]

- 사용자의 한 번의 입력 안에 여러 요청이 섞여 있을 수 있다:
  - 예: "보온병 관련 특허 5개 찾아줘, 거리 기반 곡면 디스플레이 10개 찾아줘, 각각 IPC 추천도 해줘, IPC G06F/G06T 설명도 해줘, 그리고 출원 비용도 알려줘."
- 이 경우, 필요한 만큼 도구를 여러 번 호출해도 된다.
  - 예:
    - 보온병 기술 → tool_extract_core_keywords → tool_search_patent
    - 거리 기반 곡면 디스플레이 기술 → tool_extract_core_keywords → tool_search_patent
    - 두 기술에 대해 각각 tool_search_ipc_for_tech
    - IPC 코드 설명 요청 → tool_describe_ipc_codes
    - 출원 비용이나 절차 등은 일반 상식으로 직접 답변 (도구 없이).

[응답 형식]

- 도구를 호출할지 말지는 네가 판단해.
- 도구 호출이 필요 없으면, 바로 한국어로 답해.
- 도구를 호출한 뒤에는:
  - 도구 결과(특허 목록, IPC 후보, 코드 설명 등)를 잘 요약/해석해서,
  - 사용자 입장에서 이해하기 쉽게 설명해줘.
- 하나의 입력에 여러 작업이 섞여 있어도,
  - 최종 답변은 하나로 묶어서 정리해줘.
  - 예: 특허 검색 결과 요약 + IPC 추천 요약 + 부가 설명 등.

- 항상:
  - 한국어로 대답할 것.
  - 너무 딱딱하지 않게, 하지만 과장 없이 사실 기반으로 설명할 것.
"""

def _ensure_system_prompt(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    messages 리스트의 첫 번째에 시스템 프롬프트가 없으면 추가.
    이미 SystemMessage가 있으면 그대로 둔다.
    """
    if not messages:
        return [SystemMessage(content=AGENT_SYSTEM_PROMPT)]

    first = messages[0]
    if isinstance(first, SystemMessage):
        return messages

    return [SystemMessage(content=AGENT_SYSTEM_PROMPT), *messages]


def tool_agent_node(state: State) -> State:
    """
    메인 에이전트 노드.
    - state.messages (대화 내역)를 보고,
    - LLM + tools를 한 번 호출하고,
    - AIMessage를 messages에 추가,
    - meta.turn_index / meta.last_used_tools 를 업데이트한다.
    """
    # 1) 현재 메시지들 가져오기
    messages: List[BaseMessage] = list(state.get("messages", []))

    # 2) 시스템 프롬프트 보장
    messages_with_system = _ensure_system_prompt(messages)

    # 3) LLM 호출 (도구 바인딩된 모델)
    ai_message: AIMessage = llm_with_tools.invoke(messages_with_system)

    # 4) messages 업데이트 (state에는 system 안 겹치게, 원래 messages에만 assistant 추가)
    new_messages: List[BaseMessage] = messages + [ai_message]

    # 5) meta 업데이트
    meta = dict(state.get("meta", {}))
    meta["turn_index"] = meta.get("turn_index", 0) + 1

    # tool_calls에서 이번 턴에 어떤 툴을 썼는지 기록 (없으면 빈 리스트)
    last_used_tools: List[str] = []
    tool_calls = getattr(ai_message, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            # LangChain ToolCall은 dict 형태로 name 필드를 가짐
            name = tc.get("name")
            if name:
                last_used_tools.append(name)

    meta["last_used_tools"] = last_used_tools

    return {
        **state,
        "messages": new_messages,
        "meta": meta,
    }
