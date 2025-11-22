# app/tools.py

from __future__ import annotations
from typing import List, Optional, TypedDict
from uuid import uuid4
from langchain_core.tools import tool
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from state import (
    PatentResult,
    PatentSearchRecord,
    IPCCodeInput,
    IPCDetailInfo,
    IPCKeywordInput,
    IPCMainDescription,
    OtherNoteRecord
)
from ipc_func import get_ipc_detail_data_from_code,search_ipc_with_query

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)




# 0) 키워드 개수 결정 함수

def decide_max_keywords(text: str) -> int:
    """
    입력 텍스트 길이에 따라 뽑을 키워드 개수를 3 / 5 / 7 중에서 선택.
    - 너무 짧은 설명에서 7개를 뽑으면 오히려 노이즈가 늘기 때문에 제한.
    """
    length = len(text)

    if length < 300:        # 짧은 설명 (예: 한두 문장)
        return 3
    elif length < 1500:     # 초록 + 몇 개 청구항 정도
        return 5
    else:                   # 청구항 여러 개, 긴 기술 설명
        return 7




# 1) LLM 기반 키워드 추출 툴

def tool_extract_core_keywords(
    tech_text: str,
    max_keywords: Optional[int] = None,
) -> List[str]:
    """
    기술문서(초록 + 청구항 등)에서 특허 검색용 핵심 키워드들을 뽑아서 리스트로 반환.

    - max_keywords 가 None 이면 decide_max_keywords()로 자동 결정.
    - GPT-4o를 사용해서 JSON 배열 형식의 키워드 목록을 받음.
    """
    if max_keywords is None:
        max_keywords = decide_max_keywords(tech_text)

    prompt = f"""
    너는 특허 검색을 위한 키워드 추출기야.

    아래 기술 설명을 읽고, 이 발명의 **핵심 기술 포인트**를 {max_keywords}개 이하의
    키워드로 뽑아줘.
    **핵심 기술 포인트**는 이 발명이 기존 특허들과 유사/중복되는 지점을 찾기 위해
    이 발명을 가장 잘 대표하는 검색용 키워드야.

    규칙:
    - "전자 장치", "터치 스크린 디스플레이", "무선 통신 회로", "센서", "메모리"처럼
      거의 모든 전자기기에 공통적으로 들어가는 일반적인 구성 요소 이름만
      단독으로 쓰지 마.
    - 대신, 이 발명에서 **특히 중요한 동작 흐름/조건/조합**을 담아서 써.
      예: "결제 직전/직후 위치 데이터 + POI 서버 전송",
          "환경 센서 기반 결제 로그 축적" 같은 식.
    - 각 키워드는 1줄에 하나씩만 쓰고, 설명 문장 대신
      **검색에 쓸 수 있는 짧은 구문** 형태로 써.
    - 가능하다면, 이 기술이 적용되는 **도메인
      (예: 공장 생산 라인, 의료 영상, 자율주행 등)** 이 드러나도록 작성해줘.

    기술 설명:
    \"\"\" 
    {tech_text}
    \"\"\"

    출력 형식(중요):
    - 반드시 JSON 배열 형태만 출력해.
    - 절대로 ``` 같은 마크다운 코드블록을 쓰지 마.
    - 'json'이라는 단어도 쓰지 마.
    - 예시:
      [
        "결제 직전 위치 데이터 + POI 서버 전송",
        "무선 결제 단말과 직접 연결된 위치 기반 결제",
        "환경 센서 기반 결제 로그 축적",
        "사용자 위치/환경 정보를 이용한 결제 기록 시스템"
      ]
    """

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )

    raw = res.choices[0].message.content.strip()

    # JSON 파싱 시도
    try:
        keywords = json.loads(raw)
        if isinstance(keywords, str):
            keywords = [keywords]
        elif not isinstance(keywords, list):
            keywords = [str(keywords)]
    except json.JSONDecodeError:
        # JSON 실패 시 줄 단위 분리 폴백
        keywords = [
            line.strip("-• ").strip()
            for line in raw.split("\n")
            if line.strip()
        ]

    cleaned = [k.strip() for k in keywords if str(k).strip()]

    return cleaned[:max_keywords]




# 2) 특허 검색 툴 (스텁)

@tool
def tool_search_patent(
    query_text: str,
    top_k: int = 5,
    exclude_app_nums: Optional[List[str]] = None,
) -> PatentSearchRecord:
    """
    [스텁 버전]
    - 나중에 벡터 DB(특허 28,000건)와 연결될 툴의 인터페이스 정의.
    - 지금은 DB가 없으므로, 구조만 맞는 가짜 결과를 반환한다.

    파라미터:
    - query_text: 특허 검색에 사용할 질의 텍스트 (키워드 또는 문장)
    - top_k: 상위 몇 개 결과를 원하나요?
    - exclude_app_nums: 제외하고 싶은 출원번호 리스트 (follow-up 시 활용 예정)

    반환:
    - PatentSearchRecord 타입(dict) 한 개
    """

    if exclude_app_nums is None:
        exclude_app_nums = []

    # TODO: 벡터 DB 붙인 후 실제 검색 구현
    fake_results: List[PatentResult] = []

    record: PatentSearchRecord = {
        "id": f"ps_{uuid4().hex[:8]}",
        "query": query_text,
        "top_k": top_k,
        "results": fake_results,
    }

    # exclude_app_nums 정보는 메타로 남겨도 됨 (원하면 여기에 필드 추가)
    if exclude_app_nums:
        record["exclude_app_nums"] = exclude_app_nums  # total=False 라서 나중에 추가 OK

    return record




# 3) 기술 설명 → IPC 추천 툴

@tool(args_schema=IPCKeywordInput)
def tool_search_ipc_code_with_description(
    tech_texts: List[str],
    top_k: int = 5,
) -> IPCMainDescription:
    """
    어떤 아이디어나 청구항의 내용에 대해 관계가 있는 IPC 코드들과 설명을 제시합니다.
    사용자가 아이디어나 청구항을 입력하거나 사용자의 아이디어의 키워드들에 관해 기입해야하거나 관련한 내용이 필요할때 사용하세요.
    tech_texts에 입력으로 넣어줄때에는 반드시 독립적 기술단위로 분해하여 영어로 번역하여 입력해주세요.
    예를들면 ['Organic Light Emitting Display with Pixel Electrode Contact Structure']의 형태가 아니라 ['Organic Light Emitting Display','Display Panel Opening Area','Pixel Electrode Contact Structure']처럼 기술단위로 분해해서 리스트로 입력해주세요.
    """
    result = search_ipc_with_query(tech_texts,top_k)
    return IPCMainDescription(**result)




# 4) IPC 코드 설명 툴
@tool(args_schema=IPCCodeInput)
def tool_search_ipc_description_from_code(codes: List[str]) -> List[IPCDetailInfo]:
    """
    IPC 코드 리스트를 입력받아 각 코드에 대한 상세 설명명과 코드의 계층, 해당 코드들의 상위 코드들에 대한 결과를 반환합니다.
    사용자가 특정 분류 코드의 의미를 물어보거나, IPC 코드의 전반적인 정보를 파악해야 하거나, 코드들 사이의 상위 관계나 계층을 파악해야 할 때 사용하세요.
    """
    
    # 1. 기존 함수 호출
    raw_results = get_ipc_detail_data_from_code(codes)
    
    # 2. 결과 검증 및 Pydantic 객체로 변환
    # (딕셔너리 리스트를 Pydantic 객체 리스트로 변환하여 LLM에게 전달)
    parsed_results = []
    for item in raw_results:
        # Pydantic 모델을 이용해 데이터 검증 및 포장
        parsed_results.append(IPCDetailInfo(**item))
    return parsed_results




# 5) OTHER용 메모 헬퍼

@tool
def tool_other_note(summary: str, detail: str = "") -> OtherNoteRecord:
    """
    '특허/IPC DB를 쓰지 않는 일반적인 질문'에 대해
    요약 메모를 구조화해서 남기고 싶을 때 사용할 수 있는 헬퍼.
    """
    record: OtherNoteRecord = {
        "id": f"other_{uuid4().hex[:8]}",
        "summary": summary,
        "detail": detail,
    }
    return record