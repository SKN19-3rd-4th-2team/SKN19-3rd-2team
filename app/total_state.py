# app/state.py

from typing import TypedDict, List, Dict, Any, Literal, Optional
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

# 2. 특허 검색 입력 스키마 (tool_search_patent에서 사용)
class PatentSearchInput(BaseModel):
    """
    특허 벡터 DB에서 유사 특허를 검색할 때 사용하는 입력 스키마.
    """
    query_text: str = Field(
        ...,
        description=(
            "유사 특허를 찾기 위한 핵심 기술 설명 또는 키워드. "
            "LLM이 추출한 키워드들의 조합이거나, 사용자의 원문 기술 설명을 "
            "간단히 정제한 문장일 수 있다."
        ),
    )
    top_k: int = Field(
        5,
        description=(
            "최종적으로 사용자에게 보여줄 특허 개수. "
            "사용자가 '상위 5개', '10개 정도'처럼 개수를 말하면 그 값을 사용하고, "
            "언급이 없으면 기본값 5를 사용한다."
        ),
        ge=1,
        le=30,  # 너무 크게 안 가져오도록 상한 설정
    )
    max_claims_per_patent: int = Field(
        3,
        description=(
            "각 특허별로 포함할 상위 청구항 개수. "
            "patent_hybrid_search 함수의 max_claims_per_patent에 매핑된다."
        ),
        ge=1,
        le=5,
    )
    exclude_patent_ids: List[str] = Field(
        default_factory=list,
        description=(
            "이번 검색에서 제외해야 할 특허 ID 목록. "
            "예: 이전 턴에서 이미 보여준 특허들을 다시 보여주지 않기 위해 사용한다."
        ),
    )


# 3. 개별 청구항 스니펫 스키마 (patent_hybrid_search의 claims 원소)
class PatentClaimSnippet(BaseModel):
    """
    특정 특허 안에서 검색에 걸린 개별 청구항 정보를 담는 스키마.
    """
    id: str = Field(
        ...,
        description="벡터 DB 내 청구항 단위 문서의 내부 id.",
    )
    document: str = Field(
        ...,
        description="청구항 원문 텍스트.",
    )
    title: str = Field(
        "",
        description="해당 청구항이 속한 특허의 발명의 명칭(메타데이터에 있을 경우).",
    )
    distance: float = Field(
        ...,
        description="벡터 거리 값. 0에 가까울수록 쿼리와 더 유사하다.",
    )
    hybrid_score: float = Field(
        ...,
        description="벡터 점수와 BM25 점수를 결합한 최종 하이브리드 점수. 클수록 유사도가 높다.",
    )


# 4. 특허 단위 검색 결과 스키마 (aggregated의 각 원소)
class PatentSearchResult(BaseModel):
    """
    특허 단위로 집계된 검색 결과.
    patent_hybrid_search가 반환하는 aggregated 리스트의 한 원소에 해당한다.
    """
    patent_id: str = Field(
        ...,
        description=(
            "특허 출원번호 메타데이터의 'patent_id'와 동일하며, "
            "실제 출원번호 또는 내부 id로 사용될 수 있다."
        ),
    )
    score: float = Field(
        ...,
        description="특허 단위 최종 하이브리드 점수. 상위 랭킹 정렬에 사용된다.",
    )
    top_claim: str = Field(
        ...,
        description="가장 유사도가 높은 대표 청구항의 원문 텍스트.",
    )
    top_claim_no: int = Field(
        ...,
        description="대표 청구항 번호. 메타데이터의 claim_no 값.",
    )
    claims_found: int = Field(
        ...,
        description="이 특허에서 검색에 걸린 청구항 개수.",
    )
    claims: List[PatentClaimSnippet] = Field(
        ...,
        description=(
            "이 특허 안에서 유사도가 높은 상위 청구항 리스트. "
            "길이는 max_claims_per_patent 이하."
        ),
    )


# 5. 특허 검색 전체 출력 스키마 (tool_search_patent의 반환 타입)
class PatentSearchOutput(BaseModel):
    """
    특허 검색 툴의 최종 반환 결과.
    """
    query_text: str = Field(
        ...,
        description="검색에 실제 사용된 최종 질의 텍스트(키워드/정제된 문장).",
    )
    top_k: int = Field(
        ...,
        description="사용자에게 반환된 특허 개수.",
        ge=1,
    )
    results: List[PatentSearchResult] = Field(
        ...,
        description="score 내림차순으로 정렬된 특허 단위 검색 결과 목록.",
    )


class IPCCodeInput(BaseModel):
    codes: List[str] = Field(
        description="조회할 IPC 코드들의 리스트 (예: ['G06Q11/50', 'H04N01/20', 'A01B'])",
        min_items=1
    )

# IPC 검색관련 입출력 폼
class IPCDetailInfo(BaseModel):
    ids: str = Field(description="국제특허분류(IPC) 코드")
    description: str = Field(description="해당 IPC 코드의 상세 기술 설명 및 정의")
    type: str = Field(description="해당 IPC 코드의 계층 (예: s, c, u등) s는 section, c는 main class, u는 sub class, m은 main group, 이후 숫자들은 sub group")
    ancestors : str = Field(description="해당 코드의 상위 조상 코드들 (예:A > A01 > A01B > A01B1/00)")

class IPCSimpleInfo(BaseModel):
    ids: str = Field(description="국제특허분류(IPC) 코드")
    description: str = Field(description="해당 IPC 코드의 상세 기술 설명 및 정의")

class IPCKeywordInput(BaseModel):
    tech_texts: List[str] = Field(
        description="검색을 원하는 아이디어에 대해 핵심 독립적 기술단위로 분해한 영어 키워드들의 리스트 (예: ['Organic Light Emitting Display','Display Panel Opening Area','Pixel Electrode Contact Structure'])",
        min_items=1
    )
    top_k: int = Field(description="최종적으로 몇개의 main코드의 설명을 가져올지")

class IPCMainDescription(BaseModel):
    mains: List[IPCSimpleInfo] = Field(description = '입력된 키워드들에 대해 가장 가까운 의미를 나타내는 것으로 Vector DB에서 검색된 것들과 설명')
    subs: List[IPCSimpleInfo] = Field(description = "검색되어 나온 코드들 중 mains의 키워드들과 연관이 있는 키워드들에 대한 설명 ")