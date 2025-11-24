from typing import List, Optional
from pydantic import BaseModel, Field


# ==========================
# 1. 특허 검색 관련 스키마
# ==========================

class PatentSearchInput(BaseModel):
    """
    특허 벡터 DB에서 유사 특허를 검색할 때 사용하는 입력 스키마.

    - 이 모델은 특허 검색용 툴(tool_search_patent_with_description)의 입력으로 사용된다.
    - LLM이 사용자의 자연어 질문을 분석해서 이 필드들에 값을 채워 넣는 것을 기대한다.
    """
    query_text: str = Field(
        ...,
        description=(
            "유사 특허를 찾기 위한 핵심 기술 설명 또는 키워드. "
            "LLM이 추출한 키워드들의 조합이거나, 사용자의 원문 기술 설명을 "
            "간단히 정제한 1~3문장 정도의 문장일 수 있다. "
            "이 값이 임베딩되어 벡터 검색의 쿼리로 직접 사용되므로, "
            "불필요한 배경 설명보다는 발명의 차별화 포인트가 잘 드러나도록 작성하는 것이 좋다."
        ),
    )
    top_k: int = Field(
        5,
        description=(
            "최종적으로 사용자에게 보여줄 특허 개수. "
            "사용자가 '상위 3개', '10개 정도'처럼 개수를 명시한 경우 그 값을 사용하고, "
            "언급이 없다면 기본값 5를 사용한다. "
            "값이 너무 크더라도 도구 내부에서 최대 30개까지만 사용하도록 자동으로 제한한다."
        ),
        ge=1,
    )
    max_claims_per_patent: int = Field(
        3,
        description=(
            "각 특허별로 포함할 상위 청구항 개수. "
            "patent_hybrid_search 함수의 max_claims_per_patent 인자에 매핑된다. "
            "각 특허에서 가장 유사한 청구항 몇 개만 보고 싶을 때 조절하는 파라미터로, "
            "일반적으로 1~3개면 충분하다."
            "값이 너무 크더라도 도구 내부에서 최대 5개까지만 사용하도록 자동으로 제한한다."
        ),
        ge=1,
    )
    exclude_patent_ids: List[str] = Field(
        default_factory=list,
        description=(
            "이번 검색에서 제외해야 할 특허 ID(출원번호) 목록. "
            "예를 들어, 이전 턴에서 이미 사용자에게 보여준 특허를 "
            "다시 결과에 포함하지 않고 싶을 때 사용한다. "
            "비어 있는 리스트면 아무 특허도 제외하지 않는다."
        ),
    )


class PatentClaimSnippet(BaseModel):
    """
    특정 특허 안에서 검색에 걸린 개별 청구항 정보를 담는 스키마.

    - patent_hybrid_search 함수의 결과 중, 개별 청구항(claims) 레벨 정보를 표현한다.
    - 사용자는 이 정보를 통해 어떤 청구항 내용 때문에 해당 특허가 검색되었는지 이해할 수 있다.
    """
    id: str = Field(
        ...,
        description="벡터 DB 내에서 청구항 단위 문서를 구분하기 위한 내부 id.",
    )
    document: str = Field(
        ...,
        description="해당 청구항의 원문 텍스트 전체.",
    )
    title: str = Field(
        "",
        description="해당 청구항이 속한 특허의 발명의 명칭(존재하는 경우). 없으면 빈 문자열.",
    )
    distance: float = Field(
        ...,
        description=(
            "벡터 거리 값. 0에 가까울수록 쿼리와 더 유사하다. "
            "이 값은 내부 계산용이며, 사용자에게 직접 노출할 필요는 없다."
        ),
    )
    hybrid_score: float = Field(
        ...,
        description=(
            "벡터 유사도와 BM25 점수를 결합한 최종 하이브리드 점수. "
            "값이 클수록 쿼리와 의미적으로 더 가까운 청구항이다."
        ),
    )


class PatentSearchResult(BaseModel):
    """
    특허 단위로 집계된 검색 결과.

    - patent_hybrid_search가 반환하는 aggregated 리스트의 각 원소에 해당한다.
    - 여러 청구항을 종합한 점수로 특허 단위 랭킹을 만든다.
    """
    patent_id: str = Field(
        ...,
        description=(
            "특허 출원번호 메타데이터의 'patent_id'와 동일한 값. "
            "실제 출원번호이거나, 내부에서 사용하는 고유 식별자일 수 있다."
        ),
    )
    score: float = Field(
        ...,
        description=(
            "특허 단위 최종 하이브리드 점수. "
            "여러 청구항의 hybrid_score를 종합해서 계산되며, "
            "이 값 기준으로 상위 특허가 결정된다."
        ),
    )
    top_claim: str = Field(
        ...,
        description="가장 유사도가 높은 대표 청구항의 원문 텍스트.",
    )
    top_claim_no: int = Field(
        ...,
        description="대표 청구항 번호. 메타데이터의 claim_no 값과 동일하다.",
    )
    claims_found: int = Field(
        ...,
        description="이 특허에서 검색에 걸린 청구항 개수(전체 청구항 개수가 아니라, 후보군 중 매칭된 개수).",
    )
    claims: List[PatentClaimSnippet] = Field(
        ...,
        description=(
            "이 특허 안에서 유사도가 높은 상위 청구항 리스트. "
            "길이는 max_claims_per_patent 이하이며, "
            "대표 청구항(top_claim)도 이 리스트 안에 포함된다."
        ),
    )
    result_index: int = Field(
        ...,
        description=(
            "현재 응답에서 이 특허가 몇 번째 결과인지 나타내는 1부터 시작하는 번호. "
            "예를 들어 첫번째 결과면 1, 두 번째 결과면 2. "
            "벡터 DB 전체에서의 절대 등수가 아니라, "
            "이번 tool 호출에서 반환된 결과 리스트 안에서의 위치를 의미한다."
        ),
    )


class PatentSearchOutput(BaseModel):
    """
    특허 검색 툴(tool_search_patent_with_description)의 최종 반환 결과.

    - LLM 에이전트는 이 구조를 받아서 사용자에게 자연어로 재구성해 설명한다.
    """
    query_text: str = Field(
        ...,
        description="검색에 실제 사용된 최종 질의 텍스트(키워드 조합 또는 정제된 문장).",
    )
    top_k: int = Field(
        ...,
        description="사용자에게 반환된 특허 개수(실제 결과 리스트 길이).",
        ge=1,
    )
    results: List[PatentSearchResult] = Field(
        ...,
        description="score 내림차순으로 정렬된 특허 단위 검색 결과 목록.",
    )


# ==========================
# 2. IPC DB 조회 관련 스키마
# ==========================

class IPCCodeInput(BaseModel):
    """
    IPC 코드 자체를 기준으로 상세 정보를 조회할 때 사용하는 입력 스키마.

    - 예: 사용자가 'IPC G06F, G06T 설명해줘' 라고 했을 때,
      해당 코드들을 이 리스트에 넣어 IPC DB에서 상세 정의를 가져온다.
    """
    codes: List[str] = Field(
        ...,
        description=(
            "조회할 IPC 코드들의 리스트. "
            "예: ['G06Q11/50', 'H04N1/20', 'A01B'] 처럼, 섹션부터 서브그룹까지 "
            "사용 가능한 모든 레벨의 코드를 그대로 넣는다."
        ),
        min_items=1,
    )


class IPCDetailInfo(BaseModel):
    """
    특정 IPC 코드 1개에 대한 상세 정보를 표현하는 스키마.

    - IPC 계층 구조(섹션/메인클래스/서브클래스/그룹 등)와 설명을 함께 제공해서
      사용자가 코드의 의미와 상하위 관계를 이해할 수 있게 한다.
    """
    ids: str = Field(
        ...,
        description="국제특허분류(IPC) 코드 전체 문자열 (예: 'G06F', 'G06T7/00', 'A01B1/00').",
    )
    description: str = Field(
        ...,
        description="해당 IPC 코드의 공식 정의 또는 요약 설명. 가능한 한 간단한 한국어/영어 혼합 설명.",
    )
    type: str = Field(
        ...,
        description=(
            "해당 IPC 코드의 계층 레벨을 나타내는 식별자. "
            "예: 's' = section, 'c' = main class, 'u' = subclass, "
            "'m' = main group, 그 외 숫자/문자 조합은 subgroup 수준 등을 의미한다."
        ),
    )
    ancestors: str = Field(
        ...,
        description=(
            "해당 코드의 상위 조상 코드들을 '>'로 이어서 표현한 문자열. "
            "예: 'A > A01 > A01B > A01B1/00'. "
            "사용자가 코드가 어느 분야/세부 기술에 속하는지 한눈에 파악할 수 있도록 돕는다."
        ),
    )


class IPCSimpleInfo(BaseModel):
    """
    IPC 코드에 대한 간단한 정보만 필요할 때 사용하는 축약형 스키마.

    - IPCMainDescription.mains / subs 에서 공통적으로 사용된다.
    """
    ids: str = Field(
        ...,
        description="국제특허분류(IPC) 코드 전체 문자열.",
    )
    description: str = Field(
        ...,
        description="해당 IPC 코드의 간단한 설명 또는 주요 키워드 요약.",
    )


class IPCKeywordInput(BaseModel):
    """
    기술 키워드를 기반으로 IPC 후보 코드를 추천받을 때 사용하는 입력 스키마.

    - tech_texts 는 이미 LLM이 '핵심 독립 기술 단위'로 분해한 영어 키워드/구문 목록이라고 가정한다.
      (예: ['Organic Light Emitting Display', 'Display Panel Opening Area', ...])
    - 이 키워드들을 벡터 DB에 질의해서, 의미적으로 가까운 IPC main 코드들을 찾는다.
    """
    tech_texts: List[str] = Field(
        ...,
        description=(
            "IPC 추천을 원하는 아이디어에 대해, 핵심 독립 기술 단위로 분해한 영어 키워드/구문 리스트. "
            "각 원소는 하나의 기술 포인트를 설명하는 짧은 문장 또는 구문이어야 한다. "
            "예: ['Organic Light Emitting Display', "
            "'Display Panel Opening Area', "
            "'Pixel Electrode Contact Structure']"
        ),
        min_items=1,
    )
    top_k: int = Field(
        5,
        description=(
            "벡터 DB에서 가장 유사한 IPC main 코드들을 몇 개까지 가져올지 결정하는 값. "
            "일반적으로 3~10 정도면 충분하다."
        ),
        ge=1,
    )


class IPCMainDescription(BaseModel):
    """
    IPC 키워드 검색 결과를 정리한 스키마.

    - mains: 입력된 tech_texts 와 가장 의미가 가까운 IPC main 코드들.
    - subs : mains 와 연관된 하위(sub) 코드들 중, 함께 보면 좋은 후보들.
    """
    mains: List[IPCSimpleInfo] = Field(
        ...,
        description=(
            "입력된 키워드들에 대해, 벡터 DB에서 직접 검색된 IPC main 코드들과 그 간단한 설명. "
            "사용자에게 '이 발명은 대략 어떤 IPC 대분류에 속하는지'를 안내하는 역할을 한다."
        ),
    )
    subs: List[IPCSimpleInfo] = Field(
        ...,
        description=(
            "mains에서 선택된 main 코드들과 의미적으로 연관성이 높은 하위 IPC 코드들에 대한 정보. "
            "세부 분류까지 보고 싶을 때 참고용으로 함께 제시된다."
        ),
    )


# ======================================
# 3. 출원번호 기준으로 db 접근하기 위한 스키마
# ======================================


class PatentByIdInput(BaseModel):
    """
    출원번호(또는 patent_id)를 기반으로
    해당 특허의 청구항들을 직접 조회할 때 사용하는 입력 스키마.
    """
    patent_id: str = Field(
        ...,
        description=(
            "조회할 특허의 출원번호 또는 patent_id. "
            "예: '1020230112930'. "
            "벡터 검색이 아니라, 이 ID와 일치하는 문서들의 메타데이터를 "
            "특허 청구항 벡터 DB에서 직접 조회할 때 사용한다."
        ),
    )
    max_claims: int = Field(
        0,
        description=(
            "가져올 청구항 개수 상한. "
            "0인 경우에는 해당 특허의 모든 청구항을 가져온다. "
            "너무 많은 청구항이 있을 수 있으므로, "
            "간단히 확인할 경우에는 5~20 정도로 제한해서 사용할 수 있다."
        ),
        ge=0,
        le=200,
    )


class PatentClaimFull(BaseModel):
    """
    출원번호로 조회했을 때, 개별 청구항의 전체 정보를 담는 스키마.
    """
    claim_no: int = Field(
        ...,
        description="청구항 번호 (메타데이터의 claim_no). 숫자로 정렬하여 출력한다.",
    )
    text: str = Field(
        ...,
        description="해당 청구항의 원문 텍스트 전체.",
    )


class PatentByIdOutput(BaseModel):
    """
    출원번호 기반 조회 툴의 최종 반환 결과.
    """
    patent_id: str = Field(
        ...,
        description="조회에 사용된 출원번호 또는 patent_id.",
    )
    found: bool = Field(
        ...,
        description=(
            "True 이면 DB에서 해당 출원번호를 가진 청구항들이 하나 이상 발견되었음을 의미한다. "
            "False 이면 현재 특허 벡터 DB 범위 안에 이 출원번호가 없다는 뜻이다."
        ),
    )
    title: str = Field(
        "",
        description=(
            "해당 특허의 발명의 명칭. "
            "여러 청구항 메타데이터 중 첫 번째로 발견된 title 을 대표값으로 사용한다. "
            "메타데이터에 title 정보가 없다면 빈 문자열일 수 있다."
        ),
    )
    num_claims: int = Field(
        ...,
        description="DB에서 조회된 청구항 개수.",
    )
    claims: List[PatentClaimFull] = Field(
        default_factory=list,
        description=(
            "청구항 번호(claim_no) 오름차순으로 정렬된 청구항 리스트. "
            "max_claims 가 0이면 전체, 0보다 크면 그 개수만큼 앞에서 잘라서 반환한다."
        ),
    )