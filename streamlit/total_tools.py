from typing import List, Optional

from langchain_core.tools import tool
import os
import chromadb
import re
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from total_schemas import (
    IPCCodeInput,
    IPCDetailInfo,
    IPCKeywordInput,
    IPCMainDescription,
    PatentSearchInput,
    PatentClaimSnippet,
    PatentSearchResult,
    PatentSearchOutput,
    PatentByIdInput,       
    PatentClaimFull,       
    PatentByIdOutput,    
)

from ipc_func import get_ipc_detail_data_from_code, search_ipc_with_query
from doc_func import patent_hybrid_search

from dotenv import load_dotenv
import torch

# ---------------------------------------------------------
# 공용 리소스 초기화 (임베딩 모델, 벡터 DB 클라이언트 등)
# ---------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# IPC 벡터용 OpenAI 임베딩 함수
ipc_model = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",
)

# 특허(컴퓨터 비전) 문서용 SentenceTransformer 모델
doc_model = SentenceTransformer("dragonkue/BGE-m3-ko").to(device)

# IPC 코드용 벡터 DB
ipc_client = chromadb.PersistentClient(path="../db_search/ipc_db")
ipc_collection = ipc_client.get_collection(name="ipc_clean")

# 특허 청구항용 벡터 DB
doc_client = chromadb.PersistentClient(path="../db_search/doc_db")
doc_collection = doc_client.get_collection(name="patent_claims")


# ---------------------------------------------------------
# 1) 유사 특허 검색 툴
# ---------------------------------------------------------

# 특허 검색 파라미터 안전 범위
MAX_TOP_K = 30               # DB에서 가져올 최대 특허 수
MAX_CLAIMS_PER_PATENT = 5    # 특허당 최대 청구항 수
EXTRA_MARGIN = 10            # 검색 풀 크기를 조절하기 위한 여유분


# 재검색 안전장치 내부 helper 1
def _normalize_top_k(raw_top_k: int | None) -> int:
    """
    top_k가 0이거나 너무 크더라도 안전한 범위 [1, MAX_TOP_K]로 잘라서 반환합니다.
    """
    if raw_top_k is None:
        return 5
    try:
        value = int(raw_top_k)
    except (TypeError, ValueError):
        # 이상한 값이 들어오면 기본값
        return 5

    if value < 1:
        return 1
    if value > MAX_TOP_K:
        return MAX_TOP_K
    return value

# 재검색 안전장치 내부 helper 2
def _normalize_max_claims(raw_max_claims: int | None) -> int:
    """
    max_claims_per_patent를 [1, MAX_CLAIMS_PER_PATENT] 범위로 정규화합니다.
    """
    if raw_max_claims is None:
        return 3
    try:
        value = int(raw_max_claims)
    except (TypeError, ValueError):
        return 3

    if value < 1:
        return 1
    if value > MAX_CLAIMS_PER_PATENT:
        return MAX_CLAIMS_PER_PATENT
    return value


@tool(args_schema=PatentSearchInput)
def tool_search_patent_with_description(
    query_text: str,
    top_k: int = 5,
    max_claims_per_patent: int = 3,
    exclude_patent_ids: Optional[List[str]] = None,
) -> PatentSearchOutput:
    """
    컴퓨터 비전 관련 특허 벡터 DB에서 '유사 특허'를 검색하는 툴입니다.

    이 툴을 호출해야 하는 상황 (LLM용 가이드):
    - 사용자가
      - "이런 기술에 대한 비슷한 특허가 있는지 찾아줘"
      - "유사한 특허 상위 N개만 보여줘"
      - "이미 출원된 특허 중에서 내 아이디어와 비슷한 것 찾아줘"
      와 같이 **특허 검색 / 유사 특허 리스트**를 요청할 때 사용하세요.
    - 단순히 특허 개념 설명, IPC 설명, 절차 설명 등은 이 툴이 아니라
      일반 LLM 응답 또는 다른 IPC 툴을 사용해야 합니다.

    파라미터 설명:
    - query_text:
        유사 특허를 찾기 위한 핵심 기술 설명 또는 키워드.
        *이미 LLM이 추출/정제한 핵심 기술 문장*을 넣는 것을 권장합니다.
        (예: "사용자와의 거리 변화에 따라 자동으로 곡률이 바뀌는 디스플레이 장치")

    - top_k:
        최종적으로 사용자에게 보여줄 "특허 개수"입니다.
        사용자가 "상위 5개", "10개 정도"라고 말하면 그 값을 사용하고,
        언급이 없으면 기본값 5를 사용하세요.

    - max_claims_per_patent:
        각 특허별로 함께 보여줄 상위 청구항 개수입니다.
        너무 많으면 출력이 길어지므로 3 정도가 적당합니다.

    - exclude_patent_ids:
        이번 검색에서 제외해야 할 특허 ID 목록입니다.
        이전 턴에서 이미 보여준 특허를 다시 보여주지 않거나,
        사용자가 "2번/4번은 빼고 다시 찾아줘"라고 했을 때 활용합니다.
    """

    # None 방지
    exclude_patent_ids = exclude_patent_ids or []
    safe_top_k = _normalize_top_k(top_k)
    safe_max_claims = _normalize_max_claims(max_claims_per_patent)
    search_pool_size = safe_top_k + len(exclude_patent_ids) + EXTRA_MARGIN
    per_query_top_k = max(200, search_pool_size)
    final_top_k = max(200, search_pool_size)

    # 1) 쿼리 리스트 구성
    query_list = [query_text]

    # 2) hybrid search 함수 호출
    raw_results = patent_hybrid_search(
        collection=doc_collection,
        model=doc_model,
        query_list=query_list,
        per_query_top_k=per_query_top_k,
        final_top_k=final_top_k,
        top_k=top_k * 2,  # 먼저 넉넉히 가져와서 나중에 exclude + top_k 적용
        max_claims_per_patent=safe_max_claims,
        vector_weight=0.7,
        bm25_weight=0.3,
    )
    # raw_results: [{ "patent_id": ..., "score": ..., "title": ...,
    #                 "ipcNumber": ..., "claims_found": ..., "claims": [...] }, ...] < 수정 부분

    # 3) exclude_patent_ids 적용
    filtered = [
        r for r in raw_results
        if r.get("patent_id") not in exclude_patent_ids
    ]

    # 4) 상위 top_k만 사용
    filtered = filtered[:safe_top_k]

    # 5) raw dict → Pydantic 스키마로 매핑
    results: List[PatentSearchResult] = []

    for idx, item in enumerate(filtered):
        claim_snippets: List[PatentClaimSnippet] = []
        for c in item.get("claims", []):
            claim_snippets.append(
                PatentClaimSnippet(
                    claim_number=c.get("claim_number", ""),     # < 수정 부분
                    document=c.get("document", ""),
                    # title=c.get("title", "") or "",
                    distance=float(c.get("distance", 0.0)),
                    hybrid_score=float(c.get("hybrid_score", 0.0)),
                )
            )

        result_obj = PatentSearchResult( 
            patent_id=item.get("patent_id", ""),
            score=float(item.get("score", 0.0)),
            # top_claim=item.get("top_claim", ""),              < 수정 부분
            # top_claim_no=int(item.get("top_claim_no", 0)),    < 수정 부분
            title=item.get("title", ""),                        # 수정 부분
            ipcNumber=item.get("ipcNumber", ""),                # 수정 부분
            claims_found=int(item.get("claims_found", len(claim_snippets))),
            claims=claim_snippets,
            result_index=idx+1
        )
        results.append(result_obj)

    # 6) 최종 출력 스키마 구성
    output = PatentSearchOutput(
        query_text=query_text,
        top_k=len(results),
        results=results,
    )

    return output


# ---------------------------------------------------------
# 2) 출원번호로 특허 검색하기 위한 툴
# ---------------------------------------------------------

def normalize_korean_patent_id(patent_id: str) -> str:
    """
    한국 출원번호 입력을 DB에서 사용하는 형식으로 정규화합니다.

    지원 예시:
    - '1020050108060'          -> '1020050108060'
    - '10-2005-0108060'        -> '1020050108060'
    - '10 2005 0108060'        -> '1020050108060'
    - '10/2005/0108060'        -> '1020050108060'

    기본 규칙:
    1) 먼저 'NN-YYYY-NNNNNNN' 패턴을 우선적으로 인식해서 13자리로 맞추고,
    2) 그 외에는 숫자만 남기고, 13자리면 그대로 사용합니다.
    3) 그 외 길이/형식은 그대로 반환하거나, 필요하면 빈 문자열을 반환해
       "DB에 없다" 쪽으로 처리되게 할 수 있습니다.
    """
    if not patent_id:
        return ""

    s = patent_id.strip()

    # 1) 정형 패턴: 2자리 + 구분자 + 4자리 + 구분자 + 5~7자리
    #    예: '10-2005-0108060', '10 2005 108060', '10/2005/108060'
    m = re.match(r"^\s*(\d{2})\D+(\d{4})\D+(\d{5,7})\s*$", s)
    if m:
        kind = m.group(1)      # 10, 20, 30 등
        year = m.group(2)      # 2005
        serial = m.group(3)    # 0108060 또는 108060 같은 것
        # 일단 7자리로 zero-padding (선행 0이 빠졌을 가능성 고려)
        serial = serial.zfill(7)
        return f"{kind}{year}{serial}"

    # 2) 그 외에는 숫자만 남긴다
    digits = re.sub(r"\D", "", s)

    # 13자리면 이미 우리가 쓰는 형식이라고 보고 그대로 사용
    if len(digits) == 13:
        return digits

    # 그 외 길이는 애매하므로 그대로 돌려보내거나
    # 필요하면 추가 규칙(예: 11자리면 앞에 '10' 붙이기 등)을 추가할 수 있다.
    return digits

@tool(args_schema=PatentByIdInput)
def tool_search_detail_patent_by_id(
    patent_id: str,
    max_claims: int = 0,
) -> PatentByIdOutput:
    """
    출원번호(또는 patent_id)를 기반으로, 특허 청구항 벡터 DB에서
    해당 특허에 속한 청구항들과 주요 메타데이터를 **직접 조회**하는 툴입니다.

    이 툴을 호출해야 하는 상황 (LLM용 가이드):
    - 사용자가
      - "출원번호 1020230112930에 대해서 DB에서 자료 끌어와서 알려줘"
      - "이 출원번호 특허의 청구항들을 보여줘"
      - "위에서 말한 특허 1020...의 청구항 전체를 보고 싶어"
      - "이 출원번호 특허의 IPC 코드도 같이 알려줘"
      와 같이 **특정 출원번호 하나를 정확히 지정**하고,
      그 특허의 내용(특히 청구항 및 기본 메타정보)을 확인하고자 할 때 사용하세요.

    이 툴은 청구항 텍스트뿐 아니라 다음 메타데이터도 함께 반환합니다:
    - title: 발명의 명칭
    - priority: 우선권/출원 국가 정보 (예: "대한민국")
    - register: 공개/등록/거절 등의 상태 (예: "공개")
    - claim_no: 청구항의 번호
    - ipc: IPC 코드 리스트
    - link: 특허의 도면도 url

    LLM이 이 툴을 사용할 때의 활용 팁:
    - 사용자가 "IPC 코드도 알려줘", "어느 IPC 분야인지 설명해줘"라고 하면,
      반드시 ipc 값을 기반으로 설명하고,
      임의로 IPC 코드를 만들어내지 마세요.
    - 필요하다면 ipc_codes 값을 그대로 넘겨
      IPC 설명 도구(tool_search_ipc_description_from_code)를 연쇄 호출하여
      코드별 상세 설명과 계층 구조까지 함께 제공할 수 있습니다.

    주의 사항:
    - 이 DB는 '컴퓨터 비전/모빌리티' 등 특정 도메인에 한정된 서브셋일 수 있습니다.
      따라서, 출원번호가 실제로 존재하더라도, 이 벡터 DB 안에 없을 수 있습니다.
      그런 경우에는 found=False와 함께, KIPRIS/특허로 등 외부 서비스를 안내해야 합니다.
    """
    # 1) 입력 출원번호 정규화 (공백 제거 등)
    original_input = patent_id.strip()
    normalized_id = normalize_korean_patent_id(original_input)

    # 빈 문자열 방어
    if not normalized_id:
        return PatentByIdOutput(
            patent_id=original_input,
            found=False,
            title="",
            priority="",
            register="",
            ipc_raw="",
            ipc_codes=[],
            link="",
            num_claims=0,
            claims=[],
        )

    # 2) Chroma get() + where 필터로 메타데이터 기반 조회
    raw = doc_collection.get(
        where={"patent_id": normalized_id},
        include=["metadatas", "documents"],
    )

    ids = raw.get("ids", [])
    docs = raw.get("documents", [])
    metas = raw.get("metadatas", [])

    if not ids:
        # 이 DB 범위 안에 해당 출원번호가 없는 경우
        return PatentByIdOutput(
            patent_id=normalized_id,
            found=False,
            title="",
            priority="",
            register="",
            ipc_raw="",
            ipc_codes=[],
            link="",
            num_claims=0,
            claims=[],
        )

    # 3) 메타데이터에서 claim_no, title, priority, register, link, ipc 추출해서 정리
    claim_items = []

    title_candidates = []
    priority_candidates = []
    register_candidates = []
    link_candidates = []
    ipc_candidates = []

    for doc_text, meta in zip(docs, metas):
        # claim_no 파싱 (없거나 형식 이상하면 큰 숫자로 처리해서 뒤로 밀기)
        raw_claim_no = meta.get("claim_no", None)
        try:
            claim_no = int(raw_claim_no)
        except (TypeError, ValueError):
            claim_no = 999_999

        # 공통 메타데이터 후보 수집
        title_val = meta.get("title", "")
        if title_val:
            title_candidates.append(title_val)

        priority_val = meta.get("priority", "")
        if priority_val:
            priority_candidates.append(priority_val)

        register_val = meta.get("register", "")
        if register_val:
            register_candidates.append(register_val)

        link_val = meta.get("link", "")
        if link_val:
            link_candidates.append(link_val)

        ipc_val = meta.get("ipc", "")
        # if ipc_val:
        #     ipc_candidates.append(ipc_val)

        claim_items.append(
            {
                "claim_no": claim_no,
                "text": doc_text or "",
            }
        )

    # 4) 대표 메타데이터 선택 함수
    def pick_first_non_empty(values):
        for v in values:
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    title_value = pick_first_non_empty(title_candidates)
    priority_value = pick_first_non_empty(priority_candidates)
    register_value = pick_first_non_empty(register_candidates)
    link_value = pick_first_non_empty(link_candidates)
    # ipc_raw_value = pick_first_non_empty(ipc_candidates)

    # 5) IPC 코드 파싱 (쉼표 기준 분리 + 공백 정리)
    # ipc_codes_list: List[str] = []
    # if ipc_raw_value:
    #     # 혹시 세미콜론이 섞여 있어도 처리되도록 통일
    #     raw_for_split = ipc_raw_value.replace(";", ",")
    #     for part in raw_for_split.split(","):
    #         code = part.strip()
    #         if not code:
    #             continue
    #         # 여러 공백 정리 (예: "H04M   3/42")
    #         code = " ".join(code.split())
    #         ipc_codes_list.append(code)

    # 6) claim_no 기준으로 정렬
    claim_items_sorted = sorted(
        claim_items,
        key=lambda x: x["claim_no"],
    )

    # 7) max_claims 적용 (0이면 전체)
    if max_claims > 0:
        claim_items_sorted = claim_items_sorted[:max_claims]

    # 8) Pydantic 모델로 변환
    claim_models: List[PatentClaimFull] = [
        PatentClaimFull(
            claim_no=item["claim_no"],
            text=item["text"],
        )
        for item in claim_items_sorted
    ]

    return PatentByIdOutput(
        patent_id=normalized_id,
        found=True,
        title=title_value,
        priority=priority_value,
        register=register_value,
        # ipc_raw=ipc_raw_value,
        ipc_codes=ipc_val,
        link=link_value,
        num_claims=len(claim_models),
        claims=claim_models,
    )


# ---------------------------------------------------------
# 3) 기술 설명 → IPC 추천 툴
# ---------------------------------------------------------

@tool(args_schema=IPCKeywordInput)
def tool_search_ipc_code_with_description(
    tech_texts: List[str],
    top_k: int = 5,
) -> IPCMainDescription:
    """
    아이디어/기술 설명(또는 독립적인 기술 키워드 리스트)을 기반으로
    **어울리는 IPC 코드(주로 main 코드)들을 추천**하는 툴입니다.

    이 툴을 호출해야 하는 상황 (LLM용 가이드):
    - 사용자가
      - "이 기술에 맞는 IPC를 추천해줘"
      - "내 발명의 IPC 분류를 어떻게 잡는 게 좋을까?"
      - "이 컴퓨터 비전 아이디어는 어떤 IPC로 들어갈까?"
      와 같이 **기술 → IPC 추천**을 요청할 때 사용하세요.

    파라미터 설명:
    - tech_texts:
        검색하고 싶은 기술/아이디어를 **독립적인 기술 단위로 분해한 영어 키워드 리스트**입니다.
        예시:
          ["Organic Light Emitting Display",
           "Display Panel Opening Area",
           "Pixel Electrode Contact Structure"]

        한국어 원문이 들어왔다면, 먼저 LLM이 적절히 영어로 번역/분해한 뒤
        이 리스트에 넣어주는 식으로 사용하는 것을 권장합니다.

    - top_k:
        추천할 main IPC 코드 개수입니다.
        반환값 `IPCMainDescription` 안에서
        - mains: 메인 코드들 (top_k 개)
        - subs : mains 와 의미상 연관된 서브 코드들
        형태로 함께 제공됩니다.
    """
    result = search_ipc_with_query(
        ipc_model,
        ipc_collection,
        tech_texts,
        top_k,
    )
    # result는 {"mains": [...], "subs": [...]} 형태의 dict라고 가정
    return IPCMainDescription(**result)


# ---------------------------------------------------------
# 4) IPC 코드 → 상세 설명 툴
# ---------------------------------------------------------

@tool(args_schema=IPCCodeInput)
def tool_search_ipc_description_from_code(codes: List[str]) -> List[IPCDetailInfo]:
    """
    IPC 코드 리스트를 입력받아 각 코드에 대한 상세 설명과 계층 정보를 반환하는 툴입니다.

    이 툴을 호출해야 하는 상황 (LLM용 가이드):
    - 사용자가
      - "G06F, G06T가 각각 무엇을 의미하는지 설명해줘"
      - "이 IPC 코드들의 상위/하위 구조를 알고 싶어"
      - "A01B, A01B1/00 같은 코드가 어떤 기술을 다루는지 알려줘"
      처럼 **이미 특정 IPC 코드 문자열을 알고 있고**, 그 의미·정의·계층을
      자세히 알고 싶을 때 사용합니다.

    파라미터 설명:
    - codes:
        조회할 IPC 코드들의 리스트입니다.
        예: ["B03C1/00", "E02D7/00", "E02"]

        공백이 섞여 있을 수 있으므로, 함수 내부에서
        공백 제거 및 간단한 정규화를 수행합니다.
    """
    # 1) 코드 문자열 전처리: 공백 제거, 빈 문자열 제거
    cleaned_codes: List[str] = []
    for c in codes:
        if not c:
            continue
        normalized = c.strip().replace(" ", "")
        if normalized:
            cleaned_codes.append(normalized)

    if not cleaned_codes:
        # LLM이 잘못 호출한 경우에도 최소한 빈 리스트를 반환
        return []

    # 2) 기존 함수 호출 (벡터 DB 또는 메타 DB에서 상세 정보 조회)
    raw_results = get_ipc_detail_data_from_code(ipc_collection, cleaned_codes)

    # 3) 결과를 Pydantic 모델로 감싸서 반환
    parsed_results: List[IPCDetailInfo] = []
    for item in raw_results:
        parsed_results.append(IPCDetailInfo(**item))

    return parsed_results
