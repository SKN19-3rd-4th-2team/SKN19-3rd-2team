from typing import List, Optional, TypedDict
from langchain_core.tools import tool
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from total_state import IPCCodeInput,IPCDetailInfo,IPCKeywordInput,IPCMainDescription,PatentSearchInput,PatentClaimSnippet,PatentSearchResult,PatentSearchOutput
from ipc_func import get_ipc_detail_data_from_code,search_ipc_with_query
from doc_func import patent_hybrid_search

from dotenv import load_dotenv
import torch

device = "cuda" if torch.cuda.is_available() else 'cpu'

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ipc_model = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",
)

doc_model = SentenceTransformer("dragonkue/BGE-m3-ko").to(device)

ipc_client = chromadb.PersistentClient(path="./ipc_db")
ipc_collection = ipc_client.get_collection(name="ipc_clean")

doc_client = chromadb.PersistentClient(path="./doc_db")
doc_collection = doc_client.get_collection(name="patent_claims")


@tool(args_schema=PatentSearchInput)
def tool_search_patent(
    query_text: str,
    top_k: int = 5,
    max_claims_per_patent: int = 3,
    exclude_patent_ids: Optional[List[str]] = None,
) -> PatentSearchOutput:
    """
    컴퓨터 비전 특허 벡터 DB에서 '유사 특허'를 검색하는 툴.

    - query_text: 유사 특허를 찾기 위한 핵심 기술 설명 또는 키워드
    - top_k: 최종적으로 돌려줄 특허 개수
    - max_claims_per_patent: 각 특허당 포함할 상위 청구항 개수
    - exclude_patent_ids: 이번 검색에서 제외해야 할 특허 ID 목록
    """
    # None 방지
    exclude_patent_ids = exclude_patent_ids or []

    # 1) 쿼리 리스트 구성
    #    지금은 단일 쿼리만 사용하지만, 나중에 멀티쿼리(키워드 여러 개)로 확장 가능
    query_list = [query_text]

    # 2) hybrid search 함수 호출
    raw_results = patent_hybrid_search(
        collection=doc_collection,
        model=doc_model,
        query_list=query_list,
        per_query_top_k=200,
        final_top_k=200,
        top_k=top_k * 2,              # 먼저 넉넉히 가져와서 나중에 exclude + top_k 적용
        max_claims_per_patent=max_claims_per_patent,
        vector_weight=0.7,
        bm25_weight=0.3,
    )
    # raw_results: [{ "patent_id": ..., "score": ..., "top_claim": ..., "top_claim_no": ..., "claims_found": ..., "claims": [...] }, ...]

    # 3) exclude_patent_ids 적용
    filtered = [
        r for r in raw_results
        if r.get("patent_id") not in exclude_patent_ids
    ]

    # 4) 상위 top_k만 사용
    filtered = filtered[:top_k]

    # 5) raw dict → Pydantic 스키마로 매핑
    results: List[PatentSearchResult] = []

    for item in filtered:
        claim_snippets: List[PatentClaimSnippet] = []
        for c in item.get("claims", []):
            claim_snippets.append(
                PatentClaimSnippet(
                    id=c.get("id", ""),
                    document=c.get("document", ""),
                    title=c.get("title", "") or "",
                    distance=float(c.get("distance", 0.0)),
                    hybrid_score=float(c.get("hybrid_score", 0.0)),
                )
            )

        result_obj = PatentSearchResult(
            patent_id=item.get("patent_id", ""),
            score=float(item.get("score", 0.0)),
            top_claim=item.get("top_claim", ""),
            top_claim_no=int(item.get("top_claim_no", 0)),
            claims_found=int(item.get("claims_found", len(claim_snippets))),
            claims=claim_snippets,
        )
        results.append(result_obj)

    # 6) 최종 출력 스키마 구성
    output = PatentSearchOutput(
        query_text=query_text,
        top_k=len(results),
        results=results,
    )

    return output


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
    result = search_ipc_with_query(ipc_model, ipc_collection,tech_texts,top_k)
    return IPCMainDescription(**result)




# 4) IPC 코드 설명 툴
@tool(args_schema=IPCCodeInput)
def tool_search_ipc_description_from_code(codes: List[str]) -> List[IPCDetailInfo]:
    """
    IPC 코드 리스트를 입력받아 각 코드에 대한 상세 설명명과 코드의 계층, 해당 코드들의 상위 코드들에 대한 결과를 반환합니다.
    IPC 코드 리스트는 반드시 ['B03C1/00','E02D7/00','E02']의 형태처럼 공백이 포함되지 않은 형태의 표준표기형태로 입력해야 합니다.
    사용자가 특정 분류 코드의 의미를 물어보거나, IPC 코드의 전반적인 정보를 파악해야 하거나, 코드들 사이의 상위 관계나 계층을 파악해야 할 때 사용하세요.
    """
    
    # 1. 기존 함수 호출
    raw_results = get_ipc_detail_data_from_code(ipc_collection,codes)
    
    # 2. 결과 검증 및 Pydantic 객체로 변환
    # (딕셔너리 리스트를 Pydantic 객체 리스트로 변환하여 LLM에게 전달)
    parsed_results = []
    for item in raw_results:
        # Pydantic 모델을 이용해 데이터 검증 및 포장
        parsed_results.append(IPCDetailInfo(**item))
    return parsed_results