from collections import defaultdict
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

def patent_hybrid_search(
    collection,
    model,
    query_list,
    per_query_top_k=200,
    final_top_k=200,
    top_k=30,
    max_claims_per_patent=3,
    vector_weight=0.7,
    bm25_weight=0.3
):
    
    # ========================================
    # 1. Multi-query rerank
    # ========================================
    
    # 쿼리 리스트 정규화
    if isinstance(query_list, str):
        query_list = [query_list]
    
    # 쿼리 임베딩
    query_embs = model.encode(query_list).tolist()
    
    # 단일 쿼리인 경우 바로 검색
    if len(query_list) == 1:
        results = collection.query(query_embeddings=query_embs, n_results=per_query_top_k)
    else:
        # 다중 쿼리 rerank
        candidates = []
        for emb in query_embs:
            r = collection.query(
                query_embeddings=[emb],
                n_results=per_query_top_k
            )
            
            distances = np.array(r["distances"][0])
            mean = distances.mean()
            std = distances.std() + 1e-9
            z_scores = (distances - mean) / std
            
            ids = r["ids"][0]
            docs = r["documents"][0]
            distances = r["distances"][0]
            metas = r["metadatas"][0]
            
            for pid, doc, meta, z, dist in zip(ids, docs, metas, z_scores, distances):
                candidates.append({
                    "id": pid,
                    "document": doc,
                    "metadatas": meta,
                    "distance": dist,
                    "z-score": z
                })
        
        # z-score 기준 정렬 후 상위 선택
        top_candidates = sorted(candidates, key=lambda x: x["z-score"])[:final_top_k]
        
        # collection.query 형식으로 재구성
        results = {
            "ids": [[c["id"] for c in top_candidates]],
            "documents": [[c["document"] for c in top_candidates]],
            "distances": [[c["distance"] for c in top_candidates]],
            "metadatas": [[c["metadatas"] for c in top_candidates]]
        }
    
    # ========================================
    # 2. Hybrid search (Vector + BM25)
    # ========================================
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    
    # BM25 초기화
    tokenized_docs = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    
    # 쿼리 토큰화
    combined_query = " ".join(query_list)
    tokenized_query = combined_query.split()
    
    # BM25 점수 계산
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # ========================================
    # 3. 하이브리드 점수 계산 및 그룹화
    # ========================================
    grouped = defaultdict(list)
    
    for i in range(len(ids)):
        vector_score = 1 - distances[i]
        bm25_score = bm25_scores[i]
        hybrid_score = vector_weight * vector_score + bm25_weight * bm25_score
        
        claim_data = {
            "id": ids[i],
            "document": docs[i],
            "metadata": metas[i],  # 전체 메타데이터 포함
            "distance": distances[i],
            "vector_score": vector_score,
            "bm25_score": bm25_score,
            "hybrid_score": hybrid_score
        }
        patent_id = metas[i]["patent_id"]
        grouped[patent_id].append(claim_data)
    
    # ========================================
    # 4. 특허 단위 점수 계산
    # ========================================
    def compute_patent_score(claims):
        hybrid_scores = [c["hybrid_score"] for c in claims]
        scores_sorted = sorted(hybrid_scores, reverse=True)
        
        top3 = scores_sorted[:3]
        top3_avg = sum(top3) / len(top3)
        max_score = scores_sorted[0]
        
        claim_count = len(claims)
        count_bonus = min(1.0, claim_count / 10.0)
        
        final_score = top3_avg * 0.6 + max_score * 0.3 + count_bonus * 0.1
        return final_score
    
    # ========================================
    # 5. 최종 결과 생성 (청구항 최대 3개)
    # ========================================
    aggregated = []
    for patent_id, claims in grouped.items():
        score = compute_patent_score(claims)
        
        # hybrid_score 기준 정렬
        claims_sorted = sorted(claims, key=lambda x: x["hybrid_score"], reverse=True)
        
        # 대표 청구항
        rep_claim = claims_sorted[0]
        
        # 상위 N개 청구항만 선택 (title만 포함)
        top_claims = [
            {
                "id": c["id"],
                "document": c["document"],
                "title": c["metadata"].get("title", ""),  # title만 추출
                "distance": c["distance"],
                "hybrid_score": c["hybrid_score"]
            }
            for c in claims_sorted[:max_claims_per_patent]
        ]
        
        aggregated.append({
            "patent_id": patent_id,
            "score": score,
            "top_claim": rep_claim["document"],
            "top_claim_no": rep_claim["metadata"]["claim_no"],
            "claims_found": len(claims),
            "claims": top_claims  # 최대 max_claims_per_patent개
        })
    
    # ========================================
    # 6. 점수로 정렬 후 상위 top_k개 반환
    # ========================================
    aggregated = sorted(aggregated, key=lambda x: x["score"], reverse=True)
    
    return aggregated[:top_k]

