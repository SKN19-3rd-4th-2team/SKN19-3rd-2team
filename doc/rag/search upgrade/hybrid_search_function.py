from collections import defaultdict
from rank_bm25 import BM25Okapi


def hybrid_search(multi_query_results, query_list, top_k=30, vector_weight=0.7, bm25_weight=0.3):
    # ========================================
    # 1. multi_query_results 파싱
    # ========================================
    ids = multi_query_results["ids"][0]
    docs = multi_query_results["documents"][0]
    metas = multi_query_results["metadatas"][0]
    distances = multi_query_results["distances"][0]
    
    # ========================================
    # 2. BM25 초기화 (필터링된 문서들로만)
    # ========================================
    tokenized_docs = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    
    # 쿼리 토큰화 (여러 쿼리를 하나로 합침)
    if isinstance(query_list, list):
        combined_query = " ".join(query_list)
    else:
        combined_query = query_list
    tokenized_query = combined_query.split()
    
    # BM25 점수 계산
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # ========================================
    # 3. 하이브리드 점수 계산 및 그룹화
    # ========================================
    grouped = defaultdict(list)
    
    for i in range(len(ids)):
        # 벡터 유사도 점수
        vector_score = 1 - distances[i]
        
        # BM25 점수 (정규화)
        bm25_score = bm25_scores[i]
        
        # 하이브리드 점수
        hybrid_score = vector_weight * vector_score + bm25_weight * bm25_score
        
        claim_data = {
            "id": ids[i],
            "document": docs[i],
            "metadata": metas[i],
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
        # 하이브리드 점수 사용
        hybrid_scores = [c["hybrid_score"] for c in claims]
        scores_sorted = sorted(hybrid_scores, reverse=True)
        
        # 상위 3개 평균
        top3 = scores_sorted[:3]
        top3_avg = sum(top3) / len(top3)
        
        # 최대 점수
        max_score = scores_sorted[0]
        
        # 청구항 개수 보너스 (청구항이 많을수록 관련성이 높다고 가정)
        claim_count = len(claims)
        count_bonus = min(1.0, claim_count / 10.0)
        
        # 최종 집계 점수
        final_score = top3_avg * 0.6 + max_score * 0.3 + count_bonus * 0.1
        
        return final_score
    
    # ========================================
    # 5. 최종 특허 단위 결과 생성
    # ========================================
    aggregated = []
    for patent_id, claims in grouped.items():
        score = compute_patent_score(claims)
        
        # 대표 청구항 선택 (가장 높은 hybrid_score)
        rep_claim = sorted(claims, key=lambda x: x["hybrid_score"], reverse=True)[0]
        
        # 필수 정보만 포함하도록 청구항 필터링
        filtered_claims = [
            {
                "id": c["id"],
                "document": c["document"],
                "distance": c["distance"],
                "hybrid_score": c["hybrid_score"]
            }
            for c in claims
        ]
        
        aggregated.append({
            "patent_id": patent_id,
            "score": score,
            "top_claim": rep_claim["document"],
            "top_claim_no": rep_claim["metadata"]["claim_no"],
            "claims_found": len(claims),
            "claims": filtered_claims
        })
    
    # ========================================
    # 6. 점수로 정렬 후 상위 top_k개 반환
    # ========================================
    aggregated = sorted(aggregated, key=lambda x: x["score"], reverse=True)
    
    return aggregated[:top_k]
