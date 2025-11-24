# 📘 AI 기반 특허 질의응답 시스템

---

# 1. 팀 소개

<div align="center">

### **FantAstIc 5**
 
| 신지섭 | 왕혁준 | 임상민 | 장효정 | 박도연 |
| :---: | :---: | :---: | :---: | :---: |
| <img width="228" height="291" alt="image" src="https://github.com/user-attachments/assets/ce62a970-9d0c-4b1f-b716-b7a07311fcfa" /> | <img width="227" height="283" alt="image" src="https://github.com/user-attachments/assets/ccada052-5085-41eb-a3b2-33c1d7cbb46a" /> | <img width="222" height="263" alt="image" src="https://github.com/user-attachments/assets/e54ce89b-970b-4694-aa4e-478677b18764" /> | <img width="214" height="277" alt="image" src="https://github.com/user-attachments/assets/762b7bed-ec73-4fcc-94b4-1f1d54d8263d" /> | <img width="219" height="261" alt="image" src="https://github.com/user-attachments/assets/c5d8287f-1c5d-4622-9e61-fd6d7e803320" /> |
| [![GitHub](https://img.shields.io/badge/GitHub-Melonmacaron-181717?style=flat&logo=github&logoColor=white)](https://github.com/Melonmacaron) | [![GitHub](https://img.shields.io/badge/GitHub-vibevibe26-181717?style=flat&logo=github&logoColor=white)](https://github.com/vibevibe26) | [![GitHub](https://img.shields.io/badge/GitHub-colaa222-181717?style=flat&logo=github&logoColor=white)](https://github.com/colaa222) | [![GitHub](https://img.shields.io/badge/GitHub-HyojungJ-181717?style=flat&logo=github&logoColor=white)](https://github.com/HyojungJ) | [![GitHub](https://img.shields.io/badge/GitHub-doyeon999-181717?style=flat&logo=github&logoColor=white)](https://github.com/doyeon999) |

**프로젝트 기간**: 2025.11.24 ~ 2024.11.25 (2일)

</div>

---

# 2. 프로젝트 개요

## 2.1 프로젝트 명
**특허 검색 및 질의응답 시스템**

## 2.2 프로젝트 소개
사용자가 입력한 자연어 Query에 대해
LLM이 상황에 맞는 Tool을 자동으로 선택하여
- **IPC 코드 검색**
- **컴퓨터비전 분야 특허 검색**
을 수행한 뒤,
검색된 내용을 기반으로 LLM이 최종 응답을 생성하는 **RAG 기반 특허 Q&A 챗봇**

## 2.3 프로젝트 필요성
### 청구항이란?
- 특허출원의 가장 핵심적인 부분으로서, 출원인이 **보호받고자 하는 사항**, 즉 독점권을 요구하는 **권리범위**를 기재한 것 (법 42조4항 본문)
- 청구범위에 기재되지 아니한 발명은 특허권으로서 보호받을 수 없다!
- 청구항 예시
 <img width="1171" height="637" alt="image" src="https://github.com/user-attachments/assets/fbc8f2d2-db0e-4dcb-88cb-54064aa51f0b" />

### IPC 코드란?
- 특허출원서, 실용신안 등의 특허문헌을 위한 공통 분류 규정
 <img width="503" height="368" alt="image" src="https://github.com/user-attachments/assets/e628d294-9b21-496b-bea4-4c1008f2a4fe" />

### 문제점
**1) 관련 기사 1**

<img width="1193" height="300" alt="image" src="https://github.com/user-attachments/assets/01a10fac-f137-4984-a757-5f0b05299de1" />

**2) 관련 기사 2**

<img width="1332" height="243" alt="image" src="https://github.com/user-attachments/assets/ed10205f-8d30-4f04-a764-3312bb8c7130" />


## 2.4 프로젝트 목표
- 의미 기반 검색이 가능한 임베딩 기반 특허 검색 시스템 구축
- 검색된 출원번호·청구항 기반으로 구조화된 답변 생성
- 대규모 데이터에서 빠르고 정확한 **유사도 기반** 검색을 지원


## 2.5 범위 선정
- 중소기업이 최근 특허를 많이 내고 있는 분야: **SW개발업**
  
  <img width="600" height="371" alt="image" src="https://github.com/user-attachments/assets/653f9d6c-1dde-4ad3-8825-9e513e6bdc99" />

- 그 업계에서 인공지능 관련 가장 많은 산업: **시각지능**

  <img width="600" height="197" alt="image" src="https://github.com/user-attachments/assets/a374b167-52c0-4e51-92c7-5b92fcba931a" />

- 그 핵심 기술인 **컴퓨터 비전** 관련 특허

---

# 3. 기술 스택 & 사용 모델

## 기술 스택
- Python 3.12
- LangChain / LangGraph
- ChromaDB
- GitHub
- Streamlit

## 모델
- **임베딩 모델**: IPC코드 DB: `text-embedding-3-small`, 청구항 DB: `dragonkue/BGE-m3-ko`
- **LLM**: gpt-5.1

## 폴더 구성
(구조)

---

# 4. 시스템 아키텍처
<img width="2325" height="967" alt="image" src="https://github.com/user-attachments/assets/7224c7f8-7eb1-4f5d-8225-48bb3e31665b" />


---

# 5. WBS
- (사진)

---

# 6. 요구사항 명세

## 기능 요구사항
| ID | 구분 | 내용 | 우선순위 | 비고 |
| --- | --- | --- | --- | --- |
| FR-01 | 입력 처리 | 사용자가 자연어 Query를 입력하면 시스템은 이를 LLM에 전달해야 한다 | 상 |  |
| FR-02 | Tool 선택 | LLM은 Query를 분석하여 **적절한 Tool(DB 검색 모듈)을 자동 선택**해야 한다 | 상 | LangChain Agent 기반 |
| FR-03 | DB 검색 | Tool은 DB에서 **출원번호·청구항 또는 IPC 코드**를 정확하게 검색해야 한다 | 상 | ChromaDB / IPC DB |
| FR-04 | LLM 답변 생성 | LLM은 검색 결과를 사용해 **근거 기반 설명형 답변**을 생성해야 한다 | 상 | RAG 기반 생성 |
| FR-05 | 다양한 검색 유형 | 사용자는 **기술명 / IPC / 키워드 / 문장** 등 다양한 형태로 검색이 가능해야 한다 | 중 |  |


## 비기능 요구사항
| ID | 구분 | 내용 | 목표치 | 비고 |
| --- | --- | --- | --- | --- |
| NFR-01 | 성능 | 사용자 요청 후 **전체 응답 시간은 10초 이내**여야 한다 | ≤ 10초 | 평균 기준 |
| NFR-02 | 가용성 | Tool 호출 및 DB 검색이 안정적으로 수행되어야 한다 |  |  |
| NFR-03 | 정확성 | LLM은 검색된 결과 외의 근거를 생성하면 안 된다 | Must | hallucination 방지 |

---

# 7. 수집한 데이터 및 전처리 요약

### 수집 규모
- **1. IPC 코드**: 약 70,000건
- **필드 구성**

  <img width="754" height="264" alt="image" src="https://github.com/user-attachments/assets/173823ac-99c7-4e3d-aa8b-33a6b143602c" />

  - IPC 코드(`ipc_code`)
  - IPC 영문 설명(`title_en`) -> 임베딩
  - 코드의 레벨 구분(`kind`)
  - 부모 IPC 코드(`parent`)
  - 최상위부터 현재 코드까지 전체 계층('path')
  
- **2. 공개 특허 문서**: 약 30,000건
- **청구항 개수:** 약 590,000건
- **필드 구성**

<img width="908" height="563" alt="image" src="https://github.com/user-attachments/assets/45e74890-1027-4fdd-9b0d-25a9b81f8781" />

  - 출원번호(`id`)
  - 발명의 명칭(`name`)
  - 요약(`abstract`)
  - 청구항(`claim`) -> 임베딩
  - IPC 코드(`IpcNumber`) 등


### 전처리
- **IPC 코드**
  - xml파일 파싱, 필요 정보 json 변환
  - 리스트 형태의 path를 `A > A01 > A01B` 형식의 문자열로 변경
  - 임베딩 생성 후 ChromaDB 저장
    
- **특허 문서**
  - xml파일 파싱, 필요 정보 json 변환
  - 청구항 분리
  - 임베딩 생성 후 ChromaDB 저장

---

# 8. DB 연동 구현 코드 (링크)
- (github 정리 후 링크 첨부)

- ChromaDB 구축 코드
https://github.com/____/PatentPilot/tree/main/db

- Langchain tools 코드

- DB 검색 코드

---

# 9. DB 검색 최적화 개선 노력
## 1. 청구항 DB 검색 최적화
### 1.1 multy-query
**문제점**
- results 200개를 검색해올 때, 쿼리가 여러 개면 검색 결과가 배수로 늘어나는 것을 처리하기 위해 distance 기준으로 200개만 남김 → 여러 쿼리가 모두 중요한 키워드인데 한 쿼리와 강하게 유사한 청구항이 많다면, 그 쿼리에 대한 유사도 높은 청구항만 살아남는 문제

**해결 방안**
- 각 쿼리의 검색결과로 나온 distance에 대해 평균, 표준편차를 구해서 z 정규화, 그 z-score로 200개만 남김

**해결 결과**

<img width="806" height="129" alt="image" src="https://github.com/user-attachments/assets/3e9bb923-7db3-4aa0-ad16-704f77223b6f" />

- 200개의 ids의 교집합을 비교해봤을 때, 약 25~50% 정도의 다른 청구항이 상위로 올라오며 편향 완화됨(랭킹 다양성 증가)


### 1.2 re-ranking
**문제점**
- 청구항이 아닌 출원번호로 재정렬 필요. 만약 z-score만 참조해 정렬한다면 우연히 유사도가 높은 청구항이 단 한 개가 나왔어도 해당 출원번호가 상위에 랭크되는 문재

**해결 방안**
- 출원번호로 그룹화 후 ‘검색된 청구항 중 상위 세 개의 유사도 평균’, ‘제일 높은 청구항의 유사도’, ‘검색된 청구항의 개수’를 이용해 score 계산
- BM25 기반 키워드 검색 추가 → 하이브리드 검색 방식 적용

**평가 지표**
- 유사 특허의 기준을 `같은 IPC 코드 공유`로 잡고, 특정 IPC 코드를 갖고 있는 특허의 출원번호를 저장해서  re-ranking했을 때 나오는 출원번호와 비교< 수정 필요

**해결 결과**
- BM25가 문자열 매칭을 보완하여 전체 검색 정확도와 신뢰도를 향상
- 그냥 z-score 기준 상위 TOP_K 출원번호와 비교했을 때, 30개 중 20개만 일치하는 등의 랭킹 다양성 보임
- (눈에 보이는 사진이 있을까?)
 
## 2. IPC DB 검색 최적화
### 2.1 계층 구조 관한 문제
**문제점**
- IPC코드-설명을 각 DB에 저장한 결과, IPC 하위코드가 상위 의미를 포함하지 않아 의미 단절됨
 - 예시)

  <img width="752" height="134" alt="image" src="https://github.com/user-attachments/assets/12451d86-7b36-49e4-9c0e-9a907df61fee" />

**해결 방안**
- 하위코드 텍스트에 상위코드 내용을 결합해 의미 단절 해소
 - 예시)

   <img width="894" height="117" alt="image" src="https://github.com/user-attachments/assets/bacdd066-9311-4028-ae10-84fb58e79dd9" />

### 2.2 오염 문제
**문제점**
- 앞선 계층 구조화로 인해 DB에 임베딩 되는 내용에 중복되는 내용이 다수 존재 → 상위코드의 유사도가 높으면, 그 하위코드의 유사도도 높게 나오는 문제
<img width="1805" height="582" alt="image" src="https://github.com/user-attachments/assets/82a94242-ad76-4111-a37d-0679b2b5e109" />

**해결 방안**
- 상위-하위가 모두 검색되었을 때에는 좀 더 자세한 설명을 포함하는 하위 우선하되, 부모의 정보를 함께 전달(Main을 하위 코드로 Sub를 조상 코드들로 구성해서 전달)
- 이때 임계 비율이상 쿼리 간 거리가 하위코드보다 상위코드가 더 가깝다면, Main에 부모만 전달하도록 함
- ??
- 유사한 의미의 코드가 없을 경우 상관없는 코드를 제공하는 것을 대비해 평균 컷오프 거리를 구해서 1.4이상의 거리를 가지는 코드는 컷오프 하도록 로직 수정

**해결 결과**
<img width="1803" height="279" alt="image" src="https://github.com/user-attachments/assets/5e10834d-26cc-4b55-9dbf-b76760798dbe" />


---

# 10. 챗봇 테스트 & 개선 노력

## **1) 출원번호 단건 조회가 되지 않던 문제**

### 개선 전 질문 + 답변

```
[질문]
1020230019980 이 출원번호에 대해서 대표 청구항들을 요약해서 알려줄래?

[답변]
현재 제가 접근할 수 있는 특허 DB는 “컴퓨터 비전·모빌리티” 중심의 벡터 검색용 DB라서,  
질문 주신 출원번호 **1020230019980**을 “출원번호로 직접 조회”하는 기능은 없습니다.

방금 수행한 검색은 `1020230019980`이라는 문자열을 “쿼리 텍스트”로 유사도 검색한 것이어서,  
반환된 특허들은 모두 **전혀 다른 분야(수소공정, 렌티바이러스, OLED 등)**에 관한 것이고,  
실제 출원번호 1020230019980의 내용과는 무관한 결과입니다.

따라서,

- 1020230019980의 **정확한 발명의 명칭, 요약, 청구항, IPC 분류** 등은  
  이 도구만으로는 확인이 어렵고,
- KIPRIS, 특허로, WIPO Patentscope 등 공인 특허 검색 사이트에서  
  출원번호로 직접 조회하시는 것이 필요합니다.
```

### 원인 분석

- 처음에는 유사 특허 DB 검색용 도구(`tool_search_patent_with_description`)만 존재했고, 출원번호를 직접 받는 입력 스키마나 `patent_id`(출원번호)를 기준으로 DB를 조회하는 **전용 툴 부재**
- 시스템 프롬프트에도 “출원번호를 직접 조회할 때는 어떻게 하라”는 가이드 문구가 없어 에이전트가 마땅한 도구를 고를 수 없었음
    

### 해결 방법

- **스키마 추가**
    
  - 출원번호 전용 입력/출력 스키마를 Pydantic으로 정의
    (예: `PatentIdInput`, `PatentDetailOutput` 형태로 설계)
    
- **툴 추가**
    
  - `tool_search_detail_patent_by_id` 구현.
    벡터 DB 메타데이터에 저장된 `patent_id`를 기준으로 해당 특허의 제목, 대표 청구항 텍스트 등을 모아서 반환하도록 함
    
- **프롬프트 보강**
    
  - 시스템 프롬프트에 `tool_search_detail_patent_by_id`를 명시적으로 소개하고,
    사용자가 “출원번호/등록번호 XXXXX에 대해 알려줘”라고 할 때 이 도구를 우선 사용하라고 기재
    

### 개선 후 질문 + 답변

```
[질문] 
1020230019980 이 출원번호에 대해서 대표 청구항들을 요약해서 알려줄래?

[답변] 
1. 기본 정보 요약  
- 출원번호: 1020230019980  
- 발명의 명칭: 컴퓨터 비전을 이용한 스마트 툴 박스  
- IPC(요약): 공구 수납(B25H), 산업용 로봇/매니퓰레이터(B25J), 자동 제어(G05D), 리프팅/운반(B66F) 등 기계식 툴 박스 + 자율주행/추종 제어에 관한 기술로 분류됩니다.  
- 주의: 아래 내용은 당사 컴퓨터 비전·모빌리티 중심 DB에 저장된 공보 텍스트를 기반으로 한 요약으로, 실제 공식 공보 전체와 100% 동일하다고 보기는 어렵습니다.

2. 대표 청구항(1~3항) 핵심 요약  

- 청구항 1 (가장 기본 개념, 독립항)  
  “사용자를 인식해 자율 주행으로 따라오는 스마트 툴 박스”에 관한 구성으로 판단됩니다.   

...(생략)
```

## **2) 후속 요청에서 제외/재구성 지시가 반영되지 않던 문제**

### 개선 전 질문 + 답변

```
[질문]
방금 찾은 특허 항목에서 3번은 내가 생각한 방향과 거리가 먼 것 같아. 3번 내용은 제외하고 다시 3개의 유사 특허 리스트를 만들어줘.

[답변]
현재 제공된 특허 리스트는 벡터 기반 유사도 검색을 통해 도출된 결과이며, 항목 1, 2, 3은 동일한 기준으로 계산된 유사도 순위에 의해 제시된 것입니다.

사용자가 제시한 “3번 제외” 지시는 이해하였으나, 본 시스템은 기존 검색 결과에서 특정 항목을 제거한 뒤 재정렬하는 방식이 아니라 같은 쿼리를 기반으로 다시 검색을 수행하게 됩니다.
따라서 새롭게 시도된 검색 역시 동일한 벡터 유사도 계산을 사용하게 되어 결과가 이전과 동일하게 반환되었습니다.

즉, 3번을 제외한 새로운 후보군을 생성하는 기능은 현재 제공되는 검색 메커니즘 내에서는 지원되지 않으며, 유사도 기준을 변경하거나 검색 조건을 구체적으로 조정하지 않는 이상 동일한 결과가 반복될 수밖에 없습니다.

만약 특정 기술 요소, IPC 코드, 적용 분야, 청구항 구성 등 구체적인 제외 조건을 추가한다면 검색 기준을 세분화하여 다른 후보를 제시할 가능성은 있습니다.
그러나 지금 제시된 지시만으로는 동일한 데이터 집합 내에서 새로운 리스트를 생성할 수 없기에 “재검색하였으나 동일한 결과”가 발생한 것입니다.
```

### 원인 분석

- **스키마 설계 부족**
    
    초기 `PatentSearchInput`에는 “이번 검색에서 제외할 특허”를 넘길 수 있는 필드 부재. → LLM 입장에서 “3번 빼달라”는 사용자의 의도를 도구 인자에 반영할 방법 X
    
- **번호 체계가 일관되지 않음**
    
    답변에서 특허를 소개할 때, `1번, 2번, 3번…` 뿐 아니라 `6번 유사 계열`, `(3)`, `①` 같은 **다른 형식의 번호**도 섞여 있었기 때문에, LLM이 사용자의 “3번” 요청을 **어느 번호 기준으로 이해해야 할지 계속 헷갈리는 구조**
    
- **재검색 로직 한계**
    
  - 도구는 단순히 `top_k`개만 가져와서 정렬 후 바로 반환하는 방식 → 일부 특허를 제외하고 다시 검색해도 **항상 같은 상위 후보들이 재등장**하는 문제


### 해결 방법

- **스키마 확장**
    - `PatentSearchInput`에 `exclude_patent_ids: List[str]` 필드 추가     
        → 이전 턴에서 제외해야 하는 출원번호들을 **명시적으로 넘길 수 있게** 변경.
        
    - `PatentSearchResult`에 `result_index: int` 필드 추가
        → “결과 1, 결과 2…”처럼 **한 턴 안에서의 순서를 구조화된 필드로 저장**하도록 설계.
        
- **특허 검색 툴 로직 보강**
  `tool_search_patent_with_description` 내부:
  1) 우선 `top_k`보다 훨씬 많은 후보(`search_pool_size`, 예: `top_k * 2`)를 가져오고,
        
  2) 그 안에서 `exclude_patent_ids`에 포함된 특허를 제거한 뒤,
        
  3) 남은 특허를 점수 기준으로 다시 정렬하고 **앞에서부터 `top_k`개만 선택**.
        
  4) 이 과정에서 `result_index = 1, 2, 3, …`로 **다시 번호를 매겨서** 반환하도록 변경.
        
- **시스템 프롬프트 강화**
    - 특허 리스트를 보여줄 때: 반드시 `결과 1. 특허 XXXXX`, `결과 2. 특허 YYYYY` 형식만 사용하게 하고, 다른 번호 포맷(①, (1), 1) 등은 **특허 번호 표현에 쓰지 말 것**을 명시
        
    - 사용자가 “2번, 4번 결과를 빼줘”라고 말하면: 직전 턴에서의 `결과 2`, `결과 4`에 해당하는 특허의 ID를 찾아서 `exclude_patent_ids`에 넣은 뒤 재검색하도록 가이드.
        
    - 제외를 요구하는 재검색을 여러 번 반복 시: “새 후보 2개가 기존 것보다 더 적합하다”고 **자신 있게 말하기 어려운 경우**, "DB 한계상 더 좋은 후보를 찾기 어렵다”는 식으로 **한계를 솔직하게 설명**하도록 규칙을 추가.
        

### 개선 후 질문 답변

```
[질문]
방금 찾은 특허 항목에서 3번은 내가 생각한 방향과 거리가 먼 것 같아. 3번 내용은 제외하고 다시 3개의 유사 특허 리스트를 만들어줘.

[답변]
1. 전제 정리  

- 사용자가 “3번 내용은 방향이 다르다”고 하신 것은, 바로 앞에서 제가 요약해 드린 **해당 특허의 종속 청구항(청구항 3: 초음파 장애물 회피)**로 이해됩니다.  
- 이번에는 그 장애물 회피 아이디어는 배제하고, **“컴퓨터 비전 기반 사용자 인식 + 일정 거리 유지 추종 주행”**에 더 초점을 맞춘 유사 특허 3건을, 기존 출원번호 1020230019980은 제외하여 검색했습니다.  
- 아래 유사 특허들은 모두 우리 시스템의 컴퓨터 비전·모빌리티 중심 벡터 DB를 기반으로 한 결과이므로, 실제 선행기술 조사는 KIPRIS 등에서 추가 확인이 필요합니다.

---

2. 유사 특허 리스트 (3건)

### 결과 1. 특허 1020230194714  
- 명칭: 비전 인식을 이용하여 관절 좌표를 인식하고 생성형 AI 알고리즘을 적용하여 웨이포인트 좌표를 자동 생성하는 매니퓰레이터 제어 시스템 및 그 방법

...(생략)
```



---

# 11. 수행 결과
(프로젝트 시연 캡처 이미지 첨부)

---

# 12. 프로젝트 확장 방향
- 

---
# 13.  한줄회고
(취합)
<table style="width:100%, table-layout: fixed;">

<tr>

<th style="min-width: 100px;">이름</th>

<th>회고 내용</th>

</tr>

<tr>

<td style="width: 10%" align="center">신지섭</td>

<td>(내용)</td>

</tr>

<tr>

<td style="width: 10%" align="center">왕혁준</td>

<td>(내용)</td>

</tr>

<tr>

<td style="width: 10%" align="center">임상민</td>

<td>프로젝트를 진행하면서 모델의 성능보다 설계가 훨씬 더 중요하다는 걸 확실히 느꼈다. 어떤 모델을 쓰느냐보다, 구조를 어떻게 잡고 어떤 요소들이 연결되는지 명확히 이해하는 것이 결과에 더 큰 영향을 준다. 특히 데이터의 구조를 제대로 파악하지 못하면 원하는 출력을 얻는 건 불가능하다는 것도 깨달았다. 다음 프로젝트에서는 내가 원하는 데이터의 형태가 무엇인지, 어떤 방식으로 출력되길 원하는지, 그리고 그 데이터가 다음 단계로 넘어가도 충분히 의미가 있는지를 먼저 점검하는 습관을 가져야겠다고 느꼈다. 이번 프로젝트는 그런 부분을 스스로 검증해볼 수 있는 좋은 계기였고, 이를 통해 한 단계 더 성장한 자신을 확인할 수 있었다. </td>

</tr>

<tr>

<td style="width: 10%" align="center">장효정</td>

<td>(내용)</td>

</tr>

<tr>

<td style="width: 10%" align="center">박도연</td>

<td>(내용)</td>

</tr>

</table>

---

# 라이선스
MIT License
