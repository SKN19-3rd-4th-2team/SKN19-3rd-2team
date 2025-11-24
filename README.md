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
1) 관련 기사 1

<img width="1193" height="300" alt="image" src="https://github.com/user-attachments/assets/01a10fac-f137-4984-a757-5f0b05299de1" />

2) 관련 기사 2

<img width="1332" height="243" alt="image" src="https://github.com/user-attachments/assets/ed10205f-8d30-4f04-a764-3312bb8c7130" />



## 2.4 프로젝트 목표
- 의미 기반 검색이 가능한 임베딩 기반 특허 검색 시스템 구축
- LLM이 Tool을 자동 선택해 질의응답 가능하도록 설계
- 검색된 출원번호·청구항 기반으로 구조화된 답변 생성
- 대규모 데이터에서 빠르고 정확한 유사도 기반 검색을 지원

## 2-5. 범위 선정
- 중소기업이 최근 특허를 많이 내고 있는 분야: **SW개발업**
  
  <img width="600" height="371" alt="image" src="https://github.com/user-attachments/assets/653f9d6c-1dde-4ad3-8825-9e513e6bdc99" />

- 그 업계에서 인공지능 관련 가장 많은 산업: **시각지능**

 - <img width="600" height="197" alt="image" src="https://github.com/user-attachments/assets/a374b167-52c0-4e51-92c7-5b92fcba931a" />

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
- **LLM**: gpt-5.1 / gpt-4o (테스트 환경에서 모델 교차 실험)

## 폴더 구성
(구조)

---

# 4. 시스템 아키텍처
![My First Board - 시스템 아키텍쳐](https://github.com/user-attachments/assets/b7ce8bb3-c20f-4e6a-a4e2-54fddfed540b)


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
| FR-04 | 멀티쿼리 처리 | 여러 키워드·문장을 매핑하는 경우, **멀티쿼리 검색 + 정규화**를 통해 다면적 검색이 가능해야 한다 | 중 | Z-score 정규화 적용 |
| FR-05 | 유사도 기반 정렬 | 검색된 청구항은 출원번호 단위로 그룹핑하여 **벡터 거리·BM25 결합(hybrid search)** 으로 랭킹되어야 한다 | 상 |  |
| FR-06 | LLM 답변 생성 | LLM은 검색 결과를 사용해 **근거 기반 설명형 답변**을 생성해야 한다 | 상 | RAG 기반 생성 |
| FR-07 | 다양한 검색 유형 | 사용자는 **기술명 / IPC / 키워드 / 문장** 등 다양한 형태로 검색이 가능해야 한다 | 중 |  |


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
  - IPC 영문 설명(`title_en`)
  - 코드의 레벨 구분(`kind`)
  - 부모 IPC 코드(`parent`)
  - 최상위부터 현재 코드까지 전체 계층('path')
  
- **2. 공개 특허 문서**: 약 30,000건
- **청구항 개수:** 약 590,000건
- **필드 구성**

<img width="908" height="563" alt="image" src="https://github.com/user-attachments/assets/45e74890-1027-4fdd-9b0d-25a9b81f8781" />

  - 출원번호(`id`)
  - 발명의 명칭(`name`)
  - 요약('abstract')
  - 청구항(`claim`)
  - IPC 코드(`IpcNumber`) 등


### 전처리
- **IPC 코드**
  - xml파일 파싱, 필요 정보 json 변환
  - 리스트 형태의 path를 list 형태로 변경
  - 임베딩 생성 후 ChromaDB 저장
    
- **특허 문서**
 - xml파일 파싱, 필요 정보 json 변환
 - 청구항 분리 및 정규화
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
### 1-1. multy-query
**문제점**
- results 200개를 검색해올 때, 쿼리를 여러 개 던지면 검색 결과가 배수로 늘어나는 것을 처리하기 위해 distance 기준으로 200개만 남김 → 여러 쿼리가 모두 중요한 키워드인데 한 쿼리와 강하게 유사한 청구항이 많다면, 그 쿼리에 대한 유사도 높은 청구항만 살아남는 문제

**해결 방안**
- 각 쿼리의 검색결과로 나온 distance에 대해 평균, 표준편차를 구해서 z 정규화, 그 z-score로 200개만 남김

**해결 결과**

<img width="806" height="129" alt="image" src="https://github.com/user-attachments/assets/3e9bb923-7db3-4aa0-ad16-704f77223b6f" />


- 200개의 ids의 교집합을 비교해봤을 때, 약 25~50% 정도의 다른 청구항이 상위로 올라오며 편향 완화됨(랭킹 다양성 증가)


### 1-2. re-ranking
**문제점**
- 유사도가 높은 특허라면, 많은 수의 청구항이 상위로 올라와 편향 발생

**해결 방안**
- 출원번호로 그룹화 하여 ‘검색된 청구항 중 상위 세 개의 유사도 평균’, ‘제일 높은 청구항의 유사도’, ‘검색된 청구항의 개수’를 이용해 score 계산.

**평가 지표**
- 유사 특허의 기준을 `같은 IPC 코드 공유`로 잡고, 특정 IPC 코드를 갖고 있는 특허의 출원번호를 저장해서  re-ranking했을 때 나오는 출원번호와 비교< 수정 필요

**해결 결과**
- 그냥 z-score 기준 상위 TOP_K 출원번호와 비교했을 때, 30개 중 20개만 일치하는 등의 랭킹 다양성 보임
- (눈에 보이는 사진이 있을까?)
 
## 2. IPC DB 검색 최적화
### 2-1. 계층 구조 관한 문제
**문제점**
- 1) IPC코드-설명을 각 DB에 저장한 결과, IPC 하위코드가 상위 의미를 포함하지 않아 임베딩이 단절됨
- (사진)

**해결 방안**
- 1) 하위코드 텍스트에 상위코드 내용을 결합해 의미 단절 해소
- 2) 각 metadata에 레벨 정보 추가, 프롬프트 수정 "큰 분야 먼저, 세부는 나중"

**해결 결과**

### 2-2. 오염 문제

---

# 10. 테스트 & 개선 노력

### 테스트 시나리오
- Query 예시:


### 결과 요약



---

# 11. 수행 결과
(프로젝트 시연 캡처 이미지 첨부)

---

# 12.  한줄회고
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
