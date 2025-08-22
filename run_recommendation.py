import os
import json
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from operator import itemgetter

load_dotenv()

# --- 1. Vector DB 불러오기 ---
def load_vector_db():
    """로컬에 저장된 ChromaDB를 불러와 Retriever를 생성합니다."""
    
    print("Upstage 임베딩 모델을 초기화합니다.")
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    
    print("'./chroma_db'에서 Vector DB를 불러옵니다.")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="technician_profiles"
    )
    
    # Retriever를 생성합니다. k=3은 가장 유사한 3개의 문서를 가져오라는 의미입니다.
    return vectorstore.as_retriever(search_kwargs={'k': 3})

# --- 2. 추천 체인 생성 ---
# itemgetter를 import 목록에 추가합니다.
from operator import itemgetter

def create_recommendation_chain(retriever):
    """Retriever를 입력받아 AI 추천 체인을 생성합니다."""
    print("추천 체인을 생성합니다.")
    
    # LLM을 초기화합니다.
    llm = ChatUpstage()
    
    # 프롬프트 템플릿 
    prompt_template = """
    당신은 최고의 기술 스타업 채용 전문가입니다. 
    주어진 [모집 공고]와 여러 [후보자 프로필]을 바탕으로, 공고에 가장 적합한 후보자를 순서대로 3명 추천해주세요.

    [규칙]
    - 후보자의 이름, 경력, 핵심 기술을 명시해야 합니다.
    - 추천 이유는 반드시 [후보자 프로필]에 명시된 내용만을 근거로 2~3 문장으로 설명해야 합니다. 절대로 프로필에 없는 내용을 추측하거나 지어내서는 안 됩니다.
    - 최종 답변은 반드시 아래 예시와 같은 JSON 리스트 형식으로만 출력해야 합니다. 다른 설명은 절대 추가하지 마세요.

    [JSON 예시]
    [
      {{
        "rank": 1,
        "name": "후보자 이름",
        "career": "후보자의 경력 사항",
        "main_skills": ["핵심 기술1", "기술2"],
        "reason": "추천 이유를 여기에 작성합니다."
      }},
      {{
        "rank": 2,
        "name": "다른 후보자 이름",
        "career": "다른 후보자의 경력 사항",
        "main_skills": ["기술A", "기술B"],
        "reason": "다른 후보자의 추천 이유를 여기에 작성합니다."
      }},
      {{
        "rank": 3,
        "name": "또 다른 후보자 이름",
        "career": "세 번째 후보자의 경력 사항",
        "main_skills": ["기술C", "기술D"],
        "reason": "세 번째 후보자의 추천 이유를 여기에 작성합니다."
      }}
    ]

    ---
    [모집 공고]:
    {input}

    [후보자 프로필]:
    {context}
    ---

    [JSON 답변]:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # 문서 통합 및 LLM 호출 체인 생성
    Youtube_chain = create_stuff_documents_chain(llm, prompt)
    
    # retriever | Youtube_chain 까지는 이전과 동일
    rag_chain_with_source = create_retrieval_chain(retriever, Youtube_chain)
    
    # 최종 체인 수정: 중간에 itemgetter("answer")를 추가하여 'answer' 값만 추출
    final_chain = rag_chain_with_source | itemgetter("answer") | JsonOutputParser()
    
    return final_chain


# --- 신규 추가: 질의 확장 체인 생성 함수 ---
def create_query_expansion_chain():
    """사용자의 입력을 받아 검색용 키워드를 보강하는 체인을 생성합니다."""
    print("질의 확장 체인을 생성합니다.")
    llm = ChatUpstage()
    
    # 질의 확장을 위한 프롬프트 수정: "연관 기술 키워드"만 추가하도록 명확히 지시
    expansion_prompt_template = """
    당신은 최고의 IT 헤드헌터입니다. 사용자가 입력한 모집 공고의 핵심 기술을 파악하고, 그와 관련된 동의어나 유사 기술 키워드를 추가하여 검색에 용이한 문장으로 만들어주세요.
    예를 들어, 'React'가 있다면 'Next.js'나 '상태 관리'를 추가할 수 있습니다.
    
    [규칙]
    - 원래 공고에 없던 경력, 학력 등의 새로운 자격 요건을 절대 추가하지 마세요.
    - 문장은 원래 공고의 내용을 기반으로 자연스럽게 다시 작성해주세요.
    - 결과는 다른 설명 없이 확장된 공고 내용만 반환해주세요.

    [사용자 입력 공고]:
    {original_query}

    [확장된 공고]:
    """
    prompt = ChatPromptTemplate.from_template(expansion_prompt_template)
    
    # LLM의 답변을 문자열로 받기 위해 StrOutputParser를 사용
    return prompt | llm | StrOutputParser()


# --- 메인 실행 로직 수정 ---
if __name__ == '__main__':
    # 사용자가 입력한 간단한 모집 공고
    original_recruitment_post = "TypeScript와 React 경험이 풍부한 프론트엔드 개발자 찾습니다."

    # 1. 질의 확장 체인 생성 및 실행
    query_expander = create_query_expansion_chain()
    print("\n1단계: 질의 확장을 시작합니다...")
    expanded_query = query_expander.invoke({"original_query": original_recruitment_post})
    
    print("\n--- 원본 검색어 ---")
    print(original_recruitment_post)
    print("\n--- AI가 확장한 검색어 ---")
    print(expanded_query)
    
    # 2. Vector DB에서 Retriever 로드
    retriever = load_vector_db()
    
    # 3. 추천 체인 생성
    recommender = create_recommendation_chain(retriever)
    
    # 4. 추천 체인 실행 (확장된 검색어를 입력으로 사용)
    print("\n2단계: AI 추천을 시작합니다...")
    result = recommender.invoke({"input": expanded_query})
    
    # 5. 결과 출력
    print("\n--- 최종 AI 추천 결과 ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))