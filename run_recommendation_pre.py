import os
import json
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from operator import itemgetter

# --- 스크립트 시작 부분에서 환경변수를 로드합니다 ---
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
    return vectorstore.as_retriever(search_kwargs={'k': 3})

# --- 2. 추천 체인 생성 ---
def create_recommendation_chain(retriever):
    """Retriever를 입력받아 AI 추천 체인을 생성합니다."""
    print("추천 체인을 생성합니다.")
    llm = ChatUpstage()
    
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
    Youtube_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain_with_source = create_retrieval_chain(retriever, Youtube_chain)
    final_chain = rag_chain_with_source | itemgetter("answer") | JsonOutputParser()
    return final_chain

# --- 메인 실행 로직 ---
if __name__ == '__main__':
    # 가상의 모집 공고
    recruitment_post = {
        "role": "프론트엔드 개발자",
        "tech_stack": "TypeScript, React, Next.js",
        "description": "저희는 유저 친화적인 웹 서비스를 만드는 스타트업입니다. TypeScript와 React 경험이 풍부하고, 새로운 기술 학습에 대한 열정이 있는 분을 찾습니다."
    }

    input_text = (
        f"모집 직무: {recruitment_post['role']}\n"
        f"요구 기술: {recruitment_post['tech_stack']}\n"
        f"상세 내용: {recruitment_post['description']}"
    )

    retriever = load_vector_db()
    chain = create_recommendation_chain(retriever)
    
    print("\nAI 추천을 시작합니다...")
    result = chain.invoke({"input": input_text})
    
    print("\n--- 최종 AI 추천 결과 ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))