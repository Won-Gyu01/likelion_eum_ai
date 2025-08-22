import os
import json
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
#from langchain_chroma import Chroma
from operator import itemgetter
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel

# --- 1. AI 로직 (이전과 거의 동일) ---

load_dotenv()

# AI 서버의 비밀번호를 .env 파일에서 불러옴
AI_API_KEY = os.getenv("AI_SERVER_API_KEY")

#  '문지기' 함수: x-api-key 헤더를 검사하는 함수
async def api_key_auth(x_api_key: str = Header(...)):
    if x_api_key != AI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# def load_vector_db():
#    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
#    vectorstore = Chroma(
#        persist_directory="./chroma_db",
#        embedding_function=embeddings,
#        collection_name="technician_profiles"
#    )
#    return vectorstore.as_retriever(search_kwargs={'k': 3})

# Vector DB를 불러오는 함수
def load_vector_db():
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    
    print("Pinecone에서 기존 인덱스를 불러옵니다.")
    # PineconeVectorStore.from_existing_index()를 사용하여 기존 DB에 접속합니다.
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name="eum-technicians",
        embedding=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={'k': 3})

def create_recommendation_chain(retriever):
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

# --- 2. FastAPI 서버 설정 (최종 수정) ---

app = FastAPI()

retriever = load_vector_db()
chain = create_recommendation_chain(retriever)

# 최종: 확정된 컬럼명에 맞춘 입력 형식
class RecruitmentRequest(BaseModel):
    title: str
    location: str
    position: str
    skills: str
    career: str
    content: str
    
#  @app.post 데코레이터에 '문지기' 함수를 추가
@app.post("/recommend", dependencies=[Depends(api_key_auth)])
def recommend_technicians(request: RecruitmentRequest):
    """모집 공고를 받아 AI 추천 결과를 반환하는 API 엔드포인트"""
    print("추천 요청을 받았습니다:", request)
    
    # 최종: 확정된 컬럼명을 모두 사용하여 AI에게 전달할 검색어 생성
    input_text = (
        f"모집 공고 제목: {request.title}\n"
        f"직무: {request.position}\n"
        f"근무지: {request.location}\n"
        f"요구 경력: {request.career}\n"
        f"요구 기술: {request.skills}\n"
        f"상세 내용: {request.content}"
    )
    
    result = chain.invoke({"input": input_text})
    
    return result

@app.get("/")
def read_root():
    return {"status": "AI 추천 서버가 정상적으로 동작하고 있습니다."}