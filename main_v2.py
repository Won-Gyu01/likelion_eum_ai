import os
import json
from dotenv import load_dotenv
from typing import List, Union # ⭐️ Union 타입을 사용하기 위해 추가

# LangChain & Vector DB
from langchain_upstage import UpstageEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from operator import itemgetter

# FastAPI
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel

# --- 1. 초기 설정 ---
load_dotenv()

# --- 2. AI 모델 및 Vector DB 로딩 ---

embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
index_name = "eum-technicians"

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

def create_recommendation_chain():
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
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
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain_with_source = create_retrieval_chain(retriever, question_answer_chain)
    final_chain = rag_chain_with_source | itemgetter("answer") | JsonOutputParser()
    return final_chain

recommendation_chain = create_recommendation_chain()


# --- 3. FastAPI 서버 설정 ---
app = FastAPI()

AI_API_KEY = os.getenv("AI_SERVER_API_KEY")
async def api_key_auth(x_api_key: str = Header(...)):
    if x_api_key != AI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

class RecruitmentRequest(BaseModel):
    title: str
    location: str
    position: str
    skills: str
    career: str
    content: str

class ProfileRequest(BaseModel):
    id: int
    name: str
    career: str
    introduction: str
    skills: str
    location: str
    resumeUrl: str

# --- 4. API 엔드포인트(Endpoint) 정의 ---

@app.get("/")
def read_root():
    return {"status": "AI 추천 서버가 정상적으로 동작하고 있습니다."}

@app.post("/recommend", dependencies=[Depends(api_key_auth)])
def recommend_technicians(request: RecruitmentRequest):
    input_text = (
        f"모집 공고 제목: {request.title}\n"
        f"직무: {request.position}\n"
        f"근무지: {request.location}\n"
        f"요구 경력: {request.career}\n"
        f"요구 기술: {request.skills}\n"
        f"상세 내용: {request.content}"
    )
    result = recommendation_chain.invoke({"input": input_text})
    return result

# ⭐️ 수정: 프로필 추가 및 수정 API (한 명 또는 여러 명 모두 처리 가능)
@app.post("/profiles", dependencies=[Depends(api_key_auth)])
def upsert_profiles(profiles: Union[ProfileRequest, List[ProfileRequest]]):
    """새로운 프로필(한 명 또는 여러 명)을 추가하거나 기존 프로필을 업데이트합니다."""
    
    # 입력이 단일 객체이면 리스트로 감싸서 처리의 일관성을 유지
    if isinstance(profiles, ProfileRequest):
        profiles_to_process = [profiles]
    else:
        profiles_to_process = profiles

    documents = []
    ids_to_upsert = []

    for profile in profiles_to_process:
        page_content = (
            f"[이름]: {profile.name}\n"
            f"[직무]: {profile.career}\n"
            f"[자기소개]: {profile.introduction}\n"
            f"[보유 기술]: {profile.skills}\n"
            f"[거주 지역]: {profile.location}"
        )
        metadata = {
            "user_id": profile.id,
            "name": profile.name,
            "location": profile.location,
            "resumeUrl": profile.resumeUrl
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
        ids_to_upsert.append(str(profile.id))
    
    if documents:
        vectorstore.add_documents(documents, ids=ids_to_upsert)
    
    print(f"{len(documents)}개의 프로필이 성공적으로 추가/수정되었습니다.")
    return {"status": "success", "processed_count": len(documents)}
