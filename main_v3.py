import os
import json
from dotenv import load_dotenv
from typing import List, Union

# LangChain & Vector DB
from langchain_upstage import UpstageEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from operator import itemgetter
from langchain_core.runnables import RunnableLambda

# FastAPI
from fastapi import FastAPI, Header, HTTPException, Depends, Request
from pydantic import BaseModel, Field

# --- 1. 초기 설정 ---
load_dotenv()

# --- 2. AI 모델 및 Vector DB 로딩 ---

embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
llm = ChatUpstage()

team_recommender_store = PineconeVectorStore.from_existing_index(
    index_name="eum-technicians",
    embedding=embeddings
)

support_recommender_store = PineconeVectorStore.from_existing_index(
    index_name="eum-support-programs",
    embedding=embeddings
)

# --- 3. 추천 체인 생성 ---

def create_team_recommendation_chain():
    retriever = team_recommender_store.as_retriever(search_kwargs={'k': 3})
    prompt_template = """
    당신은 최고의 기술 스타업 채용 전문가입니다. 
    주어진 [모집 공고]와 여러 [후보자 프로필]을 바탕으로, 공고에 가장 적합한 후보자를 순서대로 3명 추천해주세요.
    [규칙]
    - 후보자의 이름, 경력, 핵심 기술을 명시해야 합니다.
    - 추천 이유는 반드시 [후보자 프로필]에 명시된 내용만을 근거로 2-3 문장으로 설명해야 합니다.
    - 최종 답변은 반드시 아래 예시와 같은 JSON 리스트 형식으로만 출력해야 합니다.
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
    [모집 공고]: {input}
    [후보자 프로필]: {context}
    ---
    [JSON 답변]:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain_with_source = create_retrieval_chain(retriever, question_answer_chain)
    
    def combine_data(rag_output):
        try:
            parsed_answer = JsonOutputParser().parse(rag_output["answer"])
            retrieved_docs = rag_output["context"]

            if isinstance(parsed_answer, dict):
                answer_json = [parsed_answer]
            elif isinstance(parsed_answer, list):
                answer_json = parsed_answer
            else:
                return {"error": "AI response is not a valid JSON object or list", "raw_answer": parsed_answer}

            for i, item in enumerate(answer_json):
                if i < len(retrieved_docs) and isinstance(item, dict):
                    item["userId"] = retrieved_docs[i].metadata.get("user_id")
            
            return answer_json
        except Exception as e:
            print(f"Error processing AI response: {e}")
            return {"error": "Failed to process AI response", "raw_answer": rag_output.get("answer")}

    final_chain = rag_chain_with_source | RunnableLambda(combine_data)
    return final_chain

# ⭐️ 수정: 창업 지원 사업 추천 체인의 프롬프트와 후처리 로직 변경
def create_support_recommendation_chain():
    retriever = support_recommender_store.as_retriever(
        search_kwargs={
            'k': 3,
            'filter': {'is_recruiting': True}
        }
    )
    # ⭐️ 수정: 프롬프트에서 AI에게 사실 정보 생성을 요청하는 부분을 모두 제거
    prompt_template = """
    당신은 최고의 창업 컨설턴트입니다.
    주어진 [사용자의 창업 아이템 정보]와 여러 [창업 지원 사업 목록]을 바탕으로, 가장 적합한 지원 사업을 3개 추천해주세요.
    [규칙]
    - 지원 사업의 이름(title)과 지역(region)을 명시해야 합니다.
    - 왜 적합한지 [지원 사업 목록]의 정보를 근거로 2-3 문장으로 설명해야 합니다.
    - 최종 답변은 반드시 아래 예시와 같은 JSON 리스트 형식으로만 출력해야 합니다.
    [JSON 예시]
    [
      {{
        "rank": 1,
        "title": "지원 사업 이름",
        "region": "대상 지역",
        "reason": "추천 이유를 여기에 작성합니다."
      }}
    ]
    ---
    [사용자의 창업 아이템 정보]: {input}
    [창업 지원 사업 목록]: {context}
    ---
    [JSON 답변]:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain_with_source = create_retrieval_chain(retriever, question_answer_chain)
    
    # ⭐️ 수정: AI 답변과 메타데이터를 조합하여 모든 사실 정보를 추가하는 함수
    def combine_support_data(rag_output):
        try:
            answer_json = JsonOutputParser().parse(rag_output["answer"])
            retrieved_docs = rag_output["context"]

            for i, item in enumerate(answer_json):
                if i < len(retrieved_docs) and isinstance(item, dict):
                    # 메타데이터에서 정확한 사실 정보를 모두 가져와 추가
                    item["support_field"] = retrieved_docs[i].metadata.get("support_field")
                    item["end_date"] = retrieved_docs[i].metadata.get("end_date")
                    item["apply_url"] = retrieved_docs[i].metadata.get("apply_url")
            
            return answer_json
        except Exception as e:
            print(f"Error processing AI support response: {e}")
            return {"error": "Failed to process AI response", "raw_answer": rag_output.get("answer")}

    final_chain = rag_chain_with_source | RunnableLambda(combine_support_data)
    return final_chain

team_recommendation_chain = create_team_recommendation_chain()
support_recommendation_chain = create_support_recommendation_chain()


# --- 4. FastAPI 서버 설정 ---
app = FastAPI()

AI_API_KEY = os.getenv("AI_SERVER_API_KEY")
async def api_key_auth(x_api_key: str = Header(...)):
    if x_api_key != AI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

class TeamRecruitmentRequest(BaseModel):
    title: str
    location: str
    position: str
    skills: str
    career: str
    content: str
    
    class Config:
        alias_generator = lambda string: ''.join(word.capitalize() if i != 0 else word for i, word in enumerate(string.split('_')))
        populate_by_name = True


class ProfileRequest(BaseModel):
    id: int
    name: str
    career: str
    introduction: str
    skills: str
    location: str
    resume_url: str = Field(..., alias='resumeUrl')

class IncubationCenterRequest(BaseModel):
    id: str
    title: str
    region: str
    support_field: str = Field(..., alias='supportField')
    receipt_start_date: str = Field(..., alias='receiptStartDate')
    receipt_end_date: str = Field(..., alias='receiptEndDate')
    recruiting: Union[bool, int]
    apply_url: str = Field(..., alias='applyUrl')

# --- 5. API 엔드포인트(Endpoint) 정의 ---

@app.get("/")
def read_root():
    return {"status": "AI 추천 서버가 정상적으로 동작하고 있습니다."}

@app.post("/recommend/users", dependencies=[Depends(api_key_auth)])
def recommend_team_members(request: TeamRecruitmentRequest):
    input_text = (
        f"모집 공고 제목: {request.title}\n"
        f"직무: {request.position}\n"
        f"근무지: {request.location}\n"
        f"요구 경력: {request.career}\n"
        f"요구 기술: {request.skills}\n"
        f"상세 내용: {request.content}"
    )
    #수정: with_config를 사용해 "Team Recommendation"이라는 이름표를 붙여줍니다.
    result = team_recommendation_chain.with_config(
        {"run_name": "Team Recommendation"}
    ).invoke({"input": input_text})
    return result

@app.post("/recommend/incubation-centers", dependencies=[Depends(api_key_auth)])
def recommend_support_programs(request: TeamRecruitmentRequest):
    input_text = (
        f"창업 아이템 제목: {request.title}\n"
        f"희망 지역: {request.location}\n"
        f"필요 직무: {request.position}\n"
        f"핵심 기술: {request.skills}\n"
        f"팀 경력 수준: {request.career}\n"
        f"상세 아이템 설명: {request.content}"
    )
    #수정: with_config를 사용해 "Support Program Recommendation"이라는 이름표를 붙여줍니다.
    result = support_recommendation_chain.with_config(
        {"run_name": "Support Program Recommendation"}
    ).invoke({"input": input_text})
    return result

@app.post("/profiles", dependencies=[Depends(api_key_auth)])
def upsert_profiles(profiles: Union[ProfileRequest, List[ProfileRequest]]):
    if isinstance(profiles, ProfileRequest):
        profiles_to_process = [profiles]
    else:
        profiles_to_process = profiles
    documents, ids_to_upsert = [], []
    for profile in profiles_to_process:
        page_content = (
            f"[이름]: {profile.name}\n[직무]: {profile.career}\n[자기소개]: {profile.introduction}\n"
            f"[보유 기술]: {profile.skills}\n[거주 지역]: {profile.location}"
        )
        metadata = {"user_id": profile.id, "name": profile.name, "location": profile.location, "resumeUrl": profile.resume_url}
        documents.append(Document(page_content=page_content, metadata=metadata))
        ids_to_upsert.append(str(profile.id))
    if documents:
        team_recommender_store.add_documents(documents, ids=ids_to_upsert)
    return {"status": "success", "processed_count": len(documents)}

@app.post("/incubation-centers", dependencies=[Depends(api_key_auth)])
def upsert_incubation_centers(programs: Union[IncubationCenterRequest, List[IncubationCenterRequest]]):
    if isinstance(programs, IncubationCenterRequest):
        programs_to_process = [programs]
    else:
        programs_to_process = programs
    documents, ids_to_upsert = [], []
    for program in programs_to_process:
        page_content = (
            f"[사업명]: {program.title}\n"
            f"[지원 분야]: {program.support_field}\n"
            f"[지역]: {program.region}"
        )
        metadata = {
            "program_id": program.id, "title": program.title, "region": program.region,
            "support_field": program.support_field, "start_date": program.receipt_start_date,
            "end_date": program.receipt_end_date, "is_recruiting": bool(program.recruiting),
            "apply_url": program.apply_url
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
        ids_to_upsert.append(str(program.id))
    if documents:
        support_recommender_store.add_documents(documents, ids=ids_to_upsert)
    return {"status": "success", "processed_count": len(documents)}
