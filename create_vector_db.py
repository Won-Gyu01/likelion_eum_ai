import os
import json
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

# --- 1. JSON 파일에서 데이터 불러오기 ---
def load_profiles_from_json(filepath):
    """지정된 경로의 JSON 파일을 읽어 프로필 리스트를 반환합니다."""
    print(f"'{filepath}' 파일에서 프로필 데이터를 불러옵니다.")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# create_vector_db.py

# --- 2. 데이터 변환 ---
# create_vector_db.py

def convert_profiles_to_documents(profiles):
    """프로필 딕셔너리 리스트를 LangChain Document 객체 리스트로 변환합니다."""
    print("프로필 데이터를 LangChain Document 형식으로 변환합니다.")
    documents = []
    for profile in profiles:
        # 각 정보 앞에 명확한 태그를 붙여서 page_content를 재구성합니다.
        page_content = (
            f"[이름]: {profile['name']}\n"
            f"[직무]: {profile['career']}\n"
            f"[자기소개]: {profile['introduction']}\n"
            f"[보유 기술]: {profile['skills']}\n"
            f"[거주 지역]: {profile['location']}"
        )
        metadata = {
            "user_id": profile['id'],
            "name": profile['name'],
            "location": profile['location'],
            "resume_url": profile['resume_url']
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents

# --- 3. 임베딩 및 Vector DB 저장 ---
def create_and_store_vectors(documents):
    """Document 리스트를 벡터로 변환하여 ChromaDB에 저장합니다."""
    print("환경변수를 로드합니다.")
    load_dotenv()
    
    print("Upstage 임베딩 모델을 초기화합니다.")
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

    print("ChromaDB에 벡터를 저장합니다...")
    vectorstore = Chroma.from_documents(
        documents,
        embeddings,
        collection_name="technician_profiles",
        persist_directory="./chroma_db"
    )
    print("저장 완료!")
    print(f"총 {vectorstore._collection.count()}개의 프로필이 성공적으로 저장되었습니다.")
    return vectorstore

# --- 메인 실행 로직 ---
if __name__ == '__main__':
    # ./chroma_db 폴더가 이미 있다면, 중복 저장을 방지
    if os.path.exists("./chroma_db"):
        print("이미 './chroma_db' 폴더가 존재합니다. Vector DB가 생성되어 있습니다.")
    else:
        # 1. JSON 파일 로드
        profiles = load_profiles_from_json('profiles.json')
        
        # 2. Document 객체로 변환
        documents_to_store = convert_profiles_to_documents(profiles)
        
        # 3. Vector DB 생성 및 저장
        create_and_store_vectors(documents_to_store)