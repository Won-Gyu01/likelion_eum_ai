import os
import json
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore # ⭐️ Chroma 대신 Pinecone import

# JSON 파일에서 프로필을 로드하는 함수 (이전과 동일)
def load_profiles_from_json(filepath):
    print(f"'{filepath}' 파일에서 프로필 데이터를 불러옵니다.")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# 프로필을 LangChain Document로 변환하는 함수 (이전과 동일)
def convert_profiles_to_documents(profiles):
    print("프로필 데이터를 LangChain Document 형식으로 변환합니다.")
    documents = []
    for profile in profiles:
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
            "resumeUrl": profile['resumeUrl']
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents

# ⭐️ Pinecone에 벡터를 저장하는 함수
def create_and_store_vectors(documents):
    """Document 리스트를 벡터로 변환하여 Pinecone에 저장합니다."""
    print("Upstage 임베딩 모델을 초기화합니다.")
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

    print("Pinecone에 벡터를 저장합니다...")
    # Pinecone.from_documents()를 사용하여 데이터를 저장합니다.
    # index_name은 Pinecone에서 만든 인덱스 이름과 동일해야 합니다.
    PineconeVectorStore.from_documents(
        documents,
        embeddings,
        index_name="eum-technicians" 
    )
    print("저장 완료!")

# 메인 실행 로직
if __name__ == '__main__':
    load_dotenv()
    profiles = load_profiles_from_json('profiles.json')
    documents_to_store = convert_profiles_to_documents(profiles)
    create_and_store_vectors(documents_to_store)
    print("\n성공적으로 Pinecone에 프로필 데이터를 저장했습니다.")