import os
import json
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

def load_support_programs_from_json(filepath):
    """지정된 경로의 JSON 파일을 읽어 지원 사업 리스트를 반환합니다."""
    print(f"'{filepath}' 파일에서 지원 사업 데이터를 불러옵니다.")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_programs_to_documents(programs):
    """지원 사업 딕셔너리 리스트를 LangChain Document 객체 리스트로 변환합니다."""
    print("지원 사업 데이터를 LangChain Document 형식으로 변환합니다.")
    documents = []
    for program in programs:
        # AI가 의미를 잘 파악할 수 있도록 핵심 텍스트 정보들을 합쳐 page_content를 구성합니다.
        page_content = (
            f"[사업명]: {program['title']}\n"
            f"[지원 분야]: {program['support_field']}\n"
            f"[지역]: {program['region']}"
        )
        # 검색 결과 필터링 및 최종 정보 표시에 필요한 데이터는 metadata에 저장합니다.
        metadata = {
            "program_id": program['id'],
            "title": program['title'],
            "region": program['region'],
            "support_field": program['support_field'],
            "start_date": program['receipt_start_date'],
            "end_date": program['receipt_end_date'],
            "is_recruiting": bool(program['recruiting']),
            "apply_url": program['apply_url']
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents

def create_and_store_support_vectors(documents):
    """Document 리스트를 벡터로 변환하여 새로운 Pinecone 인덱스에 저장합니다."""
    print("Upstage 임베딩 모델을 초기화합니다.")
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

    # ⭐️ 중요: 새로운 인덱스 이름을 사용합니다.
    index_name = "eum-support-programs"

    print(f"Pinecone의 '{index_name}' 인덱스에 벡터를 저장합니다...")
    PineconeVectorStore.from_documents(
        documents,
        embeddings,
        index_name=index_name
    )
    print("저장 완료!")

# 메인 실행 로직
if __name__ == '__main__':
    load_dotenv()
    programs = load_support_programs_from_json('support_programs.json')
    documents_to_store = convert_programs_to_documents(programs)
    create_and_store_support_vectors(documents_to_store)
    print("\n성공적으로 Pinecone에 창업 지원 사업 데이터를 저장했습니다.")
