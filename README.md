# 이음(Eum) 창업 A부터 Z까지

**이음(Eum)**은 창업을 준비하는 사람들에게 AI를 기반으로 최적의 **팀원**과 **정부 지원 사업**을 추천해주는 플랫폼의 백엔드 서버입니다. 이 저장소는 FastAPI로 구현되었으며, LangChain을 통해 Upstage LLM과 Pinecone 벡터 데이터베이스를 통합하여 핵심 추천 기능을 제공합니다.

## 주요 기능

* **AI 팀원 추천**: 모집 공고 내용을 분석하여 Pinecone DB에 저장된 인재 중 가장 적합한 팀원을 추천합니다.
* **AI 지원 사업 추천**: 프로젝트 개요를 바탕으로 가장 관련성 높은 정부/지자체 지원 사업 및 창업 경진대회 정보를 추천합니다.
* **데이터 관리**: 새로운 인재 프로필과 지원 사업 정보를 DB에 추가하고 업데이트할 수 있는 API 엔드포인트를 제공합니다.
* **간편한 로컬 테스트**: FastAPI의 자동 대화형 문서를 통해 프론트엔드 없이 모든 API 기능을 직접 테스트할 수 있습니다.

---

## 핵심 로직 (AI 추천 과정)

1.  **API 요청**: 사용자가 추천을 원하는 정보(팀원 모집 공고 등)를 담아 FastAPI 엔드포인트(`_**/recommend/teams**_`)로 요청을 보냅니다.
2.  **유사도 검색 (Retrieval)**: LangChain은 요청된 텍스트를 `Upstage Embedding` 모델을 사용해 벡터로 변환한 뒤, `Pinecone` 벡터 DB에서 가장 유사도가 높은 정보(인재 프로필 등)를 검색합니다.
3.  **LLM 답변 생성 (Generation)**: 검색된 정보와 원본 요청을 `ChatUpstage` LLM에 전달하여, 가장 적합한 추천 목록을 최종적으로 생성하고 JSON 형식으로 가공합니다.
4.  **API 응답**: 서버는 LLM이 생성한 추천 결과를 사용자에게 JSON 형태로 응답합니다.

---

## 기술 스택

* **Framework**: `FastAPI`
* **AI / LLM**: `LangChain`, `Upstage LLM`, `Upstage Embeddings`
* **Vector Database**: `Pinecone`
* **Web Server**: `Uvicorn`

---

##  로컬 환경에서 실행 및 테스트하기

### 1단계: 프로젝트 복제 및 환경 설정

```bash
# 1. 저장소 복제
git clone [https://github.com/Won-Gyu01/likelion_eum_ai.git]
cd likelion_eum_ai

# 2. 파이썬 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 필요 라이브러리 설치
pip install -r requirements.txt
```

### 2단계: 환경 변수 설정 (`.env` 파일)

프로젝트 루트 디렉토리에 `.env` 파일을 만들고, 아래와 같이 본인의 API 키를 입력하세요.

```env
# .env

# Upstage AI 모델 API 키
UPSTAGE_API_KEY="YOUR_UPSTAGE_API_KEY_HERE"

# Pinecone 벡터 DB API 키
PINECONE_API_KEY="YOUR_PINECONE_API_KEY_HERE"

# 서버 내부 인증용 API 키 (자유롭게 설정)
SERVER_API_KEY="YOUR_CUSTOM_SECRET_KEY"
```

### 3단계: FastAPI 서버 실행

터미널에 아래 명령어를 입력하여 로컬 서버를 실행합니다.

```bash
uvicorn main_v3:app --reload
```

서버가 성공적으로 실행되면 터미널에 `Application startup complete.` 메시지와 함께 `http://127.0.0.1:8000` 주소가 나타납니다.

### 4단계: API 기능 테스트하기

웹 브라우저에서 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** 로 접속하세요.

FastAPI가 자동으로 생성해주는 대화형 API 문서 페이지가 나타납니다. 여기에서 프론트엔드 UI 없이 각 API 엔드포인트를 직접 테스트해볼 수 있습니다.

* `POST /recommend/teams`를 클릭하고 `Try it out` 버튼을 누른 후, 예시 데이터를 입력하여 AI 팀원 추천 기능을 확인해보세요.
* 다른 엔드포인트들도 동일한 방식으로 테스트할 수 있습니다.
