"""
RAG_1 FastAPI Server
마이크로서비스 아키텍처를 위한 독립 RAG API 서버
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import os
import json
import time
from dotenv import load_dotenv
from datetime import datetime

# from src.naiveRAG import NaiveRAG
# from src.FRAG import FRAG
from src.AtomRAG.llm.gemini import gemini_complete, gemini_embed
from src.AtomRAG import AtomRAG, QueryParam
from src.utils.config_utils import BaseConfig

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="RAG_1 API Server",
    description="Retrieval-Augmented Generation API Server",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수 -> 서버에서 계속 사용하는 객체 생성
# rag_system: Optional[NaiveRAG] = None
rag_system: Optional[AtomRAG] = None
rag_config: Optional[BaseConfig] = None
indexed_chunks: List[str] = []
startup_time: Optional[datetime] = None

# ========== Pydantic 모델 정의 ==========

class QueryRequest(BaseModel):
    """RAG 쿼리 요청"""
    query: str = Field(..., description="사용자 질문")
    top_k: Optional[int] = Field(5, description="검색할 문서 수", ge=1, le=100)
    method: Optional[str] = Field("naive", description="RAG 방법 (naive/frag)")
    max_context_tokens: Optional[int] = Field(5400, description="최대 컨텍스트 토큰", ge=100)
    is_short_answer: Optional[bool] = Field(True, description="짧은 답변 여부")

class RetrieveRequest(BaseModel):
    """문서 검색 요청"""
    query: str = Field(..., description="검색 쿼리")
    top_n: Optional[int] = Field(5, description="반환할 문서 수", ge=1, le=50)

class SourceDocument(BaseModel):
    """소스 문서"""
    title: str
    content: str
    score: float

class QueryResponse(BaseModel):
    """RAG 쿼리 응답"""
    answer: str
    sources: List[SourceDocument]
    metadata: Dict[str, Any]

class RetrieveResponse(BaseModel):
    """문서 검색 응답"""
    documents: List[SourceDocument]
    retrieval_time: float

class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    rag_initialized: bool
    method: Optional[str] = None
    dataset: Optional[str] = None
    indexed_docs: int
    uptime_seconds: Optional[float] = None

# ========== 헬퍼 함수 ==========

def split_doc_into_chunk(doc, max_tokens, overlap_token_size, title_dict):
    """문서를 청크로 분할"""
    import tiktoken
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _encode(x: str):
        return tokenizer.encode(x)
    
    def _decode(tokens):
        return tokenizer.decode(tokens)
    
    if not isinstance(doc, str):
        doc = str(doc) if doc is not None else ""
    
    if max_tokens is None or max_tokens <= 0:
        return [doc], {}
    
    overlap_tokens = max(0, min(overlap_token_size or 0, max_tokens - 1))
    step = max_tokens - overlap_tokens
    
    tokens = _encode(doc) if doc else []
    chunks = []
    
    if not tokens:
        return [""] if doc == "" else [doc], {}
    
    chunk_position_dict = {}
    
    if title_dict is None:
        i = 0
        total = len(tokens)
        while i < total:
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = _decode(chunk_tokens)
            chunks.append(chunk_text)
            i += step
    else:
        i = 0
        total = len(tokens)
        doc_len_dict = {}
        for key in title_dict.keys():
            for len_info in title_dict[key]:
                doc_len_dict[len_info] = key
        doc_len_list = sorted(list(doc_len_dict.keys()))
        point = 0
        finished_docs = ''
        
        while i < total:
            start = len(finished_docs)
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = _decode(chunk_tokens)
            chunks.append(chunk_text)
            end = len(finished_docs + chunk_text)
            finished_docs = finished_docs + _decode(tokens[i:i+step])
            i += step
            
            included = []
            for j in range(point, len(doc_len_list)):
                if doc_len_list[j][1] < start:
                    point += 1
                    continue
                if doc_len_list[j][0] > end:
                    break
                included.append(doc_len_list[j])
            
            for included_len in included:
                if doc_len_dict[included_len] not in chunk_position_dict.keys():
                    chunk_position_dict[doc_len_dict[included_len]] = [chunk_text]
                else:
                    chunk_position_dict[doc_len_dict[included_len]].append(chunk_text)
    
    return chunks, chunk_position_dict

def text_file_into_chunks(file_path, chunk_max_token_size, chunk_overlap_token_size):
    """텍스트 파일을 청크로 변환"""
    logger.info(f"텍스트 파일 로딩: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text_content = f.read()
    
    logger.info(f"텍스트 파일 크기: {len(text_content):,}자")
    
    # 텍스트를 청크로 분할
    chunks, _ = split_doc_into_chunk(
        text_content,
        chunk_max_token_size,
        chunk_overlap_token_size,
        None
    )
    
    logger.info(f"생성된 청크 수: {len(chunks)}")
    return chunks

def dataset_into_chunks(corpus_path, chunk_max_token_size, chunk_overlap_token_size, samples):
    """데이터셋을 청크로 변환"""
    all_chunks = []
    all_chunk_position_dict = {}
    all_aux_chunks = []
    
    if not os.path.exists(corpus_path):
        logger.warning(f"Corpus path does not exist: {corpus_path}")
        return all_chunks, all_chunk_position_dict, all_aux_chunks
    
    for filename in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, filename)
        chunks = []
        aux_chunks = []
        chunk_position_dict = {}
        
        if os.path.isfile(file_path):
            if filename.endswith("corpus.json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    corpus = json.load(f)
                
                if chunk_max_token_size == None:
                    if 'text' in corpus[0].keys():
                        chunks = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
                    elif 'body' in corpus[0].keys():
                        chunks = [f"{doc['title']}\n{doc['body']}" for doc in corpus]
                else:
                    title_dict = {}
                    if 'text' in corpus[0].keys():
                        merged_doc = '\n\n'.join([f"{doc['title']}\n{doc['text']}" for doc in corpus])
                        before_docs_len = 0
                        for i, doc in enumerate(corpus):
                            start = before_docs_len
                            before_docs_len = before_docs_len + len(doc['title']) + len(doc['text']) + 3
                            if doc['title'] in title_dict.keys():
                                title_dict[doc['title']].append((start, before_docs_len))
                            else:
                                title_dict[doc['title']] = [(start, before_docs_len)]
                    
                    elif 'body' in corpus[0].keys():
                        merged_doc = '\n\n'.join([f"{doc['title']}\n{doc['body']}" for doc in corpus])
                        before_docs_len = 0
                        for i, doc in enumerate(corpus):
                            start = before_docs_len
                            before_docs_len = before_docs_len + len(doc['title']) + len(doc['body']) + 3
                            if doc['title'] in title_dict.keys():
                                title_dict[doc['title']].append((start, before_docs_len))
                            else:
                                title_dict[doc['title']] = [(start, before_docs_len)]
                    if corpus[0]['title'] == '':
                        title_dict = None
                    
                    chunks, chunk_position_dict = split_doc_into_chunk(merged_doc, chunk_max_token_size, chunk_overlap_token_size, title_dict)
                    aux_chunks, _ = split_doc_into_chunk(merged_doc, 50, 25, None)
            
            elif filename.endswith("corpus.txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    doc = f.read()
                    chunks, chunk_position_dict = split_doc_into_chunk(doc, chunk_max_token_size, chunk_overlap_token_size, None)
                    aux_chunks, _ = split_doc_into_chunk(doc, 50, 25, None)
        
        all_chunks.extend(chunks)
        all_chunk_position_dict = all_chunk_position_dict | chunk_position_dict
        all_aux_chunks.extend(aux_chunks)
    
    return all_chunks, all_chunk_position_dict, all_aux_chunks

# ========== 라이프사이클 이벤트 ==========

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 RAG 시스템 초기화"""
    global rag_system, indexed_chunks, startup_time #, rag_config
    
    startup_time = datetime.now()
    logger.info("🚀 Starting RAG_1 API Server...")
    
    # 환경 변수 로드
    load_dotenv()
    
    # API 키 확인 (Gemini 또는 OpenAI)
    gemini_key = os.getenv('GEMINI_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not gemini_key and not openai_key:
        logger.warning("⚠️  Neither GEMINI_API_KEY nor OPENAI_API_KEY is set. RAG system will not be initialized.")
        logger.info("Please set GEMINI_API_KEY or OPENAI_API_KEY in .env file and restart the server.")
        return
    
    # 적절한 API 키 설정
    if gemini_key and gemini_key != 'your_gemini_api_key_here':
        os.environ['GEMINI_API_KEY'] = gemini_key
        logger.info("✅ Using Gemini API")
    elif openai_key and openai_key != 'your_gemini_api_key_here':
        os.environ['OPENAI_API_KEY'] = openai_key
        logger.info("✅ Using OpenAI API")
    else:
        logger.warning("⚠️  API keys are set to placeholder values. RAG system will not be initialized.")
        return
    
    try:
        # 설정 로드
        method_name = os.getenv('RAG_METHOD', 'AtomRAG')
        dataset_name = os.getenv('RAG_DATASET', 'sillok')
        llm_model = os.getenv('RAG_LLM_MODEL', 'gemini-3-flash-preview')
        embedding_model = os.getenv('RAG_EMBEDDING_MODEL', 'gemini-embedding-001')
        llm_base_url = os.getenv('GEMINI_LLM_BASE_URL')  # Gemini doesn't use base_url typically
        embed_base_url = os.getenv('GEMINI_EMBED_BASE_URL')  # Gemini doesn't use base_url typically
        retrieval_top_k = int(os.getenv('RAG_RETRIEVAL_TOP_K', 200))
        max_context_tokens = int(os.getenv('RAG_MAX_CONTEXT_TOKENS', 5400))
        chunk_max_token_size = int(os.getenv('RAG_CHUNK_SIZE', 1200))
        chunk_overlap_ratio = float(os.getenv('RAG_CHUNK_OVERLAP_RATIO', 0.5))
        chunk_overlap_token_size = int(chunk_max_token_size * chunk_overlap_ratio)
        save_dir = os.getenv('RAG_OUTPUT_DIR', 'outputs')
        
        logger.info(f"📝 Config: method={method_name}, dataset={dataset_name}, llm={llm_model}, embedding={embedding_model}")
        
        # BaseConfig 초기화
        rag_config = BaseConfig(
            method_name=method_name,
            dataset=dataset_name,
            llm_name=llm_model,
            embedding_model_name=embedding_model,
            llm_base_url=llm_base_url,
            embedding_base_url=embed_base_url,
            retrieval_top_k=retrieval_top_k,
            max_context_tokens=max_context_tokens,
            chunk_max_token_size=chunk_max_token_size,
            chunk_overlap_token_size=chunk_overlap_token_size,
            save_dir=save_dir,
            embedding_batch_size=128,
            max_qa_steps=3,
            qa_top_k=10,
            graph_type="dpr_only",
            max_new_tokens=None,
            corpus_len=0,
            additional_module='None'
        )
        
        # RAG 시스템 초기화
        # if method_name == 'naive':
        #     rag_system = NaiveRAG(global_config=rag_config)
        # elif method_name == 'frag':
        #     rag_system = FRAG(global_config=rag_config)
        if method_name == 'AtomRAG':
            cls = dataset_name
            WORKING_DIR = f"../{cls}"
            rag_system = AtomRAG(working_dir=WORKING_DIR, embedding_func=gemini_embed(model = embedding_model), llm_model_func=gemini_complete(model = llm_model),)
        else:
            raise ValueError(f"Invalid method_name: {method_name}")
        
        logger.info(f"✅ RAG system ({method_name}) initialized")
        
        # 데이터셋 로딩 및 인덱싱
        # data_dir = os.getenv('RAG_DATA_DIR', '/workspace/RAG_1/dataset')
        
        # sillok 데이터셋 처리
        # if dataset_name == 'sillok':
        #     sillok_file = os.path.join(data_dir, 'sillok', 'sillok_merged.txt')
            
        #     if os.path.exists(sillok_file):
        #         logger.info(f"📚 Loading sillok dataset from {sillok_file}...")
                
        #         # 텍스트 파일을 청크로 변환
        #         chunks = text_file_into_chunks(
        #             sillok_file,
        #             chunk_max_token_size,
        #             chunk_overlap_token_size
        #         )
                
        #         if chunks:
        #             logger.info(f"📦 Created {len(chunks)} chunks")
        #             indexed_chunks = chunks
                    
        #             logger.info("🔄 Indexing documents... (this may take a while)")
        #             start_time = time.time()
        #             rag_system.index(chunks, [])
        #             elapsed = time.time() - start_time
        #             logger.info(f"✅ Documents indexed successfully in {elapsed:.2f} seconds")
        #         else:
        #             logger.warning("⚠️  No chunks created from sillok file")
        #     else:
        #         logger.warning(f"⚠️  Sillok file not found: {sillok_file}")
        # else:
        #     # 기존 JSON 데이터셋 처리
        #     corpus_path = os.path.join(data_dir, dataset_name)
            
        #     if os.path.exists(corpus_path):
        #         logger.info(f"📚 Loading dataset from {corpus_path}...")
                
        #         dataset_file = os.path.join(corpus_path, f"{dataset_name}.json")
        #         if os.path.exists(dataset_file):
        #             samples = json.load(open(dataset_file, "r"))
        #             logger.info(f"📊 Loaded {len(samples)} samples")
        #         else:
        #             samples = None
                
        #         # 문서 청킹
        #         chunks, chunk_position_dict, aux_chunks = dataset_into_chunks(
        #             corpus_path,
        #             chunk_max_token_size,
        #             chunk_overlap_token_size,
        #             samples
        #         )
                
        #         if chunks:
        #             logger.info(f"📦 Created {len(chunks)} chunks")
        #             indexed_chunks = chunks
        #             rag_system.index(chunks, aux_chunks if aux_chunks else [])
        #             logger.info("✅ Documents indexed successfully")
        #         else:
        #             logger.warning("⚠️  No documents found to index")
        #     else:
        #         logger.warning(f"⚠️  Dataset path not found: {corpus_path}")
        
        logger.info("✅ RAG_1 API Server initialization complete")
        logger.info(f"🌐 Server ready at http://0.0.0.0:8000")
        logger.info(f"📖 API Docs at http://0.0.0.0:8000/docs")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {str(e)}")
        logger.exception(e)
        rag_system = None

# ========== API 엔드포인트 ==========

## 서버가 살아있는지 확인 용도
@app.get("/", tags=["Root"])
async def root():
    """루트 엔드포인트"""
    return {
        "service": "RAG_1 API Server",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

## 서버 상태 점검 용도
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """헬스체크 엔드포인트"""
    uptime = None
    if startup_time:
        uptime = (datetime.now() - startup_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if rag_system else "unhealthy",
        rag_initialized=rag_system is not None,
        method=rag_config.method_name if rag_config else None,
        dataset=rag_config.dataset if rag_config else None,
        indexed_docs=len(indexed_chunks),
        uptime_seconds=uptime
    )

@app.post("/api/rag/query", response_model=QueryResponse, tags=["RAG"])
async def rag_query(request: QueryRequest):
    """
    전체 RAG 파이프라인 실행 (검색 + 생성)
    
    - **query**: 사용자 질문
    - **top_k**: 검색할 문서 수 (기본: 5)
    - **method**: RAG 방법 (naive/frag, 기본: naive)
    - **max_context_tokens**: 최대 컨텍스트 토큰 (기본: 5400)
    - **is_short_answer**: 짧은 답변 여부 (기본: True)
    """
    
    ## RAG system이 만들어졌는지 확인
    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Check server logs."
        )
        
    sources: List[SourceDocument] = []
    
    try:
        start_time = time.time()
        logger.info(f"📥 Query received: {request.query[:100]}...")
        
        query_param = QueryParam(
            mode="ours_experiment1",
            query_mode="experiment3",
            top_mode=7,
            addon_params={"embedding_func": rag_system.embedding_func},
        )
            
        # RAG 쿼리 실행
        queries_solutions, queries_retrieves= await rag_system.aquery(
            query=request.query,
            param=query_param,
        )
        
        # 결과 추출
        if queries_solutions and len(queries_solutions) > 0:
            query_solution = queries_solutions
            answer = query_solution
            retrieved_docs = queries_retrieves
            
            # 소스 문서 파싱
            sources = []
            for doc in retrieved_docs[:request.top_k]:
                sources.append(SourceDocument(
                    title=doc.get("chunk_id", ""),                          # chunk_id
                    content=(doc.get("chunk_text", "") or "")[:500],        # chunk_text (500자 제한)
                    score=float(doc.get("max_atomic_score", 0.0)),          # max_atomic_score
                ))
        else:
            answer = "No response generated."
            sources = []
        
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"✅ Query completed in {total_time:.2f}s")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            metadata={
                "total_time": total_time,
                "method": request.method,
                "top_k": request.top_k,
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Query failed: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/retrieve", response_model=RetrieveResponse, tags=["RAG"])
async def rag_retrieve(request: RetrieveRequest):
    """
    문서 검색만 수행 (생성 없음)
    
    - **query**: 검색 쿼리
    - **top_n**: 반환할 문서 수 (기본: 5)
    """
    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Check server logs."
        )
    
    try:
        start_time = time.time()
        logger.info(f"🔍 Retrieve request: {request.query[:100]}...")
        
        # 문서 검색 실행
        retrieval_results = rag_system.retrieve(
            queries=[request.query],
            num_to_retrieve=request.top_n
        )
        
        # 결과 추출
        documents = []
        if retrieval_results and len(retrieval_results) > 0:
            docs = retrieval_results[0].docs if hasattr(retrieval_results[0], 'docs') else []
            scores = retrieval_results[0].doc_scores if hasattr(retrieval_results[0], 'doc_scores') else []
            
            for idx, doc in enumerate(docs):
                lines = doc.split('\n', 1)
                title = lines[0].strip() if lines else doc[:50]
                content = lines[1].strip() if len(lines) > 1 else doc
                score = float(scores[idx]) if idx < len(scores) else 0.0
                
                documents.append(SourceDocument(
                    title=title,
                    content=content,
                    score=score
                ))
        
        end_time = time.time()
        retrieval_time = end_time - start_time
        
        logger.info(f"✅ Retrieved {len(documents)} documents in {retrieval_time:.2f}s")
        
        return RetrieveResponse(
            documents=documents,
            retrieval_time=retrieval_time
        )
        
    except Exception as e:
        logger.error(f"❌ Retrieve failed: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

# ========== 메인 ==========

if __name__ == "__main__":
    port = int(os.getenv('RAG_PORT', 8000))
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
