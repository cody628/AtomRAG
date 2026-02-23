from dataclasses import dataclass
from typing import Optional

@dataclass
class BaseConfig:
    # ===== 기본 설정 =====
    method_name: str
    dataset: str

    # ===== LLM / Embedding =====
    llm_name: str
    embedding_model_name: str
    llm_base_url: Optional[str]
    embedding_base_url: Optional[str]

    # ===== Retrieval =====
    retrieval_top_k: int
    qa_top_k: int

    # ===== Context =====
    max_context_tokens: int
    max_new_tokens: Optional[int]

    # ===== Chunking =====
    chunk_max_token_size: int
    chunk_overlap_token_size: int

    # ===== Embedding =====
    embedding_batch_size: int

    # ===== QA / Graph =====
    max_qa_steps: int
    graph_type: str
    
    # ===== Dataset =====
    corpus_len: int

    # ===== Output =====
    save_dir: str

    # ===== Optional =====
    additional_module: Optional[str] 
