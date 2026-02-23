# AtomRAG: AI Coding Instructions

## Project Overview
AtomRAG is a **graph-based Retrieval-Augmented Generation (RAG) framework** that combines:
- **Knowledge Graph (KG)** construction from documents
- **Multi-hop reasoning** via Chain-of-Thought expansion
- **Multiple retrieval modes**: naive, local, global, hybrid, mix
- **LLM integration** with pluggable backends (OpenAI, Ollama, LMDeploy, Bedrock, etc.)

## Core Architecture

### Main Components
1. **`AtomRAG.AtomRAG` class** ([AtomRAG/AtomRAG.py](AtomRAG/AtomRAG.py)) - Central orchestrator
   - Manages document insertion, KG construction, storage backends
   - Coordinates queries across retrieval modes
   - Configurable with `QueryParam` for mode selection and output control

2. **Storage Layer** ([AtomRAG/kg/](AtomRAG/kg/)) - Pluggable backends
   - Graph storage: NetworkX, Neo4j, PostgreSQL, TiDB, Gremlin
   - Vector storage: NanoVectorDB, Faiss, Chroma, Qdrant, Milvus
   - KV storage: JSON, Redis, MongoDB, PostgreSQL
   - Default: NetworkX + NanoVectorDB + JSON

3. **Operation Layer** ([AtomRAG/operate.py](AtomRAG/operate.py)) - Query execution
   - `chunking_by_token_size()` - Document tokenization (GPT-4o tokens)
   - `extract_entities*()` - Multi-variant entity extraction (experiment0-3)
   - `kg_query*()` - KG-based retrieval with multiple strategies
   - `mix_kg_vector_query()` - Hybrid retrieval combining both modalities

4. **LLM Abstraction** ([AtomRAG/llm/](AtomRAG/llm/))
   - Provider modules: openai.py, ollama.py, gemini.py, groqapi.py, etc.
   - `Model` and `MultiModel` classes for load balancing ([AtomRAG/llm.py](AtomRAG/llm.py))
   - Async embedding functions required via `EmbeddingFunc` dataclass

5. **REST API Server** ([AtomRAG/api/AtomRAG_server.py](AtomRAG/api/AtomRAG_server.py))
   - FastAPI endpoints for document management and querying
   - Stream support, batch processing, file upload handlers
   - Query prefix modes: /local, /global, /hybrid, /naive, /mix, /bypass

## Critical Data Flows

### Document Insertion → KG Construction
```
insert_text(content) → chunking_by_token_size()
  ↓
extract_entities_experiment{0,1,2}() [extracts triples/atomics]
  ↓
embed content → store in Vector DB
  ↓
build knowledge graph edges → store in Graph DB
```

### Query Execution (Varies by Mode)
```
QueryParam(mode, query_mode) → ours_kg_query_experiment{0,1,2,3}()
  ↓
[Graph traversal + entity linking + vector similarity]
  ↓
rank_context_by_tokens() → top_k results
  ↓
llm_model_func(prompt) → response
```

## Key Configuration & Patterns

### QueryParam Modes (Required Understanding)
- **naive**: Simple vector similarity, no KG
- **local**: Context-dependent (entity/relation filtering)
- **global**: Global knowledge exploitation
- **hybrid**: local + global combined
- **mix**: KG queries + vector retrieval fusion

### Experiment Variants
Projects test multiple extraction/query strategies:
- **experiment0/1/2/3**: Different entity extraction methods
- Results stored in `pred/` directory with naming: `{mode}_{dataset}_result.json`
- Evaluation via `eval_llm.py` using LLM judge (GPT-4o-mini)

### Configuration Files
- `config.ini.example`: Database URIs (Neo4j, MongoDB, Qdrant, Redis)
- `.env`: OpenAI API keys and model selections
- Working directories: auto-created as `AtomRAG_cache_{timestamp}`

## Developer Workflows

### Setup & Installation
```bash
pip install -e .  # Editable install with dependencies
export OPENAI_API_KEY='...'  # Required for most operations
```

### Document Indexing (Reproduce Pipeline)
```bash
python reproduce/Step_0.py  # Extract document contexts
python reproduce/Step_1.py  # Build knowledge graph
python reproduce/Step_2.py  # Generate question files
python reproduce/Step_3.py  # Run AtomRAG queries
```

### Evaluation Workflow
```bash
python eval_llm.py       # Async LLM-based answer evaluation
python judge_F1.py       # Compute F1/EM/Precision/Recall
python eval_metrics.py   # Aggregate results across datasets
```

### API Server
```bash
AtomRAG-server --llm-binding openai --embedding-binding openai
# Or with Ollama: --llm-binding ollama --embedding-binding ollama
```

## Common Patterns & Conventions

### Async Pattern
All LLM calls and embeddings are **async**:
```python
embedding_func = EmbeddingFunc(
    embedding_dim=1536,
    max_token_size=8192,
    func=openai_embed  # async function
)
```

### Token Management
Uses **tiktoken (GPT-4o encoder)** for token counting:
- Chunking: default 1024 max_token_size, 128 overlap
- Truncation: `truncate_list_by_token_size()` limits context
- Embedding: max_token_size enforced per-document

### Caching & Performance
- Embedding cache: configurable with similarity_threshold
- LLM response caching via MD5 hash of prompt + arguments
- Async semaphore limiting: prevents rate limit crashes

### Dataset Support
Standardized workflows for:
- HotpotQA, MultiHopRAG, MuSiQue, 2WikiMultiHopQA, TriviaQA, NQ
- Question field variations handled: `question`, `query`
- Gold answer formats: string or list

## Integration Points

### External Services
- **OpenAI**: Default LLM + embeddings
- **Ollama**: Local LLM alternative (docker-compose.yml provided)
- **Gemini, Claude, Groq**: Provider-specific integrations
- **Database**: Neo4j, PostgreSQL, MongoDB for scalable graph storage

### OpenWebUI Integration
- `external_bindings/OpenWebuiTool/openwebui_tool.py` - Tool plugin
- Supports mode selection, entity limiting, debug modes

## When Modifying Core Components

### Adding New Query Mode
1. Implement `ours_kg_query_experiment{N}()` in `operate.py`
2. Add test case in `reproduce/Step_3.py` with `query_mode` parameter
3. Update `base.py` QueryParam docstring

### Adding Storage Backend
1. Implement abstract methods from `BaseGraphStorage`, `BaseVectorStorage`, `BaseKVStorage`
2. Register in STORAGES dict in `AtomRAG.py`
3. Add config section in `config.ini.example`

### LLM Provider Changes
1. Create provider module in `llm/` following `llm/openai.py` pattern
2. Implement async functions: `{provider}_model_complete()`, `{provider}_embed()`
3. Update MultiModel if load balancing needed

## Testing & Validation

- **Unit evaluation**: `judge_F1.py` for standard metrics
- **LLM evaluation**: `eval_llm.py` for quality assessment (CORRECT/INCORRECT verdicts)
- **Visual inspection**: Graph visualizer in `tools/lightrag_visualizer/`
- **Integration test**: `test.py`, `test_AtomRAG_ollama_chat.py`

---
**Quick Links**: [Main Readme](README.md) | [API Documentation](AtomRAG/api/README.md) | [Installation](install.sh)
