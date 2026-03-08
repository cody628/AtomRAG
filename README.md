# AtomRAG

Atomic Fact–based Retrieval-Augmented Generation (AtomRAG) system.

This repository provides:

- 🔍 Atomic fact–level retrieval
- 🧠 Gemini-based answer generation
- 🖥️ FastAPI RAG inference server
- 📚 Sillok (조선왕조실록) dataset support

---

# 🐳 1. Run with Docker (Recommended)
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 --volume ~/workspace:/workspace -it --rm --name AtomicRAG pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# 📦 2. Install Dependencies
apt update
apt install -y git
git --version
git clone https://github.com/cody628/AtomRAG.git
cd AtomRAG/AtomRAG
pip install -r requirements.txt

# 🔑 3. Set API Key
export GEMINI_API_KEY="YOUR_API_KEY"

# 📚 4. Dataset Location & DB construction for BM25
/workspace/AtomRAG/sillok

python build_atomic_bm25.py

# 🧠 5. Single Query Inference (CLI)
query_text = "태종이 회암사 승려의 불사(佛事)를 문제 삼지 말라고 한 이유는 무엇인가?"
python reproduce/Step_3.py

# 🌐 5. Run FastAPI Server
python api_server.py

apt update
apt install -y curl

curl -v --max-time 120 \
  -X POST "http://127.0.0.1:8000/api/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
        "query": "태종이 회암사 승려의 불사(佛事)를 문제 삼지 말라고 한 이유는 무엇인가?",
        "top_k": 5,
        "method": "AtomRAG",
        "max_context_tokens": 5400,
        "is_short_answer": true
      }'








