import asyncio
import re
import json
from src.AtomRAG import AtomRAG, QueryParam
from src.AtomRAG.llm.openai import gpt_4o_mini_complete, openai_embed
from src.AtomRAG.llm.gemini import gemini_complete, gemini_embed

async def process_query(query_text, rag_instance, query_param):
    try:
        result, retrieves = await rag_instance.aquery(query_text, param=query_param)
        print(retrieves)
        return {"query": query_text, "result": result}, None
    except Exception as e:
        return None, {"query": query_text, "error": str(e)}

async def main():
    cls = "sillok1"
    query_mode = "experiment3"
    mode = "ours_experiment1"
    top = 7
    query_text = "1412년 5월 경상도에 어떤 자연재해가 발생하였는가?"
    WORKING_DIR = f"../{cls}"

    rag = AtomRAG(
        working_dir=WORKING_DIR,
        embedding_func=gemini_embed,
        llm_model_func=gemini_complete,
    )

    query_param = QueryParam(
        mode=mode,
        query_mode=query_mode,
        top_mode=top,
        addon_params={"embedding_func": rag.embedding_func},
    )

    ok, err = await process_query(query_text, rag, query_param)
    if err:
        print("ERROR:", err)
    else:
        print("OK:", ok)

if __name__ == "__main__":
    asyncio.run(main())



# import re
# import json
# import asyncio
# from src.AtomRAG import AtomRAG, QueryParam
# from src.AtomRAG.llm.openai import gpt_4o_mini_complete, openai_embed
# from src.AtomRAG.llm.groqapi import groq_llama_4_scout_complete
# from src.AtomRAG.llm.ollama import ollama_model_complete
# from src.AtomRAG.llm.ollama import ollama_embed
# from src.AtomRAG.llm.gemini import gemini_complete
# from src.AtomRAG.utils import EmbeddingFunc
# from tqdm import tqdm


# # =========================================================
# # Query Extraction Functions
# # =========================================================
# def extract_queries(file_path):
#     with open(file_path, "r") as f:
#         data = f.read()
#     data = data.replace("**", "")
#     queries = re.findall(r" - Question \d+: (.+)", data)
#     return queries


# def extract_queries_novelqa(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     return [v["Question"] for v in data.values() if v.get("Question")]


# def extract_queries_infinite(file_path):
#     queries = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             item = json.loads(line)
#             if question := item.get("input"):
#                 queries.append(question)
#     return queries


# def extract_queries_hotpot(file_path):
#     queries = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             item = json.loads(line)
#             if question := item.get("question"):
#                 queries.append(question)
#     return queries


# def extract_queries_musique(file_path):
#     queries = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             item = json.loads(line)
#             if question := item.get("question"):
#                 queries.append(question.strip())
#     return queries


# def extract_queries_2wikimultihopqa(file_path):
#     queries = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             if not line.strip():
#                 continue
#             item = json.loads(line)
#             if question := item.get("question"):
#                 queries.append(question.strip())
#     return queries


# def extract_queries_triviaqa(file_path):
#     queries = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             if not line.strip():
#                 continue
#             item = json.loads(line)
#             if question := item.get("question"):
#                 queries.append(question.strip())
#     return queries


# def extract_queries_multihoprag(file_path):
#     queries = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             item = json.loads(line)
#             if question := item.get("query"):
#                 queries.append(question)
#     return queries


# # =========================================================
# # Async sequential processing logic
# # =========================================================
# async def process_query(query_text, rag_instance, query_param):
#     """단일 쿼리 직렬 실행"""
#     try:
#         result = await rag_instance.aquery(query_text, param=query_param)
#         return {"query": query_text, "result": result}, None
#     except Exception as e:
#         return None, {"query": query_text, "error": str(e)}


# async def process_queries_sequentially(queries, rag_instance, query_param):
#     """❗ 완전 직렬 실행 (한 query씩 순서대로 처리)"""
#     results = []
#     errors = []

#     for q in tqdm(queries, desc="Processing queries", unit="query"):
#         result, error = await process_query(q, rag_instance, query_param)
#         if result:
#             results.append(result)
#         else:
#             errors.append(error)

#     return results, errors


# def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
#     try:
#         return asyncio.get_event_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         return loop


# def run_queries_and_save_to_json_sequential(
#     start_idx, end_idx, queries, rag_instance, query_param, output_file, error_file
# ):
#     """직렬 처리 + 결과 저장"""
#     queries = queries[start_idx:end_idx]
#     loop = always_get_an_event_loop()

#     results, errors = loop.run_until_complete(
#         process_queries_sequentially(queries, rag_instance, query_param)
#     )

#     # 결과 저장
#     with open(output_file, "w", encoding="utf-8") as rf:
#         json.dump(results, rf, ensure_ascii=False, indent=4)

#     with open(error_file, "w", encoding="utf-8") as ef:
#         json.dump(errors, ef, ensure_ascii=False, indent=4)

#     print(f"✅ Saved {len(results)} results, {len(errors)} errors to disk.")


# # =========================================================
# # Main Execution
# # =========================================================
# if __name__ == "__main__":
#     cls = "multihoprag"
#     query_mode = "experiment3"
#     mode = "ours_experiment1"
#     start_idx = 0
#     end_idx = 1000
#     WORKING_DIR = f"../{cls}"

#     rag = AtomRAG(
#         working_dir=WORKING_DIR,
#         embedding_func=openai_embed,
#         llm_model_func=gpt_4o_mini_complete,
#     )

#     query_param = QueryParam(
#         mode=mode,
#         query_mode=query_mode,
#         addon_params={"embedding_func": rag.embedding_func},
#     )

#     # dataset별 query 로드
#     if cls == "novelqa":
#         queries = extract_queries_novelqa(f"../datasets/{cls}.json")
#     elif cls in ("infiniteqa", "infinitechoice"):
#         queries = extract_queries_infinite(f"../datasets/{cls}.jsonl")
#     elif cls == "hotpot":
#         queries = extract_queries_hotpot(f"../datasets/{cls}.jsonl")
#     elif cls == "multihoprag":
#         queries = extract_queries_multihoprag(f"../datasets/{cls}.jsonl")
#     elif cls == "musique":
#         queries = extract_queries_musique(f"../datasets/{cls}.jsonl")
#     elif cls == "2wikimultihopqa":
#         queries = extract_queries_2wikimultihopqa(f"../datasets/{cls}.jsonl")
#     elif cls == "triviaqa":
#         queries = extract_queries_triviaqa(f"../datasets/{cls}.jsonl")
#     else:
#         queries = extract_queries(f"../datasets/questions/{cls}_questions.txt")

#     # 직렬 실행
#     run_queries_and_save_to_json_sequential(
#         start_idx, end_idx, queries, rag, query_param,
#         f"{mode}_{cls}_result.json", f"{mode}_{cls}_errors.json"
#     )
