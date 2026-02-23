import asyncio
import tiktoken
import json
import math
import random
import re
import copy
from tqdm.asyncio import tqdm as tqdm_async
from sklearn.preprocessing import normalize
from typing import Any, Union
import heapq
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    statistic_data,
    get_conversation_turns,
    parse_atomic_record,
    parse_atomic_record_experiment1,
    parse_triple_record_experiment2,
    parse_triple_record,
    EmbeddingFunc,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
    TextAtomicSchema,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
import time
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import networkx as nx
from itertools import product
from itertools import islice
from collections import OrderedDict
import itertools
import os
from src.AtomRAG.llm.openai import openai_embed

def chunking_by_token_size(
    content: str,
    split_by_character: Union[str, None] = None,
    split_by_character_only: bool = False,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = encode_string_by_tiktoken(chunk, model_name=tiktoken_model)
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = encode_string_by_tiktoken(chunk, model_name=tiktoken_model)
                if len(_tokens) > max_token_size:
                    for start in range(
                        0, len(_tokens), max_token_size - overlap_token_size
                    ):
                        chunk_content = decode_tokens_by_tiktoken(
                            _tokens[start : start + max_token_size],
                            model_name=tiktoken_model,
                        )
                        new_chunks.append(
                            (min(max_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
        ):
            chunk_content = decode_tokens_by_tiktoken(
                tokens[start : start + max_token_size], model_name=tiktoken_model
            )
            results.append(
                {
                    "tokens": min(max_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """Handle entity relation summary
    For each entity or relation, input is the combined description of already existing description and new description.
    If too long, use LLM to summarize.
    """
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
        metadata={"created_at": time.time()},
    )


async def atomic_merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    already_source_ids = []
    already_atomic_ids = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    
    if already_node is not None:
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_atomic_ids.extend(
            split_string_by_multi_markers(already_node["atomic_id"], [GRAPH_FIELD_SEP])
        )

    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    
    atomic_id = GRAPH_FIELD_SEP.join(
        set([dp["atomic_id"] for dp in nodes_data] + already_atomic_ids)
    )
    
    node_data = dict(
        source_id=source_id,
        atomic_id=atomic_id,
    )
    
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data

async def triple_merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    already_source_ids = []
    already_triple_ids = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    
    if already_node is not None:
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_triple_ids.extend(
            split_string_by_multi_markers(already_node["triple_id"], [GRAPH_FIELD_SEP])
        )

    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    
    triple_id = GRAPH_FIELD_SEP.join(
        set([dp["triple_id"] for dp in nodes_data] + already_triple_ids)
    )
    
    node_data = dict(
        source_id=source_id,
        triple_id=triple_id,
    )
    
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def atomic_merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_atomic_ids = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_atomic_ids.extend(
            split_string_by_multi_markers(already_edge["atomic_id"], [GRAPH_FIELD_SEP])
        )

    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    atomic_id = GRAPH_FIELD_SEP.join(
        set([dp["atomic_id"] for dp in edges_data] + already_atomic_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "atomic_id": atomic_id,
                    "source_id": source_id,
                    "description": description,
                },
            )
    
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            description=description,
            source_id=source_id,
            atomic_id=atomic_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        atomic_id=atomic_id,
        description=description,
    )

    return edge_data

async def triple_merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_triple_ids = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_triple_ids.extend(
            split_string_by_multi_markers(already_edge["triple_id"], [GRAPH_FIELD_SEP])
        )

    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    triple_id = GRAPH_FIELD_SEP.join(
        set([dp["triple_id"] for dp in edges_data] + already_triple_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "triple_id": triple_id,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    # description = await _handle_entity_relation_summary(
    #     f"({src_id}, {tgt_id})", description, global_config
    # )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            description=description,
            source_id=source_id,
            triple_id=triple_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        triple_id=triple_id,
        description=description,
    )

    return edge_data
    
def convert_entities(all_entities_data: list[dict]) -> list[tuple[str, dict]]:
    converted = []
    for item in all_entities_data:
        atomic_id = item["atomic_id"]
        atomic_data = {
            "source_id": item.get("source_id"),
            "atomic_id": atomic_id,
            "entity_name": item.get("entity_name"),
        }
        converted.append((atomic_id, atomic_data))
    return converted

def batch_atomic_facts(ordered_atomics, batch_size=30):
    atomic_batches = []
    batch = []
    for chunk_id, atomics in ordered_atomics.items():
        for atomic_id, atomic_data in atomics:
            batch.append((atomic_id, atomic_data))
            if len(batch) == batch_size:
                atomic_batches.append(batch)
                batch = []
    if batch:
        atomic_batches.append(batch)
    return atomic_batches

async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    atomic_entity_vdb: BaseVectorStorage,
    triple_entity_vdb: BaseVectorStorage,
    global_config: dict,
    llm_response_cache: BaseKVStorage = None,
) -> Union[BaseGraphStorage, None]:
  # 1. Chunk -> Atomic
    ## Basic Setting
    use_llm_func: callable = global_config["llm_model_func"]
    enable_llm_cache_for_entity_extract: bool = global_config[
        "enable_llm_cache_for_entity_extract"
    ]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    continue_prompt = PROMPTS["atomicfact_continue_extraction"]

    ## Chunk Prepare
    ordered_chunks = list(chunks.items())

    ## Chunk -> Atomic Prompt
    cls = os.path.basename(global_config["working_dir"])
    
    atomic_examples = "\n".join(PROMPTS[f"atomic_entity_extraction_examples_experiment1_{cls}"])
           

    atomic_example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        language=language,
    )
    atomic_examples = atomic_examples.format(**atomic_example_context_base)

    if cls == "sillok":
        atomic_entity_extract_prompt = PROMPTS[f"atomic_entity_extraction_experiment1_{cls}"]
    else:
        atomic_entity_extract_prompt = PROMPTS[f"atomic_entity_extraction_experiment1"]

    atomic_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples=atomic_examples,
        language=language,
    )

    already_processed = 0
    already_entities = 0

    ## LLM Function
    async def _user_llm_func_with_cache(
        input_text: str, history_messages: list[dict[str, str]] = None
    ) -> str:
        if enable_llm_cache_for_entity_extract and llm_response_cache:
            if history_messages:
                history = json.dumps(history_messages, ensure_ascii=False)
                _prompt = history + "\n" + input_text
            else:
                _prompt = input_text

            arg_hash = compute_args_hash(_prompt)
            cached_return, _1, _2, _3 = await handle_cache(
                llm_response_cache,
                arg_hash,
                _prompt,
                "default",
                cache_type="extract",
                force_llm_cache=True,
            )
            if cached_return:
                logger.debug(f"Found cache for {arg_hash}")
                statistic_data["llm_cache"] += 1
                return cached_return

            statistic_data["llm_call"] += 1
            if history_messages:
                res: str = await use_llm_func(input_text, history_messages=history_messages)
            else:
                res: str = await use_llm_func(input_text)

            await save_to_cache(
                llm_response_cache,
                CacheData(
                    args_hash=arg_hash,
                    content=res,
                    prompt=_prompt,
                    cache_type="extract",
                ),
            )
            return res

        if history_messages:
            return await use_llm_func(input_text, history_messages=history_messages)
        else:
            return await use_llm_func(input_text)

    ## Chunk -> Atomic Function
    async def atomic_process_single_content(chunk_key_dp: tuple[str, "TextChunkSchema"]):
        nonlocal already_processed, already_entities  # ✅ atomic 단계에는 already_relations 필요 없음

        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        ### Final Prompt
        atomic_hint_prompt = atomic_entity_extract_prompt.format(
            **atomic_context_base, input_text="{input_text}"
        ).format(**atomic_context_base, input_text=content)

        ### Cache Type LLM
        final_result = await _user_llm_func_with_cache(atomic_hint_prompt)

        ### Retry LLM Prompt
        history = pack_user_ass_to_openai_messages(atomic_hint_prompt, final_result)

        ### Retry LLM execution
        glean_result = await _user_llm_func_with_cache(
            continue_prompt, history_messages=history
        )

        if glean_result and glean_result.strip():
            final_result += "\n" + glean_result.strip()

        ### Store LLM Response to records
        records = split_string_by_multi_markers(
            final_result,
            [
                atomic_context_base["record_delimiter"],
                atomic_context_base["completion_delimiter"],
            ],
        )

        ### LLM Response -> Atomic Facts
        maybe_nodes = defaultdict(list)

        for record in records:
            parsed = parse_atomic_record_experiment1(
                record, atomic_context_base["tuple_delimiter"]
            )
            if parsed is None:
                continue

            atomic_fact = parsed
            atomic_fact_id = compute_mdhash_id(atomic_fact, prefix="atom-")

            maybe_nodes[atomic_fact].append(
                {
                    "atomic_id": atomic_fact_id,
                    "source_id": chunk_key,
                }
            )

        already_processed += 1
        already_entities += len(maybe_nodes)
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        logger.debug(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated)\r",
        )

        return dict(maybe_nodes)

    ## Extracting Atomic Facts
    results = []
    for result in tqdm_async(
        asyncio.as_completed([atomic_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks),
        desc="Level 2 - Atomic Extracting entities",
        unit="chunk",
        position=1,
        leave=False,
    ):
        results.append(await result)

    ## Atomic aggregation
    maybe_nodes = defaultdict(list)
    for m_nodes in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)

    logger.debug("Atomic Inserting entities into storage...")

    aggregated_atomic = {}  # atomic_id -> {"entity_name": str, "source_ids": set([...])}

    for k, v_list in maybe_nodes.items():
        for v in v_list:
            atomic_id = v["atomic_id"]
            entity_name = k  # 여기서는 atomic_fact 문자열
            source_id = v["source_id"]

            if atomic_id not in aggregated_atomic:
                aggregated_atomic[atomic_id] = {
                    "entity_name": entity_name,
                    "source_ids": {source_id},
                }
            else:
                aggregated_atomic[atomic_id]["source_ids"].add(source_id)

    all_entities_data = [
        {
            "atomic_id": aid,
            "entity_name": data["entity_name"],
            "source_ids": list(data["source_ids"]),
        }
        for aid, data in aggregated_atomic.items()
    ]

    ## Atomic Nodes Store VDB
    if atomic_entity_vdb is not None:
        data_for_vdb = {
            dp["atomic_id"]: {
                "content": dp["entity_name"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await atomic_entity_vdb.upsert(data_for_vdb)

    ## Atomic text store KV
    # --- operate.py: extract_entities_experiment2 내부, "Atomic text store KV" 구간을 아래로 교체 ---

    # ✅ Atomic text store KV (append-only JSONL to avoid full read/write)
    # 기존: kv_store_text_atomics.json 전체 load -> update -> dump
    # 변경: kv_store_text_atomics.{run_tag}.jsonl 에 append (O(1))

    run_tag = global_config.get("addon_params", {}).get("run_tag")
    if not run_tag:
        run_tag = f"pid{os.getpid()}"

    kv_store_file = os.path.join(
        global_config["working_dir"],
        f"kv_store_text_atomics.jsonl",
    )

    # jsonl 한 줄 = 한 atomic_id 레코드 (나중에 merge 시 atomic_id 기준으로 source_ids union)
    # NOTE: ensure_ascii=False 유지
    with open(kv_store_file, "a", encoding="utf-8") as f:
        for dp in all_entities_data:
            rec = {
                "atomic_id": dp["atomic_id"],
                "content": dp["entity_name"],
                "source_ids": dp.get("source_ids", []),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        f"Appended {len(all_entities_data)} atomic entities to {os.path.basename(kv_store_file)}"
    )

    return None
    

def get_options_by_query(json_path, query):
    # JSON 파일 불러오기
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 질문 매칭
    for q_id, q_data in data.items():
        if q_data['Question'].strip() == query.strip():  # 공백 제거 후 완전 일치 비교
            return q_data['Options']

    # 매칭 실패 시 None 반환
    return None

def get_options_by_query_infinite(jsonl_path, query):
    options = None
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item['input'].strip() == query.strip():  # 공백 제거 후 완전 일치 비교
                options = item['options']
                break

    return options  # 없으면 None 반환

async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
    prompt: str = "",
) -> str:
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    keyword_start_time = time.perf_counter()
    
    # Extract keywords using extract_keywords_only function which already supports conversation history
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )
    
    keyword_end_time = time.perf_counter()
    keyword_elapsed = keyword_end_time - keyword_start_time
    print(f"[TIME] Keyword extraction took {keyword_elapsed:.4f} seconds.")
    
    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # Handle empty keywords
    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return PROMPTS["fail_response"]
    if ll_keywords == [] and query_param.mode in ["local", "hybrid"]:
        logger.warning(
            "low_level_keywords is empty, switching from %s mode to global mode",
            query_param.mode,
        )
        query_param.mode = "global"
    if hl_keywords == [] and query_param.mode in ["global", "hybrid"]:
        logger.warning(
            "high_level_keywords is empty, switching from %s mode to local mode",
            query_param.mode,
        )
        query_param.mode = "local"

    ll_keywords = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords = ", ".join(hl_keywords) if hl_keywords else ""

    logger.info("Using %s mode for query processing", query_param.mode)

    # Build context
    keywords = [ll_keywords, hl_keywords]
    
    retrieval_start_time = time.perf_counter()
    
    context = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
    )
    
    retrieval_end_time = time.perf_counter()
    retrieval_elapsed = retrieval_end_time - retrieval_start_time
    print(f"[TIME] Retrieval took {retrieval_elapsed:.4f} seconds.")
    
    time_log = {
        "query": query,
        "keyword_time": keyword_elapsed,
        "retrieval_time": retrieval_elapsed
    }

    with open("/workspace/query_time_log_hybrid.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(time_log, ensure_ascii=False) + "\n")
    
    print(f"[INFO] context 길이 : {len(context)}.")

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )
        
    cls = os.path.basename(global_config["working_dir"])
    
    if cls == "novelqa":
        option = get_options_by_query("/workspace/datasets/novelqa.json", query)
        sys_prompt_temp = prompt if prompt else PROMPTS["rag_response_novelqa"]
        sys_prompt = sys_prompt_temp.format(
            context_data=context,
            response_type=query_param.response_type,
            history=history_context,
            options=option,
        )
    elif cls == "infiniteqa":
        sys_prompt_temp = prompt if prompt else PROMPTS["rag_response_infiniteqa"]
        sys_prompt = sys_prompt_temp.format(
            context_data=context,
            response_type=query_param.response_type,
            history=history_context,
        )
    elif cls == "infinitechoice":
        option = get_options_by_query_infinite("/workspace/datasets/infinitechoice.jsonl", query)
        sys_prompt_temp = prompt if prompt else PROMPTS["rag_response_infinitechoice"]
        sys_prompt = sys_prompt_temp.format(
            context_data=context,
            response_type=query_param.response_type,
            history=history_context,
            options=option,
        )
    else:
        sys_prompt_temp = prompt if prompt else PROMPTS["rag_response"]
        sys_prompt = sys_prompt_temp.format(
            context_data=context,
            response_type=query_param.response_type,
            history=history_context,
        )

    if query_param.only_need_prompt:
        return sys_prompt

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    token_usage = len(encoding.encode(sys_prompt))
    
    # 토큰 사용량 저장
    token_log = {
        "query": query,
        "token_usage": token_usage
    }

    with open("/workspace/incontext_token_log_hybrid.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(token_log, ensure_ascii=False) + "\n")
    
    print(response)
    
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )
    return response

async def extract_keywords_only(
    text: str,
    param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).
    """

    # 1. Handle cache if needed - add cache type for keywords
    args_hash = compute_args_hash(param.mode, text, cache_type="keywords")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )
    if cached_response is not None:
        try:
            keywords_data = json.loads(cached_response)
            return keywords_data["high_level_keywords"], keywords_data[
                "low_level_keywords"
            ]
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )

    # 2. Build the examples
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["keywords_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["keywords_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["keywords_extraction_examples"])
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    # 3. Process conversation history
    history_context = ""
    if param.conversation_history:
        history_context = get_conversation_turns(
            param.conversation_history, param.history_turns
        )

    # 4. Build the keyword-extraction prompt
    kw_prompt = PROMPTS["keywords_extraction"].format(
        query=text, examples=examples, language=language, history=history_context
    )

    # 5. Call the LLM for keyword extraction
    use_model_func = global_config["llm_model_func"]
    result = await use_model_func(kw_prompt, keyword_extraction=True)
    
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    token_usage = len(encoding.encode(kw_prompt))

    # 토큰 사용량 저장
    token_log = {
        "query": text,
        "token_usage": token_usage
    }

    with open("/workspace/keyword_token_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(token_log, ensure_ascii=False) + "\n")
    
    # 6. Parse out JSON from the LLM response
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("No JSON-like structure found in the LLM respond.")
        return [], []
    try:
        keywords_data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return [], []

    hl_keywords = keywords_data.get("high_level_keywords", [])
    ll_keywords = keywords_data.get("low_level_keywords", [])

    # 7. Cache only the processed keywords with cache type
    if hl_keywords or ll_keywords:
        cache_data = {
            "high_level_keywords": hl_keywords,
            "low_level_keywords": ll_keywords,
        }
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=json.dumps(cache_data),
                prompt=text,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=param.mode,
                cache_type="keywords",
            ),
        )
    return hl_keywords, ll_keywords

async def mix_kg_vector_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:
    """
    Hybrid retrieval implementation combining knowledge graph and vector search.

    This function performs a hybrid search by:
    1. Extracting semantic information from knowledge graph
    2. Retrieving relevant text chunks through vector similarity
    3. Combining both results for comprehensive answer generation
    """
    # 1. Cache handling
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash("mix", query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, "mix", cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # 2. Execute knowledge graph and vector searches in parallel
    async def get_kg_context():
        try:
            # Extract keywords using extract_keywords_only function which already supports conversation history
            hl_keywords, ll_keywords = await extract_keywords_only(
                query, query_param, global_config, hashing_kv
            )

            if not hl_keywords and not ll_keywords:
                logger.warning("Both high-level and low-level keywords are empty")
                return None

            # Convert keyword lists to strings
            ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
            hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

            # Set query mode based on available keywords
            if not ll_keywords_str and not hl_keywords_str:
                return None
            elif not ll_keywords_str:
                query_param.mode = "global"
            elif not hl_keywords_str:
                query_param.mode = "local"
            else:
                query_param.mode = "hybrid"

            # Build knowledge graph context
            context = await _build_query_context(
                [ll_keywords_str, hl_keywords_str],
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                text_chunks_db,
                query_param,
            )

            return context

        except Exception as e:
            logger.error(f"Error in get_kg_context: {str(e)}")
            return None

    async def get_vector_context():
        # Consider conversation history in vector search
        augmented_query = query
        if history_context:
            augmented_query = f"{history_context}\n{query}"

        try:
            # Reduce top_k for vector search in hybrid mode since we have structured information from KG
            mix_topk = min(10, query_param.top_k)
            results = await chunks_vdb.query(augmented_query, top_k=mix_topk)
            if not results:
                return None

            chunks_ids = [r["id"] for r in results]
            chunks = await text_chunks_db.get_by_ids(chunks_ids)

            valid_chunks = []
            for chunk, result in zip(chunks, results):
                if chunk is not None and "content" in chunk:
                    # Merge chunk content and time metadata
                    chunk_with_time = {
                        "content": chunk["content"],
                        "created_at": result.get("created_at", None),
                    }
                    valid_chunks.append(chunk_with_time)

            if not valid_chunks:
                return None

            maybe_trun_chunks = truncate_list_by_token_size(
                valid_chunks,
                key=lambda x: x["content"],
                max_token_size=query_param.max_token_for_text_unit,
            )

            if not maybe_trun_chunks:
                return None

            # Include time information in content
            formatted_chunks = []
            for c in maybe_trun_chunks:
                chunk_text = c["content"]
                if c["created_at"]:
                    chunk_text = f"[Created at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c['created_at']))}]\n{chunk_text}"
                formatted_chunks.append(chunk_text)

            logger.info(f"Truncate {len(chunks)} to {len(formatted_chunks)} chunks")
            return "\n--New Chunk--\n".join(formatted_chunks)
        except Exception as e:
            logger.error(f"Error in get_vector_context: {e}")
            return None

    # 3. Execute both retrievals in parallel
    kg_context, vector_context = await asyncio.gather(
        get_kg_context(), get_vector_context()
    )

    # 4. Merge contexts
    if kg_context is None and vector_context is None:
        return PROMPTS["fail_response"]

    if query_param.only_need_context:
        return {"kg_context": kg_context, "vector_context": vector_context}

    # 5. Construct hybrid prompt
    sys_prompt = PROMPTS["mix_rag_response"].format(
        kg_context=kg_context
        if kg_context
        else "No relevant knowledge graph information found",
        vector_context=vector_context
        if vector_context
        else "No relevant text information found",
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    # 6. Generate response
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )

    # 清理响应内容
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

        # 7. Save cache - 只有在收集完整响应后才缓存
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=query,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode="mix",
                cache_type="query",
            ),
        )

    return response

async def ours_kg_query(
    query,
    atomic_knowledge_graph_inst: BaseGraphStorage,
    triple_knowledge_graph_inst: BaseGraphStorage,
    atomic_entities_vdb: BaseVectorStorage,
    atomic_relationships_vdb: BaseVectorStorage,
    triple_entities_vdb: BaseVectorStorage,
    triple_relationships_vdb: BaseVectorStorage,
    text_atomics_db: BaseKVStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
    prompt: str = "",
) -> str:
    # check cache(query)
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response
    
    # query extension/decomposition
    keywords = await query_process(
        query, query_param, global_config, hashing_kv
    )
    
    logger.debug(f"keywords: {keywords}")

    # Handle empty extended_query
    if keywords == []:
        logger.warning("keywords is empty")
        return PROMPTS["fail_response"]

    logger.info("Using %s mode for query processing", query_param.mode)

    # Build context
    context = await ours_build_query_context(
        query,
        keywords,
        atomic_knowledge_graph_inst,
        triple_knowledge_graph_inst,
        atomic_entities_vdb,
        atomic_relationships_vdb,
        triple_entities_vdb,
        triple_relationships_vdb,
        text_atomics_db,
        text_chunks_db,
        query_param,
    )
    
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )
        
    cls = os.path.basename(global_config["working_dir"])
    
    if cls == "novelqa":
        option = get_options_by_query("/workspace/datasets/novelqa.json", query)
        sys_prompt_temp = prompt if prompt else PROMPTS["rag_response_novelqa"]
        sys_prompt = sys_prompt_temp.format(
            context_data=context,
            response_type=query_param.response_type,
            history=history_context,
            options=option,
        )
    elif cls == "infiniteqa":
        sys_prompt_temp = prompt if prompt else PROMPTS["rag_response_infiniteqa"]
        sys_prompt = sys_prompt_temp.format(
            context_data=context,
            response_type=query_param.response_type,
            history=history_context,
        )
    elif cls == "infinitechoice":
        option = get_options_by_query_infinite("/workspace/datasets/infinitechoice.jsonl", query)
        sys_prompt_temp = prompt if prompt else PROMPTS["rag_response_infinitechoice"]
        sys_prompt = sys_prompt_temp.format(
            context_data=context,
            response_type=query_param.response_type,
            history=history_context,
            options=option,
        )
    elif cls == "hotpot":
        sys_prompt_temp = prompt if prompt else PROMPTS["rag_response_hotpot"]
        sys_prompt = sys_prompt_temp.format(
            context_data=context,
            response_type=query_param.response_type,
            history=history_context,
        )
    elif cls == "multihoprag":
        sys_prompt_temp = prompt if prompt else PROMPTS["rag_response_multihoprag"]
        sys_prompt = sys_prompt_temp.format(
            context_data=context,
            response_type=query_param.response_type,
            history=history_context,
        )
    else:
        sys_prompt_temp = prompt if prompt else PROMPTS["rag_response"]
        
        sys_prompt = sys_prompt_temp.format(
            context_data=context,
            history=history_context,
        )

    if query_param.only_need_prompt:
        return sys_prompt

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    
    print(f"[INFO] sys_prompt 길이 : {len(sys_prompt)}.")
    
    print(response)
    
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )
    return response

import os
import json

async def ours_kg_query_experiment0(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:
    # Step 0: query decomposition
    if param.query_mode in ["experiment1"]:
        queries = await query_process_1(query, param, global_config, hashing_kv)
    elif param.query_mode in ["experiment2"]:
        queries = await query_process_2(query, param, global_config, hashing_kv)
    elif param.query_mode in ["experiment3"]:
        queries = await query_process_3(query, param, global_config, hashing_kv)
    elif param.query_mode in ["experiment4"]:
        queries = await query_process_4(query, param, global_config, hashing_kv)
    else:
        queries = [query]
    
    # Step 1: chunks_vdb에서 query 관련 chunk 검색
    try:
        subquery_results = {}
        if param.query_mode in ["experiment1", "experiment2", "experiment3", "experiment4"]:
            for q in queries:
                res = await chunks_vdb.query(q, top_k=10)
                for r in res:
                    r["sub_query"] = q
                subquery_results[q] = res
        else:
            res = await chunks_vdb.query(query, top_k=10)
            subquery_results[query] = res
    except Exception as e:
        print(f"[ERROR] chunks_vdb query failed: {e}")
        subquery_results = {}

    # Step 2: 라운드 로빈 + 중복 제거 방식으로 결과 합치기
    results = []
    seen = set()
    pointers = {sq: 0 for sq in subquery_results}  # 각 subquery별 index 포인터

    while True:
        progress = False
        for sq, res_list in subquery_results.items():
            idx = pointers[sq]
            # 중복 건너뛰기
            while idx < len(res_list) and res_list[idx].get("id") in seen:
                idx += 1
            if idx < len(res_list):
                r = res_list[idx]
                chunk_id = r.get("id")
                score = r.get("distance")
                try:
                    chunk_datas = await text_chunks_db.get_by_ids([chunk_id])
                    chunk_text = chunk_datas[0].get("content", "") if chunk_datas else ""
                except Exception:
                    chunk_text = ""

                results.append({
                    "query": query,
                    "sub_query": sq,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text,
                    "score": score
                })
                seen.add(chunk_id)
                pointers[sq] = idx + 1
                progress = True

        if not progress:  # 더 이상 뽑을 게 없으면 종료
            break
    
    results = results[:10]
    
    # Step 3: JSON 저장
    cls = os.path.basename(global_config["working_dir"])
    if param.query_mode in ["experiment1"]:
        output_path = f"/workspace/AtomRAG/{cls}/query_decom1_chunk_mapping.json"
    elif param.query_mode in ["experiment2"]:
        output_path = f"/workspace/AtomRAG/{cls}/query_decom2_chunk_mapping.json"
    elif param.query_mode in ["experiment3"]:
        output_path = f"/workspace/AtomRAG/{cls}/query_decom3_chunk_mapping.json"
    elif param.query_mode in ["experiment4"]:
        output_path = f"/workspace/AtomRAG/{cls}/query_decom4_chunk_mapping.json"
    else:
        output_path = f"/workspace/AtomRAG/{cls}/query_chunk_mapping.json"
    output_entry = {
        "query": query,
        "results": results   # query와 chunk top10 리스트
    }

    try:
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                if not isinstance(old_data, list):
                    old_data = [old_data]
        else:
            old_data = []

        old_data.append(output_entry)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(old_data, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Results for query='{query}' appended to {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")

    return output_entry

###### only summary ######
# async def ours_kg_query_experiment1(
#     query,
#     atomic_knowledge_graph_inst: BaseGraphStorage,
#     atomic_entities_vdb: BaseVectorStorage,
#     text_atomics_db: BaseKVStorage,
#     text_chunks_db: BaseKVStorage,
#     param: QueryParam,
#     global_config: dict,
#     hashing_kv: BaseKVStorage = None,
# ) -> str:
#     # Step 0. Query decomposition
#     if param.query_mode == "experiment1":
#         queries = await query_process_1(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment2":
#         queries = await query_process_2(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment3":
#         queries = await query_process_3(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment4":
#         queries = await query_process_4(query, param, global_config, hashing_kv)
#     else:
#         queries = [query]

#     # Step 1. subquery별 atomic fact 검색 (top-10)
#     subquery_results = {}
#     try:
#         for sq in queries:
#             res = await atomic_entities_vdb.query(sq, top_k=10)
#             subquery_results[sq] = res
#     except Exception as e:
#         print(f"[ERROR] atomic_entities_vdb query failed: {e}")
#         return []
    
#     if not subquery_results:
#         print("[INFO] No atomic facts retrieved.")
#         return []

#     # Step 2. chunk 단위로 묶기
#     G_atomic = getattr(atomic_knowledge_graph_inst, "_graph", None)
#     chunk_to_atomics = {}

#     if G_atomic:
#         for sq, res_list in subquery_results.items():
#             for r in res_list:
#                 atom_id = r.get("id")
#                 entity_name = r.get("entity_name")
#                 score = r.get("distance", 0.0)

#                 if not entity_name or entity_name not in G_atomic:
#                     continue

#                 try:
#                     node_data = await atomic_knowledge_graph_inst.get_node(entity_name)
#                 except Exception:
#                     node_data = None
#                 if not isinstance(node_data, dict):
#                     continue

#                 src_id = node_data.get("source_id")  # 연결된 chunk id들
#                 atomic_id = node_data.get("atomic_id") or atom_id
#                 if not src_id or not atomic_id:
#                     continue

#                 # atomic fact text 가져오기
#                 try:
#                     atomic_data = await text_atomics_db.get_by_ids([atomic_id])
#                     atomic_text = atomic_data[0].get("content", "") if atomic_data else ""
#                 except Exception:
#                     atomic_text = ""

#                 # 여러 chunk에 연결될 수 있음
#                 chunk_ids = [cid.strip() for cid in src_id.split("<SEP>") if cid.strip()]
#                 for cid in chunk_ids:
#                     try:
#                         chunk_data = await text_chunks_db.get_by_ids([cid])
#                         chunk_text = chunk_data[0].get("content", "") if chunk_data else ""
#                     except Exception:
#                         chunk_text = ""

#                     if cid not in chunk_to_atomics:
#                         chunk_to_atomics[cid] = {
#                             "chunk_id": cid,
#                             "chunk_text": chunk_text,
#                             "atomic_facts": []
#                         }

#                     chunk_to_atomics[cid]["atomic_facts"].append({
#                         "atomic_id": atomic_id,
#                         "atomic_text": atomic_text,
#                         "sub_query": sq,
#                         "score": score
#                     })

#     # Step 3. 결과 리스트 정리
#     results = list(chunk_to_atomics.values())

#     # Step 4. JSON 저장 (중간 결과)
#     cls = os.path.basename(global_config["working_dir"])
#     output_path = f"/workspace/AtomRAG/{cls}/query_atomic_chunk_mapping.json"

#     output_entry = {
#         "query": query,
#         "results": results
#     }

#     try:
#         if os.path.exists(output_path):
#             with open(output_path, "r", encoding="utf-8") as f:
#                 old_data = json.load(f)
#                 if not isinstance(old_data, list):
#                     old_data = [old_data]
#         else:
#             old_data = []
#         old_data.append(output_entry)

#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(old_data, f, ensure_ascii=False, indent=2)
#         print(f"[INFO] Results for query='{query}' saved to {output_path}")
#     except Exception as e:
#         print(f"[ERROR] Failed to save results: {e}")

#     # Step 5. Chunk-level Summarization (Parallel)
#     use_model_func = global_config["llm_model_func"]
#     results = output_entry["results"]

#     if not results:
#         print("[INFO] No chunk results found. Returning empty response.")
#         return PROMPTS.get("fail_response", "No results found.")

#     summary_prompt_template = (
#         "The following is a part of a document (chunk):\n\n"
#         "{chunk_text}\n\n"
#         "And here are the key factual statements (atomic facts) extracted from this chunk:\n{atomic_facts}\n\n"
#         "Please summarize the content in the chunk that **supports or explains these facts**.\n"
#         "Do not simply repeat the facts; instead, concisely describe the **evidence or reasoning** that backs them up."
#     )

#     print(f"[INFO] Generating {len(results)} chunk-level summaries in parallel...")

#     async def summarize_chunk(i, item):
#         chunk_text = item.get("chunk_text", "")
#         atomic_facts = item.get("atomic_facts", [])
#         atomic_list_str = "\n".join(
#             [f"- {af['atomic_text']}" for af in atomic_facts if af.get("atomic_text")]
#         )

#         prompt = summary_prompt_template.format(
#             chunk_text=chunk_text,
#             atomic_facts=atomic_list_str
#         )

#         try:
#             response = await use_model_func(
#                 query,
#                 system_prompt=prompt,
#                 stream=param.stream,
#             )
#             if isinstance(response, str):
#                 response = response.strip()
#             return {
#                 "chunk_id": item.get("chunk_id", f"chunk_{i}"),
#                 "summary": response
#             }
#         except Exception as e:
#             print(f"[ERROR] Summary generation failed for chunk {i}: {e}")
#             return None

#     # 병렬 실행
#     tasks = [summarize_chunk(i, item) for i, item in enumerate(results)]
#     summaries_raw = await asyncio.gather(*tasks)
#     summaries = [s for s in summaries_raw if s is not None]

#     if not summaries:
#         print("[INFO] No summaries generated. Returning fail response.")
#         return PROMPTS.get("fail_response", "Summary generation failed.")

#     # Step 6. Summaries concat → Final Answer Prompt
#     context = "\n\n".join(
#         [f"[Chunk {i+1} Summary]\n{summ['summary']}" for i, summ in enumerate(summaries)]
#     )

#     history_context = ""
#     if param.conversation_history:
#         history_context = get_conversation_turns(param.conversation_history, param.history_turns)

#     if cls == "hotpot":
#         sys_prompt_temp = PROMPTS["rag_response_hotpot"]
#     elif cls == "multihoprag":
#         sys_prompt_temp = PROMPTS["rag_response_multihoprag"]
#     else:
#         sys_prompt_temp = PROMPTS["ours_rag_response"]

#     sys_prompt = sys_prompt_temp.format(
#         context_data=context,
#         response_type=param.response_type,
#         history=history_context,
#     )

#     print(f"[INFO] Generating final answer using summarized context...")

#     try:
#         final_response = await use_model_func(
#             query,
#             system_prompt=sys_prompt,
#             stream=param.stream,
#         )
#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return PROMPTS.get("fail_response", "Final generation failed.")

#     print(final_response)
    
#     # Step 7. 결과 저장
#     final_output_path = f"/workspace/AtomRAG/{cls}/query_final_answer.json"
#     final_entry = {
#         "query": query,
#         "summaries": summaries,
#         "final_answer": final_response
#     }

#     try:
#         if os.path.exists(final_output_path):
#             with open(final_output_path, "r", encoding="utf-8") as f:
#                 old_data = json.load(f)
#                 if not isinstance(old_data, list):
#                     old_data = [old_data]
#         else:
#             old_data = []
#         old_data.append(final_entry)

#         with open(final_output_path, "w", encoding="utf-8") as f:
#             json.dump(old_data, f, ensure_ascii=False, indent=2)
#         print(f"[INFO] Final answer for query='{query}' saved to {final_output_path}")
#     except Exception as e:
#         print(f"[ERROR] Failed to save final answer: {e}")

#     return final_response

###### only atomic fact ######
# import os
# import json
# import asyncio
# from typing import Dict, Any

# async def ours_kg_query_experiment1(
#     query,
#     atomic_knowledge_graph_inst,
#     atomic_entities_vdb,
#     text_atomics_db,
#     text_chunks_db,
#     param,
#     global_config: Dict[str, Any],
#     hashing_kv=None,
# ) -> str:
#     # Step 0. Query decomposition
#     if param.query_mode == "experiment1":
#         queries = await query_process_1(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment2":
#         queries = await query_process_2(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment3":
#         queries = await query_process_3(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment4":
#         queries = await query_process_4(query, param, global_config, hashing_kv)
#     else:
#         queries = [query]

#     # Step 1. subquery별 atomic fact 검색 (top-10)
#     subquery_results = {}
#     try:
#         for sq in queries:
#             res = await atomic_entities_vdb.query(sq, top_k=10)
#             subquery_results[sq] = res
#     except Exception as e:
#         print(f"[ERROR] atomic_entities_vdb query failed: {e}")
#         return []
    
#     if not subquery_results:
#         print("[INFO] No atomic facts retrieved.")
#         return []

#     # Step 2. chunk 단위로 묶기
#     G_atomic = getattr(atomic_knowledge_graph_inst, "_graph", None)
#     chunk_to_atomics = {}

#     if G_atomic:
#         for sq, res_list in subquery_results.items():
#             for r in res_list:
#                 atom_id = r.get("id")
#                 entity_name = r.get("entity_name")
#                 score = r.get("distance", 0.0)

#                 if not entity_name or entity_name not in G_atomic:
#                     continue

#                 try:
#                     node_data = await atomic_knowledge_graph_inst.get_node(entity_name)
#                 except Exception:
#                     node_data = None
#                 if not isinstance(node_data, dict):
#                     continue

#                 src_id = node_data.get("source_id")  # 연결된 chunk id들
#                 atomic_id = node_data.get("atomic_id") or atom_id
#                 if not src_id or not atomic_id:
#                     continue

#                 # atomic fact text 가져오기
#                 try:
#                     atomic_data = await text_atomics_db.get_by_ids([atomic_id])
#                     atomic_text = atomic_data[0].get("content", "") if atomic_data else ""
#                 except Exception:
#                     atomic_text = ""

#                 # 여러 chunk에 연결될 수 있음
#                 chunk_ids = [cid.strip() for cid in src_id.split("<SEP>") if cid.strip()]
#                 for cid in chunk_ids:
#                     try:
#                         chunk_data = await text_chunks_db.get_by_ids([cid])
#                         chunk_text = chunk_data[0].get("content", "") if chunk_data else ""
#                     except Exception:
#                         chunk_text = ""

#                     if cid not in chunk_to_atomics:
#                         chunk_to_atomics[cid] = {
#                             "chunk_id": cid,
#                             "chunk_text": chunk_text,
#                             "atomic_facts": []
#                         }

#                     chunk_to_atomics[cid]["atomic_facts"].append({
#                         "atomic_id": atomic_id,
#                         "atomic_text": atomic_text,
#                         "sub_query": sq,
#                         "score": score
#                     })

#     # Step 3. 결과 리스트 정리
#     results = list(chunk_to_atomics.values())

#     # Step 4. JSON 저장 (중간 결과)
#     cls = os.path.basename(global_config["working_dir"])
#     output_path = f"/workspace/AtomRAG/{cls}/query_atomic_chunk_mapping.json"

#     output_entry = {"query": query, "results": results}
#     try:
#         if os.path.exists(output_path):
#             with open(output_path, "r", encoding="utf-8") as f:
#                 old_data = json.load(f)
#                 if not isinstance(old_data, list):
#                     old_data = [old_data]
#         else:
#             old_data = []
#         old_data.append(output_entry)

#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(old_data, f, ensure_ascii=False, indent=2)
#         print(f"[INFO] Results for query='{query}' saved to {output_path}")
#     except Exception as e:
#         print(f"[ERROR] Failed to save results: {e}")

#     # Step 5. Summarization 생략 → Atomic facts를 직접 context로 사용
#     print("[INFO] Skipping summarization. Using raw atomic facts as context...")

#     # 모든 chunk의 atomic fact들을 합쳐서 context 생성
#     atomic_contexts = []
#     for item in results:
#         facts_str = "\n".join(
#             [f"- {af['atomic_text']}" for af in item.get("atomic_facts", []) if af.get("atomic_text")]
#         )
#         if facts_str.strip():
#             atomic_contexts.append(f"[Chunk {item['chunk_id']} Atomic Facts]\n{facts_str}")

#     context = "\n\n".join(atomic_contexts)

#     if not context.strip():
#         print("[INFO] No atomic facts to use as context.")
#         return "No atomic facts found."

#     # Step 6. Final Answer 생성
#     history_context = ""
#     if param.conversation_history:
#         history_context = get_conversation_turns(param.conversation_history, param.history_turns)

#     if cls == "hotpot":
#         sys_prompt_temp = PROMPTS["rag_response_hotpot"]
#     elif cls == "multihoprag":
#         sys_prompt_temp = PROMPTS["rag_response_multihoprag"]
#     else:
#         sys_prompt_temp = PROMPTS["ours_rag_response"]

#     sys_prompt = sys_prompt_temp.format(
#         context_data=context,
#         response_type=param.response_type,
#         history=history_context,
#     )

#     use_model_func = global_config["llm_model_func"]

#     print(f"[INFO] Generating final answer directly from atomic facts...")

#     try:
#         final_response = await use_model_func(
#             query,
#             system_prompt=sys_prompt,
#             stream=param.stream,
#         )
#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return "Final generation failed."

#     print(final_response)

#     # Step 7. 결과 저장
#     final_output_path = f"/workspace/AtomRAG/{cls}/query_final_answer.json"
#     final_entry = {
#         "query": query,
#         "summaries": [],  # 이제 summary 없음
#         "atomic_context": context,
#         "final_answer": final_response
#     }

#     try:
#         if os.path.exists(final_output_path):
#             with open(final_output_path, "r", encoding="utf-8") as f:
#                 old_data = json.load(f)
#                 if not isinstance(old_data, list):
#                     old_data = [old_data]
#         else:
#             old_data = []
#         old_data.append(final_entry)

#         with open(final_output_path, "w", encoding="utf-8") as f:
#             json.dump(old_data, f, ensure_ascii=False, indent=2)
#         print(f"[INFO] Final answer for query='{query}' saved to {final_output_path}")
#     except Exception as e:
#         print(f"[ERROR] Failed to save final answer: {e}")

#     return final_response

###### atomic fact + summary ######
# async def ours_kg_query_experiment1(
#     query,
#     atomic_knowledge_graph_inst: BaseGraphStorage,
#     atomic_entities_vdb: BaseVectorStorage,
#     text_atomics_db: BaseKVStorage,
#     text_chunks_db: BaseKVStorage,
#     param: QueryParam,
#     global_config: dict,
#     hashing_kv: BaseKVStorage = None,
# ) -> str:
#     # Step 0. Query decomposition
#     if param.query_mode == "experiment1":
#         queries = await query_process_1(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment2":
#         queries = await query_process_2(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment3":
#         queries = await query_process_3(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment4":
#         queries = await query_process_4(query, param, global_config, hashing_kv)
#     else:
#         queries = [query]

#     # Step 1. subquery별 atomic fact 검색 (top-10)
#     subquery_results = {}
#     try:
#         for sq in queries:
#             res = await atomic_entities_vdb.query(sq, top_k=10)
#             subquery_results[sq] = res
#     except Exception as e:
#         print(f"[ERROR] atomic_entities_vdb query failed: {e}")
#         return []
    
#     if not subquery_results:
#         print("[INFO] No atomic facts retrieved.")
#         return []

#     # Step 2. chunk 단위로 묶기
#     G_atomic = getattr(atomic_knowledge_graph_inst, "_graph", None)
#     chunk_to_atomics = {}

#     if G_atomic:
#         for sq, res_list in subquery_results.items():
#             for r in res_list:
#                 atom_id = r.get("id")
#                 entity_name = r.get("entity_name")
#                 score = r.get("distance", 0.0)

#                 if not entity_name or entity_name not in G_atomic:
#                     continue

#                 try:
#                     node_data = await atomic_knowledge_graph_inst.get_node(entity_name)
#                 except Exception:
#                     node_data = None
#                 if not isinstance(node_data, dict):
#                     continue

#                 src_id = node_data.get("source_id")  # 연결된 chunk id들
#                 atomic_id = node_data.get("atomic_id") or atom_id
#                 if not src_id or not atomic_id:
#                     continue

#                 # atomic fact text 가져오기
#                 try:
#                     atomic_data = await text_atomics_db.get_by_ids([atomic_id])
#                     atomic_text = atomic_data[0].get("content", "") if atomic_data else ""
#                 except Exception:
#                     atomic_text = ""

#                 # 여러 chunk에 연결될 수 있음
#                 chunk_ids = [cid.strip() for cid in src_id.split("<SEP>") if cid.strip()]
#                 for cid in chunk_ids:
#                     try:
#                         chunk_data = await text_chunks_db.get_by_ids([cid])
#                         chunk_text = chunk_data[0].get("content", "") if chunk_data else ""
#                     except Exception:
#                         chunk_text = ""

#                     if cid not in chunk_to_atomics:
#                         chunk_to_atomics[cid] = {
#                             "chunk_id": cid,
#                             "chunk_text": chunk_text,
#                             "atomic_facts": []
#                         }

#                     chunk_to_atomics[cid]["atomic_facts"].append({
#                         "atomic_id": atomic_id,
#                         "atomic_text": atomic_text,
#                         "sub_query": sq,
#                         "score": score
#                     })

#     # Step 3. 결과 리스트 정리
#     results = list(chunk_to_atomics.values())

#     # Step 4. JSON 저장 (중간 결과)
#     cls = os.path.basename(global_config["working_dir"])
#     output_path = f"/workspace/AtomRAG/{cls}/query_atomic_chunk_mapping.json"

#     output_entry = {
#         "query": query,
#         "results": results
#     }

#     try:
#         if os.path.exists(output_path):
#             with open(output_path, "r", encoding="utf-8") as f:
#                 old_data = json.load(f)
#                 if not isinstance(old_data, list):
#                     old_data = [old_data]
#         else:
#             old_data = []
#         old_data.append(output_entry)

#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(old_data, f, ensure_ascii=False, indent=2)
#         print(f"[INFO] Results for query='{query}' saved to {output_path}")
#     except Exception as e:
#         print(f"[ERROR] Failed to save results: {e}")

#     # Step 5. Chunk-level Summarization (Parallel)
#     use_model_func = global_config["llm_model_func"]
#     results = output_entry["results"]

#     if not results:
#         print("[INFO] No chunk results found. Returning empty response.")
#         return PROMPTS.get("fail_response", "No results found.")

#     summary_prompt_template = (
#         "The following is a part of a document (chunk):\n\n"
#         "{chunk_text}\n\n"
#         "And here are the key factual statements (atomic facts) extracted from this chunk:\n{atomic_facts}\n\n"
#         "Please summarize the content in the chunk that **supports or explains these facts**.\n"
#         "Do not simply repeat the facts; instead, describe the **evidence or reasoning** that backs them up."
#     )

#     print(f"[INFO] Generating {len(results)} chunk-level summaries in parallel...")

#     async def summarize_chunk(i, item):
#         chunk_text = item.get("chunk_text", "")
#         atomic_facts = item.get("atomic_facts", [])
#         atomic_list_str = "\n".join(
#             [f"- {af['atomic_text']}" for af in atomic_facts if af.get("atomic_text")]
#         )

#         prompt = summary_prompt_template.format(
#             chunk_text=chunk_text,
#             atomic_facts=atomic_list_str
#         )

#         try:
#             response = await use_model_func(
#                 query,
#                 system_prompt=prompt,
#                 stream=param.stream,
#             )
#             if isinstance(response, str):
#                 response = response.strip()
#             return {
#                 "chunk_id": item.get("chunk_id", f"chunk_{i}"),
#                 "summary": response
#             }
#         except Exception as e:
#             print(f"[ERROR] Summary generation failed for chunk {i}: {e}")
#             return None

#     # 병렬 실행
#     tasks = [summarize_chunk(i, item) for i, item in enumerate(results)]
#     summaries_raw = await asyncio.gather(*tasks)
#     summaries = [s for s in summaries_raw if s is not None]

#     if not summaries:
#         print("[INFO] No summaries generated. Returning fail response.")
#         return PROMPTS.get("fail_response", "Summary generation failed.")

#     # Step 6. Summaries concat → Final Answer Prompt
#     context_blocks = []
#     for i, summ in enumerate(summaries):
#         chunk_id = summ.get("chunk_id", f"chunk_{i}")
#         summary_text = summ.get("summary", "")

#         # 대응되는 chunk의 atomic facts 가져오기
#         related_chunk = next((r for r in results if r.get("chunk_id") == chunk_id), None)
#         if related_chunk:
#             atomic_facts = related_chunk.get("atomic_facts", [])
#             atomic_list_str = "\n".join(
#                 [f"- {af.get('atomic_text', '')}" for af in atomic_facts if af.get("atomic_text")]
#             )
#         else:
#             atomic_list_str = ""

#         # summary + atomic fact를 함께 넣기
#         block_text = (
#             f"[Chunk {i+1}]\n"
#             f"**Atomic Facts:**\n{atomic_list_str}\n\n"
#             f"**Summary:**\n{summary_text}\n"
#         )
#         context_blocks.append(block_text)

#     context = "\n\n".join(context_blocks)

#     history_context = ""
#     if param.conversation_history:
#         history_context = get_conversation_turns(param.conversation_history, param.history_turns)

#     if cls == "hotpot":
#         sys_prompt_temp = PROMPTS["rag_response_hotpot"]
#     elif cls == "multihoprag":
#         sys_prompt_temp = PROMPTS["rag_response_multihoprag"]
#     else:
#         sys_prompt_temp = PROMPTS["ours_rag_response"]

#     sys_prompt = sys_prompt_temp.format(
#         context_data=context,
#         response_type=param.response_type,
#         history=history_context,
#     )

#     print(f"[INFO] Generating final answer using summarized context...")

#     try:
#         final_response = await use_model_func(
#             query,
#             system_prompt=sys_prompt,
#             stream=param.stream,
#         )
#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return PROMPTS.get("fail_response", "Final generation failed.")

#     print(final_response)
    
#     # Step 7. 결과 저장
#     final_output_path = f"/workspace/AtomRAG/{cls}/query_final_answer.json"
#     final_entry = {
#         "query": query,
#         "summaries": summaries,
#         "final_answer": final_response
#     }

#     try:
#         if os.path.exists(final_output_path):
#             with open(final_output_path, "r", encoding="utf-8") as f:
#                 old_data = json.load(f)
#                 if not isinstance(old_data, list):
#                     old_data = [old_data]
#         else:
#             old_data = []
#         old_data.append(final_entry)

#         with open(final_output_path, "w", encoding="utf-8") as f:
#             json.dump(old_data, f, ensure_ascii=False, indent=2)
#         print(f"[INFO] Final answer for query='{query}' saved to {final_output_path}")
#     except Exception as e:
#         print(f"[ERROR] Failed to save final answer: {e}")

#     return final_response

##### chunk filtering + atomic fact + summary ######
# async def ours_kg_query_experiment1(
#     query,
#     atomic_knowledge_graph_inst: BaseGraphStorage,
#     atomic_entities_vdb: BaseVectorStorage,
#     text_atomics_db: BaseKVStorage,
#     text_chunks_db: BaseKVStorage,
#     param: QueryParam,
#     global_config: dict,
#     hashing_kv: BaseKVStorage = None,
# ) -> str:
#     # Step 0. Query decomposition
#     if param.query_mode == "experiment1":
#         queries = await query_process_1(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment2":
#         queries = await query_process_2(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment3":
#         queries = await query_process_3(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment4":
#         queries = await query_process_4(query, param, global_config, hashing_kv)
#     else:
#         queries = [query]

#     # Step 1. subquery별 atomic fact 검색 (top-10)
#     subquery_results = {}
#     try:
#         for sq in queries:
#             res = await atomic_entities_vdb.query(sq, top_k=10)
#             subquery_results[sq] = res
#     except Exception as e:
#         print(f"[ERROR] atomic_entities_vdb query failed: {e}")
#         return []
    
#     if not subquery_results:
#         print("[INFO] No atomic facts retrieved.")
#         return []

#     # Step 2. chunk 단위로 묶기
#     G_atomic = getattr(atomic_knowledge_graph_inst, "_graph", None)
#     chunk_to_atomics = {}

#     if G_atomic:
#         for sq, res_list in subquery_results.items():
#             for r in res_list:
#                 atom_id = r.get("id")
#                 entity_name = r.get("entity_name")
#                 score = r.get("distance", 0.0)

#                 if not entity_name or entity_name not in G_atomic:
#                     continue

#                 try:
#                     node_data = await atomic_knowledge_graph_inst.get_node(entity_name)
#                 except Exception:
#                     node_data = None
#                 if not isinstance(node_data, dict):
#                     continue

#                 src_id = node_data.get("source_id")  # 연결된 chunk id들
#                 atomic_id = node_data.get("atomic_id") or atom_id
#                 if not src_id or not atomic_id:
#                     continue

#                 # atomic fact text 가져오기
#                 try:
#                     atomic_data = await text_atomics_db.get_by_ids([atomic_id])
#                     atomic_text = atomic_data[0].get("content", "") if atomic_data else ""
#                 except Exception:
#                     atomic_text = ""

#                 # 여러 chunk에 연결될 수 있음
#                 chunk_ids = [cid.strip() for cid in src_id.split("<SEP>") if cid.strip()]
#                 for cid in chunk_ids:
#                     try:
#                         chunk_data = await text_chunks_db.get_by_ids([cid])
#                         chunk_text = chunk_data[0].get("content", "") if chunk_data else ""
#                     except Exception:
#                         chunk_text = ""

#                     if cid not in chunk_to_atomics:
#                         chunk_to_atomics[cid] = {
#                             "chunk_id": cid,
#                             "chunk_text": chunk_text,
#                             "atomic_facts": []
#                         }

#                     chunk_to_atomics[cid]["atomic_facts"].append({
#                         "atomic_id": atomic_id,
#                         "atomic_text": atomic_text,
#                         "sub_query": sq,
#                         "score": score
#                     })

#     # Step 3. 결과 리스트 정리
#     results = list(chunk_to_atomics.values())

#     # Step 3-1. Chunk 대표 score 기반 Top-K Filtering
#     top_k_chunks = 10  # 상위 K개 chunk만 선택

#     scored_results = []
#     for item in results:
#         atomic_facts = item.get("atomic_facts", [])
#         if not atomic_facts:
#             continue

#         # atomic fact들의 cosine similarity를 그대로 사용
#         similarities = []
#         for af in atomic_facts:
#             d = af.get("score", None)
#             if isinstance(d, (int, float)):
#                 sim = d  # 이미 cosine similarity 값임 → 변환 불필요
#                 similarities.append(sim)

#         # chunk의 대표 score = 해당 chunk 내 가장 높은 similarity
#         chunk_score = max(similarities) if similarities else 0.0

#         scored_results.append({
#             **item,
#             "chunk_score": chunk_score
#         })

#     # 점수 기준으로 정렬 후 상위 K개만 선택
#     scored_results.sort(key=lambda x: x["chunk_score"], reverse=True)
#     filtered_results = scored_results[:top_k_chunks]

#     print(f"[INFO] Filtered top-{top_k_chunks} chunks out of {len(results)} total "
#         f"based on max cosine similarity per chunk.")
#     results = filtered_results

#     # Step 4. JSON 저장 (중간 결과)
#     cls = os.path.basename(global_config["working_dir"])
#     output_path = f"/workspace/AtomRAG/{cls}/query_atomic_chunk_mapping.json"

#     output_entry = {
#         "query": query,
#         "results": results
#     }

#     try:
#         if os.path.exists(output_path):
#             with open(output_path, "r", encoding="utf-8") as f:
#                 old_data = json.load(f)
#                 if not isinstance(old_data, list):
#                     old_data = [old_data]
#         else:
#             old_data = []
#         old_data.append(output_entry)

#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(old_data, f, ensure_ascii=False, indent=2)
#         print(f"[INFO] Results for query='{query}' saved to {output_path}")
#     except Exception as e:
#         print(f"[ERROR] Failed to save results: {e}")

#     # Step 5. Chunk-level Summarization (Parallel)
#     use_model_func = global_config["llm_model_func"]
#     results = output_entry["results"]

#     if not results:
#         print("[INFO] No chunk results found. Returning empty response.")
#         return PROMPTS.get("fail_response", "No results found.")

#     print(f"[INFO] Generating {len(results)} chunk-level summaries in parallel...")

#     async def summarize_chunk(i, item):
#         chunk_text = item.get("chunk_text", "")
#         atomic_facts = item.get("atomic_facts", [])
#         atomic_list_str = "\n".join(
#             [f"- {af['atomic_text']}" for af in atomic_facts if af.get("atomic_text")]
#         )
        
#         summary_prompt_template = PROMPTS["chunk_summary"]
        
#         prompt = summary_prompt_template.format(
#             chunk_text=chunk_text,
#             atomic_facts=atomic_list_str
#         )

#         try:
#             response = await use_model_func(
#                 system_prompt=prompt,
#                 stream=param.stream,
#             )
#             if isinstance(response, str):
#                 response = response.strip()
#             return {
#                 "chunk_id": item.get("chunk_id", f"chunk_{i}"),
#                 "summary": response
#             }
#         except Exception as e:
#             print(f"[ERROR] Summary generation failed for chunk {i}: {e}")
#             return None

#     # 병렬 실행
#     tasks = [summarize_chunk(i, item) for i, item in enumerate(results)]
#     summaries_raw = await asyncio.gather(*tasks)
#     summaries = [s for s in summaries_raw if s is not None]

#     if not summaries:
#         print("[INFO] No summaries generated. Returning fail response.")
#         return PROMPTS.get("fail_response", "Summary generation failed.")

#     # Step 6. Summaries concat → Final Answer Prompt
#     context_blocks = []
#     for i, summ in enumerate(summaries):
#         chunk_id = summ.get("chunk_id", f"chunk_{i}")
#         summary_text = summ.get("summary", "")

#         # 대응되는 chunk의 atomic facts 가져오기
#         related_chunk = next((r for r in results if r.get("chunk_id") == chunk_id), None)
#         if related_chunk:
#             atomic_facts = related_chunk.get("atomic_facts", [])
#             atomic_list_str = "\n".join(
#                 [f"- {af.get('atomic_text', '')}" for af in atomic_facts if af.get("atomic_text")]
#             )
#         else:
#             atomic_list_str = ""

#         # summary + atomic fact를 함께 넣기
#         block_text = (
#             f"[Chunk {i+1}]\n"
#             f"**Atomic Facts:**\n{atomic_list_str}\n\n"
#             f"**Summary:**\n{summary_text}\n"
#         )
#         context_blocks.append(block_text)

#     context = "\n\n".join(context_blocks)

#     if cls == "hotpot":
#         sys_prompt_temp = PROMPTS["rag_response_hotpot"]
#     elif cls == "multihoprag":
#         sys_prompt_temp = PROMPTS["rag_response_multihoprag"]
#     else:
#         sys_prompt_temp = PROMPTS["ours_rag_response"]

#     sys_prompt = sys_prompt_temp.format(
#         context_data=context,
#     )

#     print(f"[INFO] Generating final answer using summarized context...")

#     try:
#         final_response = await use_model_func(
#             query,
#             system_prompt=sys_prompt,
#             stream=param.stream,
#         )
#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return PROMPTS.get("fail_response", "Final generation failed.")

#     print("fianl prompt length:", len(sys_prompt))
    
#     print(final_response)
    
#     # Step 7. 결과 저장
#     final_output_path = f"/workspace/AtomRAG/{cls}/query_final_answer.json"
#     final_entry = {
#         "query": query,
#         "summaries": summaries,
#         "final_prompt": sys_prompt,
#         "final_answer": final_response,
#     }

#     try:
#         if os.path.exists(final_output_path):
#             with open(final_output_path, "r", encoding="utf-8") as f:
#                 old_data = json.load(f)
#                 if not isinstance(old_data, list):
#                     old_data = [old_data]
#         else:
#             old_data = []
#         old_data.append(final_entry)

#         with open(final_output_path, "w", encoding="utf-8") as f:
#             json.dump(old_data, f, ensure_ascii=False, indent=2)
#         print(f"[INFO] Final answer for query='{query}' saved to {final_output_path}")
#     except Exception as e:
#         print(f"[ERROR] Failed to save final answer: {e}")

#     return final_response



##### chunk filtering + atomic fact + rewriting (atomic fact 연결하기 전 최종) ######
# async def ours_kg_query_experiment1(
#     query: str,
#     atomic_knowledge_graph_inst,
#     atomic_entities_vdb,
#     text_atomics_db,
#     text_chunks_db,
#     param,
#     global_config: dict,
#     hashing_kv=None,
# ) -> str:
#     """
#     Optimized version of the atomic fact-based retrieval and rewriting pipeline.
#     """

#     # -------------------------
#     # Step 0. Query decomposition
#     # -------------------------
#     query_modes = {
#         "experiment1": query_process_1,
#         "experiment2": query_process_2,
#         "experiment3": query_process_3,
#         "experiment4": query_process_4,
#     }
#     queries = await query_modes.get(param.query_mode, lambda *a: [query])(query, param, global_config, hashing_kv)
#     if not queries:
#         queries = [query]

#     # -------------------------
#     # Step 1. Retrieve atomic facts (async parallel)
#     # -------------------------
#     print(f"[INFO] Retrieving atomic facts for {len(queries)} subqueries...")

#     async def fetch_atomic_facts(sq):
#         try:
#             return sq, await atomic_entities_vdb.query(sq, top_k=10)
#         except Exception as e:
#             print(f"[ERROR] Retrieval failed for '{sq}': {e}")
#             return sq, []

#     subquery_results = dict(await asyncio.gather(*(fetch_atomic_facts(sq) for sq in queries)))

#     # -------------------------
#     # Step 2. Build chunk ↔ atomic mapping with caching
#     # -------------------------
#     G_atomic = getattr(atomic_knowledge_graph_inst, "_graph", None)
#     if not G_atomic:
#         print("[ERROR] No atomic knowledge graph found.")
#         return "No atomic graph."

#     chunk_to_atomics = defaultdict(lambda: {"atomic_facts": []})
#     atomic_cache, chunk_cache = {}, {}

#     async def fetch_atomic_and_chunk_data(atom_id, entity_name):
#         if entity_name not in G_atomic:
#             return None

#         # Cache atomic node
#         if entity_name not in atomic_cache:
#             try:
#                 atomic_cache[entity_name] = await atomic_knowledge_graph_inst.get_node(entity_name)
#             except Exception:
#                 atomic_cache[entity_name] = None

#         node_data = atomic_cache.get(entity_name)
#         if not isinstance(node_data, dict):
#             return None

#         src_ids = [cid.strip() for cid in node_data.get("source_id", "").split("<SEP>") if cid.strip()]
#         atomic_id = node_data.get("atomic_id") or atom_id

#         # Cache atomic text
#         if atomic_id not in atomic_cache:
#             try:
#                 data = await text_atomics_db.get_by_ids([atomic_id])
#                 atomic_cache[atomic_id] = data[0].get("content", "") if data else ""
#             except Exception:
#                 atomic_cache[atomic_id] = ""

#         atomic_text = atomic_cache[atomic_id]
#         return atomic_id, atomic_text, src_ids

#     tasks = []
#     for sq, results in subquery_results.items():
#         for r in results:
#             tasks.append(fetch_atomic_and_chunk_data(r.get("id"), r.get("entity_name")))

#     atomic_data_list = [res for res in await asyncio.gather(*tasks) if res]

#     # Build chunk mapping
#     for (sq, results) in subquery_results.items():
#         for r in results:
#             entity_name, score = r.get("entity_name"), r.get("distance", 0.0)
#             if not entity_name or entity_name not in G_atomic:
#                 continue

#             atomic_info = next((a for a in atomic_data_list if a and a[0] == r["id"]), None)
#             if not atomic_info:
#                 continue

#             atomic_id, atomic_text, src_ids = atomic_info
#             for cid in src_ids:
#                 # Cache chunk text
#                 if cid not in chunk_cache:
#                     try:
#                         data = await text_chunks_db.get_by_ids([cid])
#                         chunk_cache[cid] = data[0].get("content", "") if data else ""
#                     except Exception:
#                         chunk_cache[cid] = ""

#                 chunk_to_atomics[cid]["chunk_id"] = cid
#                 chunk_to_atomics[cid]["chunk_text"] = chunk_cache[cid]
#                 chunk_to_atomics[cid]["atomic_facts"].append({
#                     "atomic_id": atomic_id,
#                     "atomic_text": atomic_text,
#                     "sub_query": sq,
#                     "score": score
#                 })

#     # Deduplicate atomics per chunk
#     for item in chunk_to_atomics.values():
#         seen, unique = set(), []
#         for af in item["atomic_facts"]:
#             aid = af["atomic_id"]
#             if aid not in seen:
#                 seen.add(aid)
#                 unique.append(af)
#         item["atomic_facts"] = unique

#     results = list(chunk_to_atomics.values())
#     if not results:
#         print("[INFO] No atomic-chunk mappings found.")
#         return "No results."

#     # -------------------------
#     # Step 3. Rank chunks by max similarity
#     # -------------------------
#     for item in results:
#         sims = [af["score"] for af in item["atomic_facts"] if isinstance(af.get("score"), (int, float))]
#         item["chunk_score"] = max(sims, default=0.0)

#     top_k = min(10, len(results))
#     results = sorted(results, key=lambda x: x["chunk_score"], reverse=False)[:top_k]
#     print(f"[INFO] Selected top-{top_k} chunks.")

#     # -------------------------
#     # Step 4. Parallel rewriting
#     # -------------------------
#     print(f"[INFO] Running rewriting for {len(results)} chunks...")

#     async def rewrite_chunk(item, idx):
#         chunk_text = item["chunk_text"]
#         facts = item["atomic_facts"]
#         atomic_list = "\n".join(f"- {af['atomic_text']}" for af in facts if af.get("atomic_text"))

#         prompt = PROMPTS["chunk_rewriting"].format(
#             main_query=query,
#             chunk_text=chunk_text,
#             atomic_facts=atomic_list
#         )
#         try:
#             resp = await global_config["llm_model_func"](prompt, stream=param.stream)
#             return {"chunk_id": item["chunk_id"], "rewriting": resp.strip() if isinstance(resp, str) else resp}
#         except Exception as e:
#             print(f"[ERROR] Rewriting failed (chunk {idx}): {e}")
#             return None

#     rewriting = [r for r in await asyncio.gather(*(rewrite_chunk(item, i) for i, item in enumerate(results))) if r]

#     if not rewriting:
#         return PROMPTS.get("fail_response", "Rewriting failed.")

#     # -------------------------
#     # Step 5. Build final context
#     # -------------------------
#     def extract_field(text, key):
#         for line in text.splitlines():
#             if line.strip().startswith(f"[{key}]"):
#                 return line.split(":", 1)[-1].strip()
#         return None

#     context_blocks = []
#     for i, r in enumerate(rewriting):
#         rewriting_text = r.get("rewriting", "").strip()
#         evidence = extract_field(rewriting_text, "Rewritten Evidence")

#         evidence = evidence or rewriting_text

#         context_blocks.append(
#             f"### [Chunk {i+1}]\n"
#             f"- Rewritten Evidence: {evidence.strip()}\n"
#         )

#     context = "\n\n".join(context_blocks)


#     # -------------------------
#     # Step 6. Final answer generation
#     # -------------------------
#     if not context_blocks:
#         print("[INFO] No valid context (all unrelated). Returning fail response.")
#         return PROMPTS.get("fail_response", "No relevant chunks found.")

#     # Select system prompt by dataset type
#     cls = os.path.basename(global_config.get("working_dir", ""))
#     sys_prompt_temp = PROMPTS.get(f"rag_response_{cls}", PROMPTS["ours_rag_response"])

#     # Build system prompt
#     sys_prompt = sys_prompt_temp.format(context_data=context)
#     print(f"[INFO] Generating final answer from {len(context_blocks)} chunks (prompt length: {len(sys_prompt)})")

#     # Generate final response
#     try:
#         final_response = await global_config["llm_model_func"](
#             query,
#             system_prompt=sys_prompt,
#             stream=param.stream,
#         )
#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return PROMPTS.get("fail_response", "Final generation failed.")

#     # Log concise summary
#     print("[INFO] Final answer generation successful.")
#     if isinstance(final_response, str):
#         print(final_response[:300] + ("..." if len(final_response) > 300 else ""))
    
#     # -------------------------
#     # Step 7. Save full pipeline results as JSON (append mode)
#     # -------------------------
#     try:
#         cls = os.path.basename(global_config.get("working_dir", ""))
#         save_dir = f"/workspace/AtomRAG/{cls}"
#         os.makedirs(save_dir, exist_ok=True)
#         full_save_path = os.path.join(save_dir, f"{cls}_full_results.json")

#         # rewriting prompt + result 매핑 수집
#         full_chunks = []
#         for item, rewrite_data in zip(results, rewriting):
#             chunk_id = item.get("chunk_id", "")
#             chunk_text = item.get("chunk_text", "")
#             atomic_facts = item.get("atomic_facts", [])
#             rewriting_text = rewrite_data.get("rewriting", "")

#             # prompt 재생성 (기록용)
#             rewriting_prompt = PROMPTS["chunk_rewriting"].format(
#                 main_query=query,
#                 chunk_text=chunk_text,
#                 atomic_facts="\n".join(f"- {af['atomic_text']}" for af in atomic_facts if af.get("atomic_text"))
#             )

#             full_chunks.append({
#                 "chunk_id": chunk_id,
#                 "chunk_text": chunk_text,
#                 "atomic_facts": atomic_facts,
#                 "rewriting_prompt": rewriting_prompt,
#                 "rewriting_result": rewriting_text
#             })

#         # 전체 결과 구조
#         full_output = {
#             "query": query,
#             "chunks": full_chunks,
#             "final_answer": final_response if isinstance(final_response, str) else str(final_response),
#             "system_prompt": sys_prompt,   # 최종 시스템 프롬프트 저장 추가
#         }

#         # 파일이 없으면 리스트 초기화, 있으면 기존 리스트 불러오기
#         if os.path.exists(full_save_path):
#             with open(full_save_path, "r", encoding="utf-8") as f:
#                 try:
#                     existing_data = json.load(f)
#                     if not isinstance(existing_data, list):
#                         existing_data = [existing_data]
#                 except json.JSONDecodeError:
#                     existing_data = []
#         else:
#             existing_data = []

#         # 새 결과 추가 및 다시 저장
#         existing_data.append(full_output)
#         with open(full_save_path, "w", encoding="utf-8") as f:
#             json.dump(existing_data, f, ensure_ascii=False, indent=4)

#         print(f"[INFO] Appended full pipeline results to {full_save_path}")

#     except Exception as e:
#         print(f"[ERROR] Failed to save full results: {e}")


#     return final_response

# ##### chunk filtering + atomic fact + rewriting (atomic fact 연결) ######
# async def ours_kg_query_experiment1(
#     query: str,
#     atomic_knowledge_graph_inst,
#     atomic_entities_vdb,
#     text_triple_db,
#     text_atomics_db,
#     text_chunks_db,
#     param,
#     global_config: dict,
#     hashing_kv=None,
# ) -> str:
#     """
#     Optimized version of the atomic fact-based retrieval and rewriting pipeline.
#     Includes triple-based intra-chunk expansion and detailed step-by-step logging.
#     """

#     # -------------------------
#     # Step 0. Query decomposition
#     # -------------------------
#     query_modes = {
#         "experiment1": query_process_1,
#         "experiment2": query_process_2,
#         "experiment3": query_process_3,
#         "experiment4": query_process_4,
#     }
#     queries = await query_modes.get(param.query_mode, lambda *a: [query])(query, param, global_config, hashing_kv)
#     if not queries:
#         queries = [query]
#     print(f"[INFO] Step 0 complete — {len(queries)} subqueries generated.")

#     # -------------------------
#     # Step 1. Retrieve atomic facts (async parallel)
#     # -------------------------
#     print(f"[INFO] Retrieving atomic facts for {len(queries)} subqueries...")

#     async def fetch_atomic_facts(sq):
#         try:
#             return sq, await atomic_entities_vdb.query(sq, top_k=10)
#         except Exception as e:
#             print(f"[ERROR] Retrieval failed for '{sq}': {e}")
#             return sq, []

#     subquery_results = dict(await asyncio.gather(*(fetch_atomic_facts(sq) for sq in queries)))

#     # -------------------------
#     # Step 2. Build chunk ↔ atomic mapping (using text_atomics_db)
#     # -------------------------
#     chunk_to_atomics = defaultdict(lambda: {"atomic_facts": []})
#     chunk_cache: dict[str, str] = {}

#     retrieved_items = []
#     for sq, results in subquery_results.items():
#         for r in results:
#             rid = r.get("id")
#             if rid:
#                 retrieved_items.append((sq, rid, r.get("distance", 0.0)))

#     if not retrieved_items:
#         print("[INFO] No retrieved atomic facts from Step 1.")
#         return "No results."

#     unique_atomic_ids = list({rid for _, rid, _ in retrieved_items})
#     atomic_records = await text_atomics_db.get_by_ids(unique_atomic_ids)

#     # atomic_id → atomic info mapping
#     atomic_info_map = {}
#     for aid, rec in zip(unique_atomic_ids, (atomic_records or [])):
#         if not rec or not isinstance(rec, dict):
#             continue
#         atomic_info_map[aid] = {
#             "content": rec.get("content", ""),
#             "source_ids": rec.get("source_ids", []) or [],
#             "triple_ids": rec.get("triple_ids", []) or [],
#         }

#     if not atomic_info_map:
#         print("[INFO] text_atomics_db returned no atomic records.")
#         return "No results."

#     # chunk ↔ atomic 매핑 생성
#     for sq, aid, score in retrieved_items:
#         ainfo = atomic_info_map.get(aid)
#         if not ainfo:
#             continue
#         atomic_text = ainfo["content"]
#         src_chunks = ainfo["source_ids"]
#         if not src_chunks:
#             continue

#         for cid in src_chunks:
#             if cid not in chunk_cache:
#                 try:
#                     cdata = await text_chunks_db.get_by_ids([cid])
#                     chunk_cache[cid] = (cdata[0].get("content", "") if cdata else "")
#                 except Exception:
#                     chunk_cache[cid] = ""

#             chunk_to_atomics[cid]["chunk_id"] = cid
#             chunk_to_atomics[cid]["chunk_text"] = chunk_cache[cid]
#             chunk_to_atomics[cid]["atomic_facts"].append({
#                 "atomic_id": aid,
#                 "atomic_text": atomic_text,
#                 "sub_query": sq,
#                 "score": score,
#             })

#     # 중복 제거
#     for item in chunk_to_atomics.values():
#         seen, unique = set(), []
#         for af in item["atomic_facts"]:
#             aid = af["atomic_id"]
#             if aid not in seen:
#                 seen.add(aid)
#                 unique.append(af)
#         item["atomic_facts"] = unique

#     chunk_results = list(chunk_to_atomics.values())
#     if not chunk_results:
#         print("[INFO] No atomic-chunk mappings found (after text_atomics_db).")
#         return "No results."

#     print(f"[INFO] Step 2 complete — {len(chunk_results)} chunks mapped.")

#     # -------------------------
#     # Step 3. Rank chunks by max similarity
#     # -------------------------
#     for item in chunk_results:
#         sims = [af["score"] for af in item["atomic_facts"] if isinstance(af.get("score"), (int, float))]
#         item["chunk_score"] = max(sims, default=0.0)

#     top_k = min(10, len(chunk_results))
#     chunk_results = sorted(chunk_results, key=lambda x: x["chunk_score"], reverse=True)[:top_k]
#     print(f"[INFO] Step 3 complete — Selected top-{top_k} chunks.")
#     original_chunk_results = copy.deepcopy(chunk_results)

#     # -------------------------
#     # Step 3.5. Atomic Fact Expansion (triple-based intra-chunk)
#     # -------------------------
#     print(f"[INFO] Expanding atomic facts (triple-based intra-chunk) for {len(chunk_results)} chunks...")

#     all_atomic_ids = {af["atomic_id"] for item in chunk_results for af in item["atomic_facts"]}
#     atomic_records = await text_atomics_db.get_by_ids(list(all_atomic_ids))
#     atomic_to_triples = {}
#     for aid, rec in zip(all_atomic_ids, atomic_records or []):
#         if rec and isinstance(rec, dict):
#             atomic_to_triples[aid] = rec.get("triple_ids", []) or []

#     for item in chunk_results:
#         cid = item["chunk_id"]
#         existing_atomic_ids = {af["atomic_id"] for af in item["atomic_facts"]}
#         expanded = []

#         cdata = await text_chunks_db.get_by_ids([cid])
#         if not cdata or not isinstance(cdata[0], dict):
#             continue
#         all_chunk_atomic_ids = cdata[0].get("atomic_ids", [])
#         if not all_chunk_atomic_ids:
#             continue

#         chunk_atomic_records = await text_atomics_db.get_by_ids(all_chunk_atomic_ids)
#         chunk_atomic_info = []
#         for aid, rec in zip(all_chunk_atomic_ids, chunk_atomic_records or []):
#             if rec and isinstance(rec, dict):
#                 chunk_atomic_info.append({
#                     "atomic_id": aid,
#                     "content": rec.get("content", ""),
#                     "triple_ids": rec.get("triple_ids", []) or []
#                 })

#         selected_triples = set()
#         for af in item["atomic_facts"]:
#             aid = af["atomic_id"]
#             selected_triples.update(atomic_to_triples.get(aid, []))

#         for af in chunk_atomic_info:
#             aid = af["atomic_id"]
#             if aid in existing_atomic_ids:
#                 continue
#             overlap = selected_triples.intersection(set(af["triple_ids"]))
#             if overlap:
#                 expanded.append({
#                     "atomic_id": aid,
#                     "atomic_text": af["content"],
#                     "sub_query": "[triple-linked]",
#                     "score": 0.9
#                 })
#                 existing_atomic_ids.add(aid)

#         if expanded:
#             item["atomic_facts"].extend(expanded)
#             print(f"[INFO] Chunk {cid}: expanded {len(expanded)} atomic facts via shared triples.")

#     print(f"[INFO] Step 3.5 complete — Atomic fact expansion finished.")

#     # -------------------------
#     # Step 3.6. Save intermediate results for analysis
#     # -------------------------
#     print("[INFO] Saving intermediate results for analysis...")

#     save_dir = "/workspace/AtomRAG/logs"
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, "ours_experiment1_log.json")

#     log_data = {
#         "query": query,
#         "subqueries": queries,
#         "steps": {}
#     }

#     # Step 1
#     step1_log = {}
#     for sq, results in subquery_results.items():
#         step1_log[sq] = [
#             {
#                 "atomic_id": r.get("id"),
#                 "entity_name": r.get("entity_name", ""),
#                 "score": r.get("distance", 0.0),
#             }
#             for r in results[:10]
#         ]
#     log_data["steps"]["step1_retrieval"] = step1_log

#     # Step 2
#     step2_log = []
#     for item in original_chunk_results: 
#         chunk_text = item.get("chunk_text", "") or ""
#         step2_log.append({
#             "chunk_id": item.get("chunk_id", "UNKNOWN"),
#             "chunk_text_preview": chunk_text[:150] + "..." if len(chunk_text) > 150 else chunk_text,
#             "num_atomics": len(item.get("atomic_facts", [])),
#             "atomic_facts": [
#                 {
#                     "atomic_id": af.get("atomic_id"),
#                     "atomic_text": af.get("atomic_text"),
#                     "sub_query": af.get("sub_query"),
#                     "score": af.get("score")
#                 }
#                 for af in item.get("atomic_facts", [])
#             ]
#         })
#     log_data["steps"]["step2_mapping"] = step2_log

#     # Step 3
#     step3_log = [
#         {
#             "chunk_id": item.get("chunk_id"),
#             "chunk_score": item.get("chunk_score"),
#             "num_atomics": len(item.get("atomic_facts", [])),
#             "atomic_ids": [af.get("atomic_id") for af in item.get("atomic_facts", [])],
#         }
#         for item in chunk_results
#     ]
#     log_data["steps"]["step3_topk_chunks"] = step3_log

#     # Step 3.5
#     step35_log = []
#     for item in chunk_results:
#         new_added = [
#             af for af in item.get("atomic_facts", [])
#             if "[triple-linked]" in af.get("sub_query", "")
#         ]
#         if not new_added:
#             continue
#         step35_log.append({
#             "chunk_id": item.get("chunk_id"),
#             "num_expanded": len(new_added),
#             "expanded_atomics": [
#                 {
#                     "atomic_id": af.get("atomic_id"),
#                     "atomic_text": af.get("atomic_text"),
#                     "score": af.get("score")
#                 }
#                 for af in new_added
#             ]
#         })
#     log_data["steps"]["step35_expansion"] = step35_log

#     # -------------------------
#     # Step 3.7. Append results to JSON file (safe)
#     # -------------------------
#     try:
#         if os.path.exists(save_path):
#             with open(save_path, "r", encoding="utf-8") as f:
#                 try:
#                     existing_logs = json.load(f)
#                     if not isinstance(existing_logs, list):
#                         existing_logs = [existing_logs]
#                 except json.JSONDecodeError:
#                     print("[WARN] Existing log file corrupted or empty; reinitializing.")
#                     existing_logs = []
#         else:
#             existing_logs = []

#         existing_logs.append(log_data)
#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(existing_logs, f, ensure_ascii=False, indent=4)
#             f.flush()
#             os.fsync(f.fileno())

#         print(f"[INFO] ✅ Successfully appended analysis log to {save_path}")

#     except Exception as e:
#         print(f"[ERROR] ❌ Failed to save analysis log: {e}")

#     # -------------------------
#     # Step 4. Parallel rewriting
#     # -------------------------
#     print(f"[INFO] Step 4 — Running rewriting for {len(chunk_results)} chunks...")

#     async def rewrite_chunk(item, idx):
#         """각 chunk에 대해 atomic facts 기반 reasoning rewriting 수행"""
#         chunk_text = item["chunk_text"]
#         facts = item["atomic_facts"]
#         atomic_list = "\n".join(f"- {af['atomic_text']}" for af in facts if af.get("atomic_text"))

#         prompt = PROMPTS["chunk_rewriting"].format(
#             main_query=query,
#             chunk_text=chunk_text,
#             atomic_facts=atomic_list
#         )

#         try:
#             resp = await global_config["llm_model_func"](prompt, stream=param.stream)
#             rewriting_text = resp.strip() if isinstance(resp, str) else str(resp)
#             return {
#                 "chunk_id": item["chunk_id"],
#                 "chunk_text": chunk_text,
#                 "rewriting": rewriting_text,
#                 "atomic_facts": facts,
#                 "prompt": prompt,
#             }
#         except Exception as e:
#             print(f"[ERROR] Rewriting failed (chunk {idx}): {e}")
#             return None

#     rewriting_results = [
#         r for r in await asyncio.gather(*(rewrite_chunk(item, i) for i, item in enumerate(chunk_results))) if r
#     ]

#     if not rewriting_results:
#         print("[INFO] No rewriting results available.")
#         return PROMPTS.get("fail_response", "Rewriting failed or returned empty results.")

#     print(f"[INFO] Step 4 complete — Rewriting finished for {len(rewriting_results)} chunks.")

#     # -------------------------
#     # Step 5. Build final context
#     # -------------------------
#     def extract_field(text, key):
#         """LLM 응답 내 특정 필드([Relation], [Rewritten Evidence] 등) 추출"""
#         for line in text.splitlines():
#             if line.strip().startswith(f"[{key}]"):
#                 return line.split(":", 1)[-1].strip()
#         return None
    
#     rewriting_results_sorted = sorted(
#         rewriting_results,
#         key=lambda r: next(
#             (c["chunk_score"] for c in chunk_results if c["chunk_id"] == r["chunk_id"]),
#             0.0,
#         )
#     )

#     context_blocks = []
#     for i, r in enumerate(rewriting_results_sorted):
#         rewriting_text = r.get("rewriting", "").strip()
#         evidence = extract_field(rewriting_text, "Rewritten Evidence") or rewriting_text

#         context_blocks.append(
#             f"### Rewritten Evidence{i+1}: {evidence.strip()}\n"
#         )

#     if not context_blocks:
#         print("[INFO] No valid context (all unrelated). Returning fail response.")
#         return PROMPTS.get("fail_response", "No relevant chunks found.")

#     context = "\n\n".join(context_blocks)
#     print(f"[INFO] Step 5 complete — Final context built with {len(context_blocks)} rewritten chunks.")

#     # -------------------------
#     # Step 6. Final answer generation
#     # -------------------------
#     cls = os.path.basename(global_config.get("working_dir", ""))
#     sys_prompt_template = PROMPTS.get(f"rag_response_{cls}", PROMPTS["ours_rag_response"])
#     sys_prompt = sys_prompt_template.format(context_data=context)

#     print(f"[INFO] Step 6 — Generating final answer using context from {len(context_blocks)} chunks...")
#     print(f"[DEBUG] System prompt length: {len(sys_prompt)}")

#     try:
#         final_response = await global_config["llm_model_func"](
#             query,
#             system_prompt=sys_prompt,
#             stream=param.stream,
#         )
#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return PROMPTS.get("fail_response", "Final generation failed.")

#     print("[INFO] Final answer generation successful.")
#     if isinstance(final_response, str):
#         preview = final_response[:300] + ("..." if len(final_response) > 300 else "")
#         print(preview)

#     # -------------------------
#     # Step 7. Save full pipeline results
#     # -------------------------
#     try:
#         cls = os.path.basename(global_config.get("working_dir", ""))
#         save_dir = f"/workspace/AtomRAG/{cls}"
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f"{cls}_full_results.json")

#         # 전체 결과 구조 구성
#         full_output = {
#             "query": query,
#             "subqueries": queries,
#             "original_chunks": original_chunk_results,    # Step 3 전 버전
#             "expanded_chunks": chunk_results,             # Step 3.5 후 버전
#             "rewriting": rewriting_results,               # Step 4 결과
#             "final_context": context,
#             "final_answer": final_response if isinstance(final_response, str) else str(final_response),
#             "system_prompt": sys_prompt,
#         }

#         # append 저장
#         if os.path.exists(save_path):
#             with open(save_path, "r", encoding="utf-8") as f:
#                 try:
#                     existing = json.load(f)
#                     if not isinstance(existing, list):
#                         existing = [existing]
#                 except json.JSONDecodeError:
#                     existing = []
#         else:
#             existing = []

#         existing.append(full_output)
#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(existing, f, ensure_ascii=False, indent=4)

#         print(f"[INFO] Step 7 complete — Results appended to {save_path}")
#     except Exception as e:
#         print(f"[ERROR] Failed to save full results: {e}")

#     return final_response








# # ####### 일반 chunk만 넣고 진행한 방식 #########
# async def ours_kg_query_experiment1(
#     query: str,
#     atomic_knowledge_graph_inst,
#     atomic_entities_vdb,
#     text_triple_db,
#     text_atomics_db,
#     text_chunks_db,
#     param,
#     global_config: dict,
#     hashing_kv=None,
# ) -> str:

#     import os, json, copy, asyncio
#     from collections import defaultdict

#     # =======================================================
#     # Step 0. Query decomposition
#     # =======================================================
#     query_modes = {
#         "experiment1": query_process_1,
#         "experiment2": query_process_2,
#         "experiment3": query_process_3,
#         "experiment4": query_process_4,
#     }
#     queries = await query_modes.get(param.query_mode, lambda *a: [query])(
#         query, param, global_config, hashing_kv
#     )
#     if not queries:
#         queries = [query]

#     print(f"[INFO] Step 0 complete — {len(queries)} subqueries generated.")

#     # =======================================================
#     # Step 1. Atomic fact retrieval
#     # =======================================================
#     print(f"[INFO] Retrieving atomic facts for {len(queries)} subqueries...")

#     async def fetch_atomic_facts(sq):
#         try:
#             results = await atomic_entities_vdb.query(sq, top_k=10)
#             return sq, results
#         except Exception as e:
#             print(f"[ERROR] Retrieval failed for '{sq}': {e}")
#             return sq, []

#     subquery_results = dict(
#         await asyncio.gather(*(fetch_atomic_facts(sq) for sq in queries))
#     )

#     step1_retrieval_log = {
#         sq: [
#             {
#                 "atomic_id": r.get("id"),
#                 "entity_name": r.get("entity_name", ""),
#                 "score": r.get("distance", 0.0),
#             }
#             for r in results
#         ]
#         for sq, results in subquery_results.items()
#     }

#     # =======================================================
#     # Step 2. Atomic → Chunk mapping
#     # =======================================================

#     chunk_to_atomics = defaultdict(lambda: {"atomic_facts": []})
#     chunk_cache = {}

#     retrieved_items = []
#     for sq, results in subquery_results.items():
#         for r in results:
#             rid = r.get("id")
#             if rid:
#                 retrieved_items.append((sq, rid, r.get("distance", 0.0)))

#     if not retrieved_items:
#         return "No results."

#     unique_atomic_ids = list({rid for _, rid, _ in retrieved_items})
#     atomic_records = await text_atomics_db.get_by_ids(unique_atomic_ids)

#     atomic_info_map = {}
#     for aid, rec in zip(unique_atomic_ids, atomic_records or []):
#         if rec and isinstance(rec, dict):
#             atomic_info_map[aid] = {
#                 "content": rec.get("content", ""),
#                 "source_ids": rec.get("source_ids", []) or [],
#             }

#     for sq, aid, score in retrieved_items:
#         ainfo = atomic_info_map.get(aid)
#         if not ainfo:
#             continue

#         atomic_text = ainfo["content"]
#         src_chunks = ainfo["source_ids"]

#         for cid in src_chunks:
#             if cid not in chunk_cache:
#                 try:
#                     cdata = await text_chunks_db.get_by_ids([cid])
#                     chunk_cache[cid] = cdata[0].get("content", "") if cdata else ""
#                 except Exception:
#                     chunk_cache[cid] = ""

#             chunk_to_atomics[cid]["chunk_id"] = cid
#             chunk_to_atomics[cid]["chunk_text"] = chunk_cache[cid]
#             chunk_to_atomics[cid]["atomic_facts"].append({
#                 "atomic_id": aid,
#                 "atomic_text": atomic_text,
#                 "sub_query": sq,
#                 "score": score,
#             })

#     for item in chunk_to_atomics.values():
#         seen = set()
#         uniq = []
#         for af in item["atomic_facts"]:
#             aid = af["atomic_id"]
#             if aid not in seen:
#                 seen.add(aid)
#                 uniq.append(af)
#         item["atomic_facts"] = uniq

#     chunk_results = list(chunk_to_atomics.values())
#     print(f"[INFO] Step 2 complete — {len(chunk_results)} chunks mapped.")

#     step2_mapping_log = []
#     for item in chunk_results:
#         txt = item.get("chunk_text", "")
#         step2_mapping_log.append({
#             "chunk_id": item["chunk_id"],
#             "chunk_text_preview": txt[:150] + ("..." if len(txt) > 150 else ""),
#             "num_atomics": len(item["atomic_facts"]),
#             "atomic_facts": item["atomic_facts"],
#         })

#     # =======================================================
#     # Step 3. Rank chunks (ONLY max_atomic_score)
#     # =======================================================

#     for item in chunk_results:
#         item["max_atomic_score"] = max([af["score"] for af in item["atomic_facts"]], default=0.0)

#     chunk_results_sorted = sorted(
#         chunk_results,
#         key=lambda x: x["max_atomic_score"],
#         reverse=True,
#     )

#     top_k = min(10, len(chunk_results_sorted))
#     chunk_results_sorted = chunk_results_sorted[:top_k]

#     print(f"[INFO] Step 3 complete — Selected top-{top_k} chunks (by max_atomic_score).")

#     step3_topk_log = [
#         {
#             "chunk_id": item["chunk_id"],
#             "max_atomic_score": item["max_atomic_score"],
#             "atomic_ids": [af["atomic_id"] for af in item["atomic_facts"]],
#         }
#         for item in chunk_results_sorted
#     ]

#     # =======================================================
#     # Step 4. Build context (NO relevance label)
#     # =======================================================

#     context_blocks = []
#     for idx, item in enumerate(chunk_results_sorted):
#         chunk_text = item.get("chunk_text", "").strip()
#         context_blocks.append(
#             f"### Retrieved Chunk {idx+1}\n{chunk_text}\n"
#         )

#     context = "\n\n".join(context_blocks)
#     print("[INFO] Step 4 complete — Context built (NO labels).")

#     # =======================================================
#     # Step 5. Final answer generation
#     # =======================================================

#     cls = os.path.basename(global_config.get("working_dir", ""))
#     sys_prompt_template = PROMPTS.get("rag_response_origin")
#     sys_prompt = sys_prompt_template.format(context_data=context)

#     try:
#         final_response = await global_config["llm_model_func"](
#             query,
#             system_prompt=sys_prompt,
#             stream=param.stream,
#         )
#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return "Final generation failed."

#     # =======================================================
#     # Step 6. Save FULL JSON LOG
#     # =======================================================

#     try:
#         save_dir = f"/workspace/AtomRAG/{cls}"
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f"{cls}_full_results.json")

#         full_output = {
#             "query": query,
#             "subqueries": queries,
#             "step1_retrieval": step1_retrieval_log,
#             "step2_mapping": step2_mapping_log,
#             "step3_topk_chunks": step3_topk_log,
#             "final_context": context,
#             "final_answer": final_response if isinstance(final_response, str) else str(final_response),
#             "system_prompt": sys_prompt,
#         }

#         if os.path.exists(save_path):
#             with open(save_path, "r", encoding="utf-8") as f:
#                 try:
#                     existing = json.load(f)
#                     if not isinstance(existing, list):
#                         existing = [existing]
#                 except:
#                     existing = []
#         else:
#             existing = []

#         existing.append(full_output)

#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(existing, f, ensure_ascii=False, indent=4)

#         print(f"[INFO] JSON saved to {save_path}")

#     except Exception as e:
#         print(f"[ERROR] Saving JSON failed: {e}")

#     print(final_response)
#     return final_response






# ####### 일반 chunk 넣고 atomic fact count와 atomic fact score를 함꼐 고려해서 reranking하고 관련성을 표시한 방식#########
# async def ours_kg_query_experiment1(
#     query: str,
#     atomic_knowledge_graph_inst,
#     atomic_entities_vdb,
#     text_triple_db,
#     text_atomics_db,
#     text_chunks_db,
#     param,
#     global_config: dict,
#     hashing_kv=None,
# ) -> str:
#     """
#     Simplified version:
#     - Run Step 0~3 (retrieval)
#     - Use top-K raw chunks as context
#     - Generate final answer
#     - Save entire reasoning chain to JSON file
#     """

#     import os, json, copy, asyncio
#     from collections import defaultdict

#     # =======================================================
#     # Step 0. Query decomposition
#     # =======================================================
#     query_modes = {
#         "experiment1": query_process_1,
#         "experiment2": query_process_2,
#         "experiment3": query_process_3,
#         "experiment4": query_process_4,
#     }
#     queries = await query_modes.get(param.query_mode, lambda *a: [query])(
#         query, param, global_config, hashing_kv
#     )
#     if not queries:
#         queries = [query]

#     print(f"[INFO] Step 0 complete — {len(queries)} subqueries generated.")

#     # =======================================================
#     # Step 1. Atomic fact retrieval
#     # =======================================================
#     print(f"[INFO] Retrieving atomic facts for {len(queries)} subqueries...")

#     async def fetch_atomic_facts(sq):
#         try:
#             results = await atomic_entities_vdb.query(sq, top_k=10)
#             return sq, results
#         except Exception as e:
#             print(f"[ERROR] Retrieval failed for '{sq}': {e}")
#             return sq, []

#     subquery_results = dict(
#         await asyncio.gather(*(fetch_atomic_facts(sq) for sq in queries))
#     )

#     # store logs
#     step1_retrieval_log = {
#         sq: [
#             {
#                 "atomic_id": r.get("id"),
#                 "entity_name": r.get("entity_name", ""),
#                 "score": r.get("distance", 0.0),
#             }
#             for r in results
#         ]
#         for sq, results in subquery_results.items()
#     }

#     # =======================================================
#     # Step 2. Atomic → Chunk mapping
#     # =======================================================
#     chunk_to_atomics = defaultdict(lambda: {"atomic_facts": []})
#     chunk_cache = {}

#     retrieved_items = []
#     for sq, results in subquery_results.items():
#         for r in results:
#             rid = r.get("id")
#             if rid:
#                 retrieved_items.append((sq, rid, r.get("distance", 0.0)))

#     if not retrieved_items:
#         return "No results."

#     unique_atomic_ids = list({rid for _, rid, _ in retrieved_items})
#     atomic_records = await text_atomics_db.get_by_ids(unique_atomic_ids)

#     atomic_info_map = {}
#     for aid, rec in zip(unique_atomic_ids, atomic_records or []):
#         if rec and isinstance(rec, dict):
#             atomic_info_map[aid] = {
#                 "content": rec.get("content", ""),
#                 "source_ids": rec.get("source_ids", []) or [],
#             }

#     # mapping
#     for sq, aid, score in retrieved_items:
#         ainfo = atomic_info_map.get(aid)
#         if not ainfo:
#             continue

#         atomic_text = ainfo["content"]
#         src_chunks = ainfo["source_ids"]

#         for cid in src_chunks:
#             # load chunk on demand
#             if cid not in chunk_cache:
#                 try:
#                     cdata = await text_chunks_db.get_by_ids([cid])
#                     chunk_cache[cid] = cdata[0].get("content", "") if cdata else ""
#                 except Exception:
#                     chunk_cache[cid] = ""

#             chunk_to_atomics[cid]["chunk_id"] = cid
#             chunk_to_atomics[cid]["chunk_text"] = chunk_cache[cid]
#             chunk_to_atomics[cid]["atomic_facts"].append({
#                 "atomic_id": aid,
#                 "atomic_text": atomic_text,
#                 "sub_query": sq,
#                 "score": score,
#             })

#     # remove duplicates
#     for item in chunk_to_atomics.values():
#         seen = set()
#         uniq = []
#         for af in item["atomic_facts"]:
#             aid = af["atomic_id"]
#             if aid not in seen:
#                 seen.add(aid)
#                 uniq.append(af)
#         item["atomic_facts"] = uniq

#     chunk_results = list(chunk_to_atomics.values())
#     print(f"[INFO] Step 2 complete — {len(chunk_results)} chunks mapped.")

#     # Log step2
#     step2_mapping_log = []
#     for item in chunk_results:
#         txt = item.get("chunk_text", "")
#         step2_mapping_log.append({
#             "chunk_id": item["chunk_id"],
#             "chunk_text_preview": txt[:150] + ("..." if len(txt) > 150 else ""),
#             "num_atomics": len(item["atomic_facts"]),
#             "atomic_facts": item["atomic_facts"],
#         })

#     # =======================================================
#     # Step 3. Rank chunks (NEW combined scoring)
#     # =======================================================

#     # 1) 먼저 각 chunk에 대해 atomic fact 개수와 max atomic score 계산
#     for item in chunk_results:
#         num_atomics = len(item["atomic_facts"])
#         max_score = max([af["score"] for af in item["atomic_facts"]], default=0.0)

#         item["num_atomics"] = num_atomics
#         item["max_atomic_score"] = max_score

#     # 2) normalization (min-max)
#     def safe_norm(val, min_v, max_v):
#         if max_v - min_v < 1e-9:
#             return 0.0
#         return (val - min_v) / (max_v - min_v)

#     all_nums = [x["num_atomics"] for x in chunk_results]
#     all_scores = [x["max_atomic_score"] for x in chunk_results]

#     min_num, max_num = min(all_nums), max(all_nums)
#     min_score, max_score = min(all_scores), max(all_scores)

#     # 3) Combined Score
#     w1 = 0.6  # atomic_fact_count weight
#     w2 = 0.4  # max_atomic_score weight

#     for item in chunk_results:
#         norm_num = safe_norm(item["num_atomics"], min_num, max_num)
#         norm_score = safe_norm(item["max_atomic_score"], min_score, max_score)

#         item["chunk_score"] = w1 * norm_num + w2 * norm_score

#     # top-3 → top-5 → top-10
#     top_k = min(10, len(chunk_results))
#     chunk_results_sorted = sorted(chunk_results, key=lambda x: x["chunk_score"], reverse=True)[:top_k]

#     print(f"[INFO] Step 3 complete — Selected top-{top_k} chunks (combined scoring).")

#     # Log update
#     step3_topk_log = [
#         {
#             "chunk_id": item["chunk_id"],
#             "chunk_score": item["chunk_score"],
#             "num_atomics": item["num_atomics"],
#             "max_atomic_score": item["max_atomic_score"],
#             "atomic_ids": [af["atomic_id"] for af in item["atomic_facts"]],
#         }
#         for item in chunk_results_sorted
#     ]

#     # =======================================================
#     # Step 4. Build context with relevance labels
#     # =======================================================

#     # relevance labels
#     def relevance_label(rank):
#         if rank <= 3:
#             return "[HIGHLY RELEVANT]"
#         elif rank <= 6:
#             return "[MODERATELY RELEVANT]"
#         else:
#             return "[LIGHTLY RELEVANT]"

#     chunk_results_sorted = sorted(chunk_results_sorted, key=lambda x: x["chunk_score"], reverse=True)

#     context_blocks = []
#     for idx, item in enumerate(chunk_results_sorted):
#         chunk_text = item.get("chunk_text", "").strip()
#         label = relevance_label(idx + 1)
#         context_blocks.append(
#             f"### Retrieved Chunk {idx+1} {label}\n{chunk_text}\n"
#         )

#     context = "\n\n".join(context_blocks)
#     print("[INFO] Step 4 complete — Context built (with relevance labels).")


#     # =======================================================
#     # Step 5. Final answer generation
#     # =======================================================
#     cls = os.path.basename(global_config.get("working_dir", ""))
#     sys_prompt_template = PROMPTS.get("rag_response_origin")
#     sys_prompt = sys_prompt_template.format(context_data=context)

#     try:
#         final_response = await global_config["llm_model_func"](
#             query,
#             system_prompt=sys_prompt,
#             stream=param.stream,
#         )
#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return "Final generation failed."

#     # =======================================================
#     # Step 6. Save FULL JSON LOG
#     # =======================================================
#     try:
#         save_dir = f"/workspace/AtomRAG/{cls}"
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f"{cls}_full_results.json")

#         full_output = {
#             "query": query,
#             "subqueries": queries,
#             "step1_retrieval": step1_retrieval_log,
#             "step2_mapping": step2_mapping_log,
#             "step3_topk_chunks": step3_topk_log,
#             "final_context": context,
#             "final_answer": final_response if isinstance(final_response, str) else str(final_response),
#             "system_prompt": sys_prompt,
#         }

#         # append mode
#         if os.path.exists(save_path):
#             with open(save_path, "r", encoding="utf-8") as f:
#                 try:
#                     existing = json.load(f)
#                     if not isinstance(existing, list):
#                         existing = [existing]
#                 except:
#                     existing = []
#         else:
#             existing = []

#         existing.append(full_output)

#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(existing, f, ensure_ascii=False, indent=4)

#         print(f"[INFO] JSON saved to {save_path}")

#     except Exception as e:
#         print(f"[ERROR] Saving JSON failed: {e}")
        
#     print(final_response)

#     return final_response


# ####### 1차(query로 chunk 추출) + 2차(subquery로 chunk reranking) cot방식의 generation을 사용한 방법#########
# async def ours_kg_query_experiment1(
#     query: str,
#     atomic_knowledge_graph_inst,
#     atomic_entities_vdb,
#     chunks_vdb,
#     text_triple_db,
#     text_atomics_db,
#     text_chunks_db,
#     param,
#     global_config: dict,
#     hashing_kv=None,
# ) -> str:
#     """
#     Simplified version:
#     - Run Step 0~3 (retrieval)
#     - Use top-K raw chunks as context
#     - Generate final answer using structured-CoT RAG format
#     - Save entire reasoning chain to JSON file
#     """

#     # =======================================================
#     # Step 0. Query decomposition (same)
#     # =======================================================
#     query_modes = {
#         "experiment1": query_process_1,
#         "experiment2": query_process_2,
#         "experiment3": query_process_3,
#         "experiment4": query_process_4,
#     }
#     queries = await query_modes.get(param.query_mode, lambda *a: [query])(
#         query, param, global_config, hashing_kv
#     )
#     if not queries:
#         queries = [query]
#     print(f"[INFO] Step 0 complete — {len(queries)} subqueries generated.")


#     # =======================================================
#     # Step 1. Chunk-first retrieval (Top-200 by main query)
#     # =======================================================
#     print(f"[INFO] Retrieving chunks via chunks_vdb...")

#     try:
#         chunk_candidates = await chunks_vdb.query(query, top_k=200)
#     except Exception as e:
#         print(f"[ERROR] Chunk retrieval failed: {e}")
#         return "Chunk retrieval failed."

#     if not chunk_candidates:
#         print("[WARN] No chunks retrieved.")
#         return "No chunks retrieved."

#     # chunk structure: {chunk_id: {... placeholder ...}}
#     chunk_info = {}
#     for c in chunk_candidates:
#         cid = c.get("id")
#         if not cid:
#             continue

#         # text loading from text_chunks_db
#         try:
#             cdata = await text_chunks_db.get_by_ids([cid])
#             cdata = cdata[0] if cdata else {}
#         except Exception as e:
#             print(f"[WARN] Failed to fetch chunk {cid}: {e}")
#             cdata = {}

#         chunk_text = cdata.get("content", "")
#         atomic_ids = cdata.get("atomic_ids", [])

#         chunk_info[cid] = {
#             "chunk_id": cid,
#             "chunk_text": chunk_text,
#             "atomic_ids": atomic_ids,
#             "atomic_facts": [],
#             "max_score": 0.0
#         }

#     print(f"[INFO] Step 1 complete — Retrieved and resolved {len(chunk_info)} chunks.")


#     # =======================================================
#     # Step 2. Load Atomic Facts for these chunks (Fixed)
#     # =======================================================
#     print("[INFO] Step 2 — Collecting atomic IDs inside retrieved chunks...")


#     # =======================================================
#     # Step 3. Optimized Atomic Similarity (Batch Scoring)
#     # =======================================================

#     print("[INFO] Optimized scoring — single query per subquery")

#     # 1) Gather all atomic_ids from Top-200 chunks
#     all_atomic_ids = set()
#     for info in chunk_info.values():
#         all_atomic_ids.update(info["atomic_ids"])
#     all_atomic_ids = list(all_atomic_ids)

#     if not all_atomic_ids:
#         print("[WARN] No atomic facts available in candidate chunks.")
#         return "No atomic facts available."

#     allowed_ids_set = set(all_atomic_ids)

#     # 2) Score all atomic_ids for each subquery
#     atomic_scores = {aid: 0.0 for aid in all_atomic_ids}

#     async def score_single_subquery(sq):
#         # filter only target atomic IDs
#         filter_fn = (lambda dp, allowed=allowed_ids_set:
#                     dp["__id__"] in allowed)

#         # query ALL allowed atomic IDs at once
#         results = await atomic_entities_vdb.query(
#             query=sq,
#             top_k=len(all_atomic_ids),
#             filter_lambda=filter_fn
#         )

#         # update max scores
#         for r in results:
#             aid = r["id"]
#             score = r["distance"]
#             atomic_scores[aid] = max(atomic_scores[aid], score)

#     # Run subqueries asynchronously
#     await asyncio.gather(*(score_single_subquery(sq) for sq in queries))

#     # 3) Assign best atomic score to each chunk
#     for cid, info in chunk_info.items():
#         chunk_aids = info["atomic_ids"]
#         info["max_score"] = max(
#             (atomic_scores.get(aid, 0.0) for aid in chunk_aids),
#             default=0.0
#         )

#     # 4) Rank → Top-10 selection
#     chunk_results_sorted = sorted(
#         chunk_info.values(),
#         key=lambda x: x["max_score"],
#         reverse=True
#     )[:5]

#     print(f"[INFO] Step 3 complete — Selected Top-10 ranked chunks (Optimized).")


#     # =======================================================
#     # Step 4. Build simple context (NO relevance labels)
#     # =======================================================

#     context_blocks = []
#     for item in chunk_results_sorted:
#         chunk_text = item.get("chunk_text", "").strip()
#         context_blocks.append(chunk_text)

#     context = "\n\n".join(context_blocks)
#     print("[INFO] Step 4 complete — Context built (no labels).")

#     # =======================================================
#     # Step 5. FINAL ANSWER GENERATION (HOTPOT vs MULTIHOP BY cls)
#     # =======================================================

#     # working_dir 기준으로 어떤 데이터셋인지 판단
#     cls = os.path.basename(global_config.get("working_dir", ""))

#     # ---------------------------
#     # HOTPOT one-shot prompt
#     # ---------------------------
#     hotpot_system = (
#         'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
#         'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
#         'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
#     )

#     hotpot_docs = (
#         """The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
#         """Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
#         """Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
#         """Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
#         """Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"""
#     )

#     hotpot_input = (
#         f"{hotpot_docs}"
#         "\n\nQuestion: "
#         "When was Neville A. Stanton's employer founded?"
#         "\nThought: "
#     )

#     hotpot_output = (
#         "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
#         "\nAnswer: 1862."
#     )

#     # ---------------------------
#     # MULTIHOP one-shot prompt
#     # ---------------------------
#     multihop_system = (
#         'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
#         'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
#         'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
#         'Use only the given Information; if it is insufficient, reply with "Answer: Insufficient information.".'
#         'If you need to answer like yes or no, use "Answer: Yes" or "Answer: No" only.'
#     )

#     multihop_docs = (
#         """Milk and Honey (album)\nMilk and Honey is an album by John Lennon and Yoko Ono released in 1984. Following the compilation "The John Lennon Collection", it is Lennon's eighth and final studio album, and the first posthumous release of new Lennon music, having been recorded in the last months of his life during and following the sessions for their 1980 album "Double Fantasy". It was assembled by Yoko Ono in association with the Geffen label.\n\n"""
#         """John Lennon Museum\nJohn Lennon Museum (ジョン・レノン・ミュージアム , Jon Renon Myūjiamu ) was a museum located inside the Saitama Super Arena in Chūō-ku, Saitama, Saitama Prefecture, Japan. It was established to preserve knowledge of John Lennon's life and musical career. It displayed Lennon's widow Yoko Ono's collection of his memorabilia as well as other displays. The museum opened on October 9, 2000, the 60th anniversary of Lennon's birth, and closed on September 30, 2010, when its exhibit contract with Yoko Ono expired. A tour of the museum began with a welcoming message and short film narrated by Yoko Ono (in Japanese with English headphones available), and ended at an avant-garde styled "reflection room" full of chairs facing a slide show of moving words and images. After this room there was a gift shop with John Lennon memorabilia available.\n\n"""
#         """Walls and Bridges\nWalls and Bridges is the fifth studio album by English musician John Lennon. It was issued by Apple Records on 26 September 1974 in the United States and on 4 October in the United Kingdom. Written, recorded and released during his 18-month separation from Yoko Ono, the album captured Lennon in the midst of his "Lost Weekend". "Walls and Bridges" was an American "Billboard" number-one album and featured two hit singles, "Whatever Gets You thru the Night" and "#9 Dream". The first of these was Lennon's first number-one hit in the United States as a solo artist, and his only chart-topping single in either the US or Britain during his lifetime.\n\n"""
#         """Nobody Loves You (When You're Down and Out)\n"Nobody Loves You (When You're Down and Out)" is a song written by John Lennon released on his 1974 album "Walls and Bridges". The song is included on the 1986 compilation "Menlove Ave.", the 1990 boxset "Lennon", the 1998 boxset "John Lennon Anthology", the 2005 two-disc compilation "", and the 2010 boxset "Gimme Some Truth".\n\n"""
#         """Give Peace a Chance\n"Give Peace a Chance" is an anti-war song written by John Lennon (credited to Lennon–McCartney), and performed with Yoko Ono in Montreal, Quebec, Canada. Released as a single in 1969 by the Plastic Ono Band on Apple Records (catalogue Apple 13 in the United Kingdom, Apple 1809 in the United States), it is the first solo single issued by Lennon, released when he was still a member of the Beatles, and became an anthem of the American anti-war movement during the 1970s. It peaked at number 14 on the "Billboard" Hot 100 and number 2 on the British singles chart.\n"""
#     )

#     multihop_input = (
#         f"{multihop_docs}"
#         "\n\nQuestion: "
#         "Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?"
#         "\nThought: "
#     )

#     multihop_output = (
#         "The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges. Nobody Loves You was written by John Lennon on Walls and Bridges album. "
#         "\nAnswer: Walls and Bridges."
#     )

#     # -------------------------------------------------------
#     # cls 이름을 기준으로 사용할 프롬프트 선택
#     # -------------------------------------------------------
#     cls_lower = cls.lower()
#     if "multihop" in cls_lower:
#         system_prompt = multihop_system
#         one_shot_input = multihop_input
#         one_shot_output = multihop_output
#     else:
#         # 기본값 hotpot (cls에 hotpot이 포함되어 있거나, 그 외 케이스)
#         system_prompt = hotpot_system
#         one_shot_input = hotpot_input
#         one_shot_output = hotpot_output

#     # -------------------------------------------------------
#     # 최종 유저 프롬프트 (retrieved context + query)
#     # -------------------------------------------------------
#     final_user_prompt = (
#         f"{context}"
#         f"\n\nQuestion: {query}"
#         "\nThought: "
#     )

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": one_shot_input},
#         {"role": "assistant", "content": one_shot_output},
#         {"role": "user", "content": final_user_prompt},
#     ]

#     try:
#         prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])

#         final_response = await global_config["llm_model_func"](
#             prompt=prompt,
#             stream=getattr(param, "stream", False),
#         )
#         def extract_answer(response: str) -> str:
#             if "Answer:" in response:
#                 return response.split("Answer:", 1)[1].strip()
#             return response.strip()

#         final_response = extract_answer(final_response)
#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return "Final generation failed."


#     # =======================================================
#     # Step 6. SAVE FULL JSON LOG
#     # =======================================================
#     try:
#         # cls는 위에서 이미 계산됨
#         save_dir = f"/workspace/AtomRAG/{cls}"
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f"{cls}_full_results.json")

#         full_output = {
#             "query": query,
#             "dataset_cls": cls,
#             "subqueries": queries,
#             "final_context": context,
#             "messages_prompt_used": messages,
#             "final_answer": (
#                 final_response if isinstance(final_response, str)
#                 else str(final_response)
#             ),
#         }

#         if os.path.exists(save_path):
#             with open(save_path, "r", encoding="utf-8") as f:
#                 try:
#                     existing = json.load(f)
#                     if not isinstance(existing, list):
#                         existing = [existing]
#                 except Exception:
#                     existing = []
#         else:
#             existing = []

#         existing.append(full_output)

#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(existing, f, ensure_ascii=False, indent=4)

#         print(f"[INFO] JSON saved to {save_path}")

#     except Exception as e:
#         print(f"[ERROR] Saving JSON failed: {e}")

#     print(final_response)
    
#     return final_response


# ###### Query와 Chunk를 비교한 score와 SubQuery와 Atomic fact를 비교한 score 방식의 rerank + cot방식의 generation을 사용한 방법#########
# async def ours_kg_query_experiment1(
#     query: str,
#     atomic_knowledge_graph_inst,
#     atomic_entities_vdb,
#     chunks_vdb,
#     text_triple_db,
#     text_atomics_db,
#     text_chunks_db,
#     param,
#     global_config: dict,
#     hashing_kv=None,
# ) -> str:
#     """
#     Simplified version:
#     - Run Step 0~3 (retrieval)
#     - Use top-K raw chunks as context
#     - Generate final answer using structured-CoT RAG format
#     - Save entire reasoning chain to JSON file
#     """

#     # =======================================================
#     # Step 0. Query decomposition
#     # =======================================================
#     query_modes = {
#         "experiment1": query_process_1,
#         "experiment2": query_process_2,
#         "experiment3": query_process_3,
#         "experiment4": query_process_4,
#     }
#     queries = await query_modes.get(param.query_mode, lambda *a: [query])(
#         query, param, global_config, hashing_kv
#     )
#     if not queries:
#         queries = [query]

#     print(f"[INFO] Step 0 complete — {len(queries)} subqueries generated.")


#     # =======================================================
#     # Step 1. Retrieve *ALL* chunks by query(=baseline signal)
#     # =======================================================
#     print("[INFO] Retrieving ALL chunks via chunks_vdb for main query scoring...")

#     all_chunks = await chunks_vdb.query(query, top_k=len(chunks_vdb._client))

#     chunk_info = {}
#     for c in all_chunks:
#         cid = c.get("id")
#         if not cid:
#             continue
        
#         main_score = c.get("distance", 0.0)  # query vs chunk

#         cdata = await text_chunks_db.get_by_ids([cid])
#         cdata = cdata[0] if cdata else {}

#         chunk_info[cid] = {
#             "chunk_id": cid,
#             "chunk_text": cdata.get("content", ""),
#             "atomic_ids": cdata.get("atomic_ids", []),
#             "main_score": main_score,
#             "sub_scores": [],
#             "final_score": 0.0,
#         }

#     print(f"[INFO] Step 1 complete — total chunks: {len(chunk_info)}")


#     # =======================================================
#     # Step 2. Atomic similarity by subqueries (batch per subq)
#     # =======================================================
#     print("[INFO] Computing batch atomic similarity for each subquery...")

#     # gather all atomic IDs from all chunks
#     all_atomic_ids = set()
#     for info in chunk_info.values():
#         all_atomic_ids.update(info["atomic_ids"])
#     all_atomic_ids = list(all_atomic_ids)

#     if not all_atomic_ids:
#         print("[WARN] No atomic facts in ALL chunks.")
#         return "No atomic facts available."

#     allowed_atomic_ids = set(all_atomic_ids)

#     # per chunk → fill zeros first
#     for info in chunk_info.values():
#         info["sub_scores"] = [0.0] * len(queries)


#     async def score_single_subquery(subq_idx, sq):
#         filter_fn = (lambda dp, allowed=allowed_atomic_ids:
#                         dp["__id__"] in allowed)

#         results = await atomic_entities_vdb.query(
#             query=sq,
#             top_k=len(atomic_entities_vdb._client),
#             filter_lambda=filter_fn,
#         )

#         # map atomic score
#         temp_scores = {r["id"]: r["distance"] for r in results}

#         # assign chunk-level max score
#         for cid, info in chunk_info.items():
#             scores_for_chunk = [
#                 temp_scores.get(aid, 0.0)
#                 for aid in info["atomic_ids"]
#             ]
#             sub_score = max(scores_for_chunk) if scores_for_chunk else 0.0
#             info["sub_scores"][subq_idx] = sub_score


#     await asyncio.gather(*(
#         score_single_subquery(i, sq) for i, sq in enumerate(queries)
#     ))


#     # =======================================================
#     # Step 3. Final score = mean(main_score + sub_scores)
#     # =======================================================
#     print("[INFO] Computing final aggregated chunk scores (softmax-weighted)...")

#     for cid, info in chunk_info.items():
#         # M, s1, s2, s3 ...
#         scores = [info["main_score"]] + info["sub_scores"]

#         # Softmax weights
#         exp_scores = [math.exp(s) for s in scores]
#         Z = sum(exp_scores)

#         if Z == 0:
#             # fallback: simple mean (rare case)
#             info["final_score"] = sum(scores) / len(scores)
#         else:
#             # softmax weighted score
#             info["final_score"] = sum((e / Z) * s for e, s in zip(exp_scores, scores))

#     # Top-10 by final score
#     chunk_results_sorted = sorted(
#         chunk_info.values(),
#         key=lambda x: x["final_score"],
#         reverse=True
#     )[:10]

#     print(f"[INFO] Step 3 done — Top-10 chunks selected.")


#     # =======================================================
#     # Step 4. Build simple context (NO relevance labels)
#     # =======================================================

#     context_blocks = []
#     for item in chunk_results_sorted:
#         chunk_text = item.get("chunk_text", "").strip()
#         context_blocks.append(chunk_text)

#     context = "\n\n".join(context_blocks)
#     print("[INFO] Step 4 complete — Context built (no labels).")

#     # =======================================================
#     # Step 5. FINAL ANSWER GENERATION (HOTPOT vs MULTIHOP BY cls)
#     # =======================================================

#     # working_dir 기준으로 어떤 데이터셋인지 판단
#     cls = os.path.basename(global_config.get("working_dir", ""))

#     # ---------------------------
#     # HOTPOT one-shot prompt
#     # ---------------------------
#     hotpot_system = (
#         'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
#         'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
#         'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
#     )

#     hotpot_docs = (
#         """The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
#         """Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
#         """Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
#         """Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
#         """Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"""
#     )

#     hotpot_input = (
#         f"{hotpot_docs}"
#         "\n\nQuestion: "
#         "When was Neville A. Stanton's employer founded?"
#         "\nThought: "
#     )

#     hotpot_output = (
#         "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
#         "\nAnswer: 1862."
#     )

#     # ---------------------------
#     # MULTIHOP one-shot prompt
#     # ---------------------------
#     multihop_system = (
#         'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
#         'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
#         'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
#         'Use only the given Information; if it is insufficient, reply with "Answer: Insufficient information.".'
#         'If you need to answer like yes or no, use "Answer: Yes" or "Answer: No" only.'
#     )

#     multihop_docs = (
#         """Milk and Honey (album)\nMilk and Honey is an album by John Lennon and Yoko Ono released in 1984. Following the compilation "The John Lennon Collection", it is Lennon's eighth and final studio album, and the first posthumous release of new Lennon music, having been recorded in the last months of his life during and following the sessions for their 1980 album "Double Fantasy". It was assembled by Yoko Ono in association with the Geffen label.\n\n"""
#         """John Lennon Museum\nJohn Lennon Museum (ジョン・レノン・ミュージアム , Jon Renon Myūjiamu ) was a museum located inside the Saitama Super Arena in Chūō-ku, Saitama, Saitama Prefecture, Japan. It was established to preserve knowledge of John Lennon's life and musical career. It displayed Lennon's widow Yoko Ono's collection of his memorabilia as well as other displays. The museum opened on October 9, 2000, the 60th anniversary of Lennon's birth, and closed on September 30, 2010, when its exhibit contract with Yoko Ono expired. A tour of the museum began with a welcoming message and short film narrated by Yoko Ono (in Japanese with English headphones available), and ended at an avant-garde styled "reflection room" full of chairs facing a slide show of moving words and images. After this room there was a gift shop with John Lennon memorabilia available.\n\n"""
#         """Walls and Bridges\nWalls and Bridges is the fifth studio album by English musician John Lennon. It was issued by Apple Records on 26 September 1974 in the United States and on 4 October in the United Kingdom. Written, recorded and released during his 18-month separation from Yoko Ono, the album captured Lennon in the midst of his "Lost Weekend". "Walls and Bridges" was an American "Billboard" number-one album and featured two hit singles, "Whatever Gets You thru the Night" and "#9 Dream". The first of these was Lennon's first number-one hit in the United States as a solo artist, and his only chart-topping single in either the US or Britain during his lifetime.\n\n"""
#         """Nobody Loves You (When You're Down and Out)\n"Nobody Loves You (When You're Down and Out)" is a song written by John Lennon released on his 1974 album "Walls and Bridges". The song is included on the 1986 compilation "Menlove Ave.", the 1990 boxset "Lennon", the 1998 boxset "John Lennon Anthology", the 2005 two-disc compilation "", and the 2010 boxset "Gimme Some Truth".\n\n"""
#         """Give Peace a Chance\n"Give Peace a Chance" is an anti-war song written by John Lennon (credited to Lennon–McCartney), and performed with Yoko Ono in Montreal, Quebec, Canada. Released as a single in 1969 by the Plastic Ono Band on Apple Records (catalogue Apple 13 in the United Kingdom, Apple 1809 in the United States), it is the first solo single issued by Lennon, released when he was still a member of the Beatles, and became an anthem of the American anti-war movement during the 1970s. It peaked at number 14 on the "Billboard" Hot 100 and number 2 on the British singles chart.\n"""
#     )

#     multihop_input = (
#         f"{multihop_docs}"
#         "\n\nQuestion: "
#         "Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?"
#         "\nThought: "
#     )

#     multihop_output = (
#         "The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges. Nobody Loves You was written by John Lennon on Walls and Bridges album. "
#         "\nAnswer: Walls and Bridges."
#     )

#     # -------------------------------------------------------
#     # cls 이름을 기준으로 사용할 프롬프트 선택
#     # -------------------------------------------------------
#     cls_lower = cls.lower()
#     if "multihop" in cls_lower:
#         system_prompt = multihop_system
#         one_shot_input = multihop_input
#         one_shot_output = multihop_output
#     else:
#         # 기본값 hotpot (cls에 hotpot이 포함되어 있거나, 그 외 케이스)
#         system_prompt = hotpot_system
#         one_shot_input = hotpot_input
#         one_shot_output = hotpot_output

#     # -------------------------------------------------------
#     # 최종 유저 프롬프트 (retrieved context + query)
#     # -------------------------------------------------------
#     final_user_prompt = (
#         f"{context}"
#         f"\n\nQuestion: {query}"
#         "\nThought: "
#     )

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": one_shot_input},
#         {"role": "assistant", "content": one_shot_output},
#         {"role": "user", "content": final_user_prompt},
#     ]

#     try:
#         prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])

#         final_response = await global_config["llm_model_func"](
#             prompt=prompt,
#             stream=getattr(param, "stream", False),
#         )
#         def extract_answer(response: str) -> str:
#             if "Answer:" in response:
#                 return response.split("Answer:", 1)[1].strip()
#             return response.strip()

#         final_response = extract_answer(final_response)
#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return "Final generation failed."


#     # =======================================================
#     # Step 6. SAVE FULL JSON LOG
#     # =======================================================
#     try:
#         # cls는 위에서 이미 계산됨
#         save_dir = f"/workspace/AtomRAG/{cls}"
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f"{cls}_full_results.json")

#         full_output = {
#             "query": query,
#             "dataset_cls": cls,
#             "subqueries": queries,
#             "final_context": context,
#             "messages_prompt_used": messages,
#             "final_answer": (
#                 final_response if isinstance(final_response, str)
#                 else str(final_response)
#             ),
#         }

#         if os.path.exists(save_path):
#             with open(save_path, "r", encoding="utf-8") as f:
#                 try:
#                     existing = json.load(f)
#                     if not isinstance(existing, list):
#                         existing = [existing]
#                 except Exception:
#                     existing = []
#         else:
#             existing = []

#         existing.append(full_output)

#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(existing, f, ensure_ascii=False, indent=4)

#         print(f"[INFO] JSON saved to {save_path}")

#     except Exception as e:
#         print(f"[ERROR] Saving JSON failed: {e}")

#     print(final_response)
    
#     return final_response

def compute_cost(token_log):
    prompt_cost = token_log["prompt_tokens"] / 1_000_000 * 0.15
    completion_cost = token_log["completion_tokens"] / 1_000_000 * 0.60
    return prompt_cost + completion_cost


import time
from collections import defaultdict

GLOBAL_TIME_LOG = []
GLOBAL_TOKEN_LOG = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}


####### 일반 chunk를 넣어주고 cot방식의 generation을 사용한 방법#########
async def ours_kg_query_experiment1(
    query: str,
    atomic_knowledge_graph_inst,
    atomic_entities_vdb,
    chunks_vdb,
    text_triple_db,
    text_atomics_db,
    text_chunks_db,
    param,
    global_config: dict,
    hashing_kv=None,
) -> str:
    """
    Simplified version:
    - Run Step 0~3 (retrieval)
    - Use top-K raw chunks as context
    - Generate final answer using structured-CoT RAG format
    - Save entire reasoning chain to JSON file
    """
    t_query_start = time.perf_counter()

    # =======================================================
    # Step 0. Query decomposition
    # =======================================================
    query_modes = {
        "experiment1": query_process_1,
        "experiment2": query_process_2,
        "experiment3": query_process_3,
        "experiment4": query_process_4,
    }
    queries = await query_modes.get(param.query_mode, lambda *a: [query])(
        query, param, global_config, hashing_kv
    )
    if not queries:
        queries = [query]

    print(f"[INFO] Step 0 complete — {len(queries)} subqueries generated.")

    # =======================================================
    # Step 1. Atomic fact retrieval
    # =======================================================
    t_step1_start = time.perf_counter()

    print(f"[INFO] Retrieving atomic facts for {len(queries)} subqueries...")

    async def fetch_atomic_facts(sq):
        try:
            results = await atomic_entities_vdb.query(sq, top_k=20)
            return sq, results
        except Exception as e:
            print(f"[ERROR] Retrieval failed for '{sq}': {e}")
            return sq, []

    subquery_results = dict(
        await asyncio.gather(*(fetch_atomic_facts(sq) for sq in queries))
    )

    step1_retrieval_log = {
        sq: [
            {
                "atomic_id": r.get("id"),
                "entity_name": r.get("entity_name", ""),
                "score": r.get("distance", 0.0),
            }
            for r in results
        ]
        for sq, results in subquery_results.items()
    }
    
    t_step1_end = time.perf_counter()
    step1_time = t_step1_end - t_step1_start


    # =======================================================
    # Step 2. Atomic → Chunk mapping
    # =======================================================
    t_step2_start = time.perf_counter()

    chunk_to_atomics = defaultdict(lambda: {"atomic_facts": []})
    chunk_cache = {}

    retrieved_items = []
    for sq, results in subquery_results.items():
        for r in results:
            rid = r.get("id")
            if rid:
                retrieved_items.append((sq, rid, r.get("distance", 0.0)))

    if not retrieved_items:
        return "No results."

    unique_atomic_ids = list({rid for _, rid, _ in retrieved_items})
    atomic_records = await text_atomics_db.get_by_ids(unique_atomic_ids)

    atomic_info_map = {}
    for aid, rec in zip(unique_atomic_ids, atomic_records or []):
        if rec and isinstance(rec, dict):
            atomic_info_map[aid] = {
                "content": rec.get("content", ""),
                "source_ids": rec.get("source_ids", []) or [],
            }

    for sq, aid, score in retrieved_items:
        ainfo = atomic_info_map.get(aid)
        if not ainfo:
            continue

        atomic_text = ainfo["content"]
        src_chunks = ainfo["source_ids"]

        for cid in src_chunks:
            if cid not in chunk_cache:
                try:
                    cdata = await text_chunks_db.get_by_ids([cid])
                    chunk_cache[cid] = cdata[0].get("content", "") if cdata else ""
                except Exception:
                    chunk_cache[cid] = ""

            chunk_to_atomics[cid]["chunk_id"] = cid
            chunk_to_atomics[cid]["chunk_text"] = chunk_cache[cid]
            chunk_to_atomics[cid]["atomic_facts"].append({
                "atomic_id": aid,
                "atomic_text": atomic_text,
                "sub_query": sq,
                "score": score,
            })

    for item in chunk_to_atomics.values():
        seen = set()
        uniq = []
        for af in item["atomic_facts"]:
            aid = af["atomic_id"]
            if aid not in seen:
                seen.add(aid)
                uniq.append(af)
        item["atomic_facts"] = uniq

    chunk_results = list(chunk_to_atomics.values())
    print(f"[INFO] Step 2 complete — {len(chunk_results)} chunks mapped.")


    step2_mapping_log = []
    for item in chunk_results:
        txt = item.get("chunk_text", "")
        step2_mapping_log.append({
            "chunk_id": item["chunk_id"],
            "chunk_text_preview": txt[:150] + ("..." if len(txt) > 150 else ""),
            "num_atomics": len(item["atomic_facts"]),
            "atomic_facts": item["atomic_facts"],
        })
        
    t_step2_end = time.perf_counter()
    step2_time = t_step2_end - t_step2_start
    # =======================================================
    # Step 3. Rank chunks using only max_atomic_score
    # =======================================================

    for item in chunk_results:
        item["max_atomic_score"] = max([af["score"] for af in item["atomic_facts"]], default=0.0)

    # sort by max_atomic_score only
    top = param.top_mode
    top_k = min(top, len(chunk_results))
    chunk_results_sorted = sorted(
        chunk_results,
        key=lambda x: x["max_atomic_score"],
        reverse=True
    )[:top_k]

    print(f"[INFO] Step 3 complete — Selected top-{top_k} chunks.")

    step3_topk_log = [
        {
            "chunk_id": item["chunk_id"],
            "max_atomic_score": item["max_atomic_score"],
            "chunk_text": item.get("chunk_text", ""),
            "atomic_ids": [af["atomic_id"] for af in item["atomic_facts"]],
        }
        for item in chunk_results_sorted
    ]
    
    # =======================================================
    # Step 4. Build simple context (NO relevance labels)
    # =======================================================

    context_blocks = []
    for item in chunk_results_sorted:
        chunk_text = item.get("chunk_text", "").strip()
        context_blocks.append(chunk_text)

    context = "\n\n".join(context_blocks)
    print("[INFO] Step 4 complete — Context built (no labels).")

    # =======================================================
    # Step 5. FINAL ANSWER GENERATION (HOTPOT vs MULTIHOP BY cls)
    # =======================================================

    # working_dir 기준으로 어떤 데이터셋인지 판단
    cls = os.path.basename(global_config.get("working_dir", ""))

    # ---------------------------
    # HOTPOT one-shot prompt
    # ---------------------------
    hotpot_system = (
        'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
        'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
        'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
    )

    hotpot_docs = (
        """The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
        """Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
        """Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
        """Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
        """Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"""
    )

    hotpot_input = (
        f"{hotpot_docs}"
        "\n\nQuestion: "
        "When was Neville A. Stanton's employer founded?"
        "\nThought: "
    )

    hotpot_output = (
        "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
        "\nAnswer: 1862."
    )

    # ---------------------------
    # MULTIHOP one-shot prompt
    # ---------------------------
    multihop_system = (
        'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
        'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
        'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
        'Use only the given Information; if it is insufficient, reply with "Answer: Insufficient information.".'
        'If you need to answer like yes or no, use "Answer: Yes" or "Answer: No" only.'
    )

    multihop_docs = (
        """Milk and Honey (album)\nMilk and Honey is an album by John Lennon and Yoko Ono released in 1984. Following the compilation "The John Lennon Collection", it is Lennon's eighth and final studio album, and the first posthumous release of new Lennon music, having been recorded in the last months of his life during and following the sessions for their 1980 album "Double Fantasy". It was assembled by Yoko Ono in association with the Geffen label.\n\n"""
        """John Lennon Museum\nJohn Lennon Museum (ジョン・レノン・ミュージアム , Jon Renon Myūjiamu ) was a museum located inside the Saitama Super Arena in Chūō-ku, Saitama, Saitama Prefecture, Japan. It was established to preserve knowledge of John Lennon's life and musical career. It displayed Lennon's widow Yoko Ono's collection of his memorabilia as well as other displays. The museum opened on October 9, 2000, the 60th anniversary of Lennon's birth, and closed on September 30, 2010, when its exhibit contract with Yoko Ono expired. A tour of the museum began with a welcoming message and short film narrated by Yoko Ono (in Japanese with English headphones available), and ended at an avant-garde styled "reflection room" full of chairs facing a slide show of moving words and images. After this room there was a gift shop with John Lennon memorabilia available.\n\n"""
        """Walls and Bridges\nWalls and Bridges is the fifth studio album by English musician John Lennon. It was issued by Apple Records on 26 September 1974 in the United States and on 4 October in the United Kingdom. Written, recorded and released during his 18-month separation from Yoko Ono, the album captured Lennon in the midst of his "Lost Weekend". "Walls and Bridges" was an American "Billboard" number-one album and featured two hit singles, "Whatever Gets You thru the Night" and "#9 Dream". The first of these was Lennon's first number-one hit in the United States as a solo artist, and his only chart-topping single in either the US or Britain during his lifetime.\n\n"""
        """Nobody Loves You (When You're Down and Out)\n"Nobody Loves You (When You're Down and Out)" is a song written by John Lennon released on his 1974 album "Walls and Bridges". The song is included on the 1986 compilation "Menlove Ave.", the 1990 boxset "Lennon", the 1998 boxset "John Lennon Anthology", the 2005 two-disc compilation "", and the 2010 boxset "Gimme Some Truth".\n\n"""
        """Give Peace a Chance\n"Give Peace a Chance" is an anti-war song written by John Lennon (credited to Lennon–McCartney), and performed with Yoko Ono in Montreal, Quebec, Canada. Released as a single in 1969 by the Plastic Ono Band on Apple Records (catalogue Apple 13 in the United Kingdom, Apple 1809 in the United States), it is the first solo single issued by Lennon, released when he was still a member of the Beatles, and became an anthem of the American anti-war movement during the 1970s. It peaked at number 14 on the "Billboard" Hot 100 and number 2 on the British singles chart.\n"""
    )

    multihop_input = (
        f"{multihop_docs}"
        "\n\nQuestion: "
        "Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?"
        "\nThought: "
    )

    multihop_output = (
        "The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges. Nobody Loves You was written by John Lennon on Walls and Bridges album. "
        "\nAnswer: Walls and Bridges."
    )
    
    sillok_system = (
        '상급 독해 보조 도우미로서, 당신의 임무는 주어진 지문과 해당 질문을 세밀하게 분석하는 것입니다.'
        '당신의 응답은 반드시 "생각: " 이후부터 시작해야 하며, 그 안에서 결론에 도달하는 과정을 체계적으로 단계별로 설명해야 합니다.'
        '마지막에는 "답변: "로 마무리하며, 추가적인 설명 없이 간결하고 명확한 최종 답변만 제시해야 합니다.'
    )

    sillok_docs = (
        # 1. noise 문서 (무공 찬양)
        """敗走, 乘勝奄至 高州 之境, 卿卷甲兼行, 逐出疆外; 歲癸卯, 庶孽 德興君 擧兵入西鄙, 卿率輕騎, 挫其鋒銳; 歲丁巳, 倭奴 寇 海州, 諸相奔潰, 卿獨身先士卒, 擊之幾盡; 歲庚申, 倭奴 自 鎭浦 下岸, 橫行 楊廣、慶尙、全羅 之境, 焚蕩郡邑, 殺掠士女, 三道騷然, 元帥 裵彦、朴修敬 等敗死。 卿出萬死不顧之計, 率其麾下, 鏖戰 引月 之驛, 捕獲無遺, 民賴以安。 其行師也, 動遵紀律, 秋毫不犯, 民畏其威, 民懷其德, 雖古名將, 何以加焉? 卿之豐功偉烈, 在人耳目者, 赫赫如此, 而不自矜伐, 歉然退托, 國人益以倚重。""",

        # 2. 필요 문서 (8월 임명)
        """太祖實錄一卷, 總序九十三條。 ○八月, 昌 以 太祖 都摠中外諸軍事。""",

        # 3. 필요 문서 (10월 임명)
        """太祖實錄一卷, 總序九十四條。 ○十月, 以 太祖 兼判尙瑞司事。""",

        # 4. 필요 문서 (이색 칭찬)
        """太祖實錄一卷, 總序九十五條。 ○自 恭愍王 薨, 天子每徵執政大臣, 皆懼不敢行。 門下侍中 李穡 欲 昌 親朝, 又欲王官監國, 自請入朝。 昌 遣 穡 及僉書密直 李崇仁, 如京師賀正, 且請王官監國。 太祖 稱 穡 曰: "慷慨哉, 是翁!" 穡 以 太祖 威德日盛, 中外歸心, 恐其未還乃有變, 請一子從行, 太祖 以殿下爲書狀官。""",

        # 5. noise 문서 (전제 개혁)
        """太祖實錄一卷, 總序九十六條。 ○ 恭讓王 元年己巳。 是時, 田制大毁, 兼幷之家, 攘奪土田, 籠山絡野, 毒痡日深, 民胥怨咨。 太祖 與大司憲 趙浚, 議革私田, 以杜兼幷, 以厚民業, 於是中外大悅, 民心益附。"""
    )

    sillok_input = (
        f"{sillok_docs}"
        "\n\n질문: "
        "태조가 도총중외제군사와 겸판상서사사로 임명된 달은 각각 언제이며, 태조가 이색을 칭찬하며 한 말은 무엇인가?"
        "\n생각: "
    )

    sillok_output = (
        "도총중외제군사 임명 시점은 '○八月, 昌 以 太祖 都摠中外諸軍事'에서 8월임을 확인한다. "
        "겸판상서사사 임명 시점은 '○十月, 以 太祖 兼判尙瑞司事'에서 10월임을 확인한다. "
        "이색에 대한 칭찬은 '太祖 稱 穡 曰: \"慷慨哉, 是翁!\"'에서 확인된다."
        "\n답변: 도총중외제군사는 8월, 겸판상서사사는 10월, 칭찬의 말은 “慷慨哉, 是翁!”이다."
    )

    # -------------------------------------------------------
    # cls 이름을 기준으로 사용할 프롬프트 선택
    # -------------------------------------------------------
    cls_lower = cls.lower()
    if "multihop" in cls_lower:
        system_prompt = multihop_system
        one_shot_input = multihop_input
        one_shot_output = multihop_output
    elif "sillok" in cls_lower:
        system_prompt = sillok_system
        one_shot_input = sillok_input
        one_shot_output = sillok_output
    else:
        # 기본값 hotpot (cls에 hotpot이 포함되어 있거나, 그 외 케이스)
        system_prompt = hotpot_system
        one_shot_input = hotpot_input
        one_shot_output = hotpot_output

    # -------------------------------------------------------
    # 최종 유저 프롬프트 (retrieved context + query)
    # -------------------------------------------------------
    final_user_prompt = (
        f"{context}"
        f"\n\n질문: {query}"
        "\n생각: "
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": one_shot_input},
        {"role": "assistant", "content": one_shot_output},
        {"role": "user", "content": final_user_prompt},
    ]
    t_gen_start = time.perf_counter()

    try:
        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        
        final_response = await global_config["llm_model_func"](
            prompt=prompt,
            stream=getattr(param, "stream", False),
        )
        def extract_answer(response: str) -> str:
            if "답변:" in response:
                return response.split("답변:", 1)[1].strip()
            return response.strip()

        final_response = extract_answer(final_response)
    except Exception as e:
        print(f"[ERROR] Final answer generation failed: {e}")
        return "Final generation failed."
    t_gen_end = time.perf_counter()
    gen_time = t_gen_end - t_gen_start

    # # =======================================================
    # # Step 6. SAVE FULL JSON LOG
    # # =======================================================
    # try:
    #     # cls는 위에서 이미 계산됨
    #     save_dir = f"/workspace/AtomRAG/AtomRAG/retrieve"
    #     os.makedirs(save_dir, exist_ok=True)
    #     save_path = os.path.join(save_dir, f"atomic_20_top{top}_ours_experiment1_{cls}_full_results.json")

    #     full_output = {
    #         "query": query,
    #         "dataset_cls": cls,
    #         "subqueries": queries,
    #         "step1_retrieval": step1_retrieval_log,
    #         "step2_mapping": None,
    #         "step3_topk_chunks": step3_topk_log,
    #         "final_context": context,
    #         "messages_prompt_used": messages,
    #         "final_answer": (
    #             final_response if isinstance(final_response, str)
    #             else str(final_response)
    #         ),
    #     }

    #     if os.path.exists(save_path):
    #         with open(save_path, "r", encoding="utf-8") as f:
    #             try:
    #                 existing = json.load(f)
    #                 if not isinstance(existing, list):
    #                     existing = [existing]
    #             except Exception:
    #                 existing = []
    #     else:
    #         existing = []

    #     existing.append(full_output)

    #     with open(save_path, "w", encoding="utf-8") as f:
    #         json.dump(existing, f, ensure_ascii=False, indent=4)

    #     print(f"[INFO] JSON saved to {save_path}")

    # except Exception as e:
    #     print(f"[ERROR] Saving JSON failed: {e}")

    print(final_response)
    
    t_query_end = time.perf_counter()

    query_time = t_query_end - t_query_start

    GLOBAL_TIME_LOG.append({
        "query": query,
        "total_time_sec": query_time,
    })

    print(f"[TIME] Query total time: {query_time:.4f} sec")
    print(f"[TIME] Query total time: {step1_time:.4f} sec")
    print(f"[TIME] Query total time: {step2_time:.4f} sec")
    print(f"[TIME] Query total time: {gen_time:.4f} sec")

    GLOBAL_TIME_LOG[-1].update({
        "step1_retrieval_sec": step1_time,
        "step2_mapping_sec": step2_time,
        "generation_sec": gen_time,
    })
    
    total_cost = compute_cost(GLOBAL_TOKEN_LOG)
    cost_per_1k_queries = total_cost / len(GLOBAL_TIME_LOG) * 1000

    print(f"[COST] Total cost: ${total_cost:.4f}")
    print(f"[COST] Cost per 1k queries: ${cost_per_1k_queries:.4f}")


    return final_response, step3_topk_log


####### 일반 chunk를 넣어주고 cot방식의 generation을 사용한 방법######### 이거는 ablation
# async def ours_kg_query_experiment1(
#     query: str,
#     atomic_knowledge_graph_inst,
#     atomic_entities_vdb,
#     chunks_vdb,
#     text_triple_db,
#     text_atomics_db,
#     text_chunks_db,
#     param,
#     global_config: dict,
#     hashing_kv=None,
# ) -> str:
#     """
#     Simplified version:
#     - Run Step 0~3 (retrieval)
#     - Use top-K raw chunks as context
#     - Generate final answer using structured-CoT RAG format
#     - Save entire reasoning chain to JSON file
#     """
#     # =======================================================
#     # Step 0. Query decomposition
#     # =======================================================
#     query_modes = {
#         "experiment1": query_process_1,
#         "experiment2": query_process_2,
#         "experiment3": query_process_3,
#         "experiment4": query_process_4,
#     }
#     queries = await query_modes.get(param.query_mode, lambda *a: [query])(
#         query, param, global_config, hashing_kv
#     )
#     if not queries:
#         queries = [query]

#     print(f"[INFO] Step 0 complete — {len(queries)} subqueries generated.")
    
#     # # =======================================================
#     # # Step 0. NO query decomposition (ABLATION)
#     # # =======================================================
#     # queries = [query]
#     # print("[INFO] Step 0 skipped — using original query only.")



#     # =======================================================
#     # Step 1. Atomic fact retrieval
#     # =======================================================
#     print(f"[INFO] Retrieving atomic facts for {len(queries)} subqueries...")

#     async def fetch_atomic_facts(sq):
#         try:
#             results = await atomic_entities_vdb.query(sq, top_k=10)
#             return sq, results
#         except Exception as e:
#             print(f"[ERROR] Retrieval failed for '{sq}': {e}")
#             return sq, []

#     subquery_results = dict(
#         await asyncio.gather(*(fetch_atomic_facts(sq) for sq in queries))
#     )

#     step1_retrieval_log = {
#         sq: [
#             {
#                 "atomic_id": r.get("id"),
#                 "entity_name": r.get("entity_name", ""),
#                 "score": r.get("distance", 0.0),
#             }
#             for r in results
#         ]
#         for sq, results in subquery_results.items()
#     }
    
#     # # =======================================================
#     # # Step 1. Subquery → Chunk retrieval (ABLATION)
#     # # =======================================================
#     # print(f"[INFO] Retrieving chunks for {len(queries)} subqueries...")

#     # async def fetch_chunks(sq):
#     #     try:
#     #         results = await chunks_vdb.query(sq, top_k=10)
#     #         return sq, results
#     #     except Exception as e:
#     #         print(f"[ERROR] Chunk retrieval failed for '{sq}': {e}")
#     #         return sq, []

#     # subquery_results = dict(
#     #     await asyncio.gather(*(fetch_chunks(sq) for sq in queries))
#     # )

#     # step1_retrieval_log = {
#     #     sq: [
#     #         {
#     #             "chunk_id": r.get("id"),
#     #             "score": r.get("distance", 0.0),
#     #         }
#     #         for r in results
#     #     ]
#     #     for sq, results in subquery_results.items()
#     # }



#     # =======================================================
#     # Step 2. Atomic → Chunk mapping
#     # =======================================================
#     chunk_to_atomics = defaultdict(lambda: {"atomic_facts": []})
#     chunk_cache = {}

#     retrieved_items = []
#     for sq, results in subquery_results.items():
#         for r in results:
#             rid = r.get("id")
#             if rid:
#                 retrieved_items.append((sq, rid, r.get("distance", 0.0)))

#     if not retrieved_items:
#         return "No results."

#     unique_atomic_ids = list({rid for _, rid, _ in retrieved_items})
#     atomic_records = await text_atomics_db.get_by_ids(unique_atomic_ids)

#     atomic_info_map = {}
#     for aid, rec in zip(unique_atomic_ids, atomic_records or []):
#         if rec and isinstance(rec, dict):
#             atomic_info_map[aid] = {
#                 "content": rec.get("content", ""),
#                 "source_ids": rec.get("source_ids", []) or [],
#             }

#     for sq, aid, score in retrieved_items:
#         ainfo = atomic_info_map.get(aid)
#         if not ainfo:
#             continue

#         atomic_text = ainfo["content"]
#         src_chunks = ainfo["source_ids"]

#         for cid in src_chunks:
#             if cid not in chunk_cache:
#                 try:
#                     cdata = await text_chunks_db.get_by_ids([cid])
#                     chunk_cache[cid] = cdata[0].get("content", "") if cdata else ""
#                 except Exception:
#                     chunk_cache[cid] = ""

#             chunk_to_atomics[cid]["chunk_id"] = cid
#             chunk_to_atomics[cid]["chunk_text"] = chunk_cache[cid]
#             chunk_to_atomics[cid]["atomic_facts"].append({
#                 "atomic_id": aid,
#                 "atomic_text": atomic_text,
#                 "sub_query": sq,
#                 "score": score,
#             })

#     for item in chunk_to_atomics.values():
#         seen = set()
#         uniq = []
#         for af in item["atomic_facts"]:
#             aid = af["atomic_id"]
#             if aid not in seen:
#                 seen.add(aid)
#                 uniq.append(af)
#         item["atomic_facts"] = uniq

#     chunk_results = list(chunk_to_atomics.values())
#     print(f"[INFO] Step 2 complete — {len(chunk_results)} chunks mapped.")


#     step2_mapping_log = []
#     for item in chunk_results:
#         txt = item.get("chunk_text", "")
#         step2_mapping_log.append({
#             "chunk_id": item["chunk_id"],
#             "chunk_text_preview": txt[:150] + ("..." if len(txt) > 150 else ""),
#             "num_atomics": len(item["atomic_facts"]),
#             "atomic_facts": item["atomic_facts"],
#         })

#     # # =======================================================
#     # # Step 2. Merge chunk scores across subqueries
#     # # =======================================================
#     # from collections import defaultdict

#     # chunk_score_map = defaultdict(list)

#     # for sq, results in subquery_results.items():
#     #     for r in results:
#     #         cid = r.get("id")
#     #         if cid:
#     #             chunk_score_map[cid].append(r.get("distance", 0.0))

#     # if not chunk_score_map:
#     #     return "No results."

    

#     # =======================================================
#     # Step 3. Rank chunks using only max_atomic_score
#     # =======================================================

#     for item in chunk_results:
#         item["max_atomic_score"] = max([af["score"] for af in item["atomic_facts"]], default=0.0)

#     # sort by max_atomic_score only
#     top = param.top_mode
#     top_k = min(top, len(chunk_results))
#     chunk_results_sorted = sorted(
#         chunk_results,
#         key=lambda x: x["max_atomic_score"],
#         reverse=True
#     )[:top_k]

#     print(f"[INFO] Step 3 complete — Selected top-{top_k} chunks.")

#     step3_topk_log = [
#         {
#             "chunk_id": item["chunk_id"],
#             "max_atomic_score": item["max_atomic_score"],
#             "atomic_ids": [af["atomic_id"] for af in item["atomic_facts"]],
#         }
#         for item in chunk_results_sorted
#     ]
    
    
#     # # =======================================================
#     # # Step 3. Rank chunks by MAX score (ABLATION)
#     # # =======================================================
#     # chunk_items = []

#     # for cid, scores in chunk_score_map.items():
#     #     chunk_items.append({
#     #         "chunk_id": cid,
#     #         "score": max(scores),  # 🔥 max score
#     #     })

#     # chunk_items = sorted(
#     #     chunk_items,
#     #     key=lambda x: x["score"],
#     #     reverse=True
#     # )

#     # top = 7
#     # top_k = min(top, len(chunk_items))
#     # top_chunks = chunk_items[:top_k]

#     # print(f"[INFO] Step 3 complete — Selected top-{top_k} chunks.")

#     # step3_topk_log = top_chunks
    
#     # # =======================================================
#     # # Step 3. RANDOM chunk selection (ABLATION)
#     # # =======================================================
#     # import random

#     # random_pool = chunk_results[:]  # atomic → chunk mapping 결과 그대로 사용

#     # if not random_pool:
#     #     return "No results."
    
#     # top = 7
#     # top_k = min(top, len(random_pool))
#     # chunk_results_sorted = random.sample(random_pool, top_k)

#     # print(f"[INFO] Step 3 complete — Randomly selected {top_k} chunks.")

#     # step3_topk_log = [
#     #     {
#     #         "chunk_id": item["chunk_id"],
#     #         "selection": "random",
#     #     }
#     #     for item in chunk_results_sorted
#     # ]




#     # =======================================================
#     # Step 4. Build simple context (NO relevance labels)
#     # =======================================================

#     context_blocks = []
#     for item in chunk_results_sorted:
#         chunk_text = item.get("chunk_text", "").strip()
#         context_blocks.append(chunk_text)

#     context = "\n\n".join(context_blocks)
#     print("[INFO] Step 4 complete — Context built (no labels).")
    
#     # # =======================================================
#     # # Step 4. Build context from top chunks
#     # # =======================================================
#     # chunk_ids = [c["chunk_id"] for c in top_chunks]
#     # chunk_records = await text_chunks_db.get_by_ids(chunk_ids)

#     # context_blocks = []
#     # for rec in chunk_records:
#     #     if rec and "content" in rec:
#     #         context_blocks.append(rec["content"].strip())

#     # context = "\n\n".join(context_blocks)
#     # print("[INFO] Step 4 complete — Context built.")


#     # =======================================================
#     # Step 5. FINAL ANSWER GENERATION (HOTPOT vs MULTIHOP BY cls)
#     # =======================================================

#     # working_dir 기준으로 어떤 데이터셋인지 판단
#     cls = os.path.basename(global_config.get("working_dir", ""))

#     # ---------------------------
#     # HOTPOT one-shot prompt
#     # ---------------------------
#     hotpot_system = (
#         'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
#         'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
#         'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
#     )

#     hotpot_docs = (
#         """The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
#         """Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
#         """Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
#         """Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
#         """Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"""
#     )

#     hotpot_input = (
#         f"{hotpot_docs}"
#         "\n\nQuestion: "
#         "When was Neville A. Stanton's employer founded?"
#         "\nThought: "
#     )

#     hotpot_output = (
#         "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
#         "\nAnswer: 1862."
#     )

#     # ---------------------------
#     # MULTIHOP one-shot prompt
#     # ---------------------------
#     multihop_system = (
#         'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
#         'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
#         'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
#         'Use only the given Information; if it is insufficient, reply with "Answer: Insufficient information.".'
#         'If you need to answer like yes or no, use "Answer: Yes" or "Answer: No" only.'
#     )

#     multihop_docs = (
#         """Milk and Honey (album)\nMilk and Honey is an album by John Lennon and Yoko Ono released in 1984. Following the compilation "The John Lennon Collection", it is Lennon's eighth and final studio album, and the first posthumous release of new Lennon music, having been recorded in the last months of his life during and following the sessions for their 1980 album "Double Fantasy". It was assembled by Yoko Ono in association with the Geffen label.\n\n"""
#         """John Lennon Museum\nJohn Lennon Museum (ジョン・レノン・ミュージアム , Jon Renon Myūjiamu ) was a museum located inside the Saitama Super Arena in Chūō-ku, Saitama, Saitama Prefecture, Japan. It was established to preserve knowledge of John Lennon's life and musical career. It displayed Lennon's widow Yoko Ono's collection of his memorabilia as well as other displays. The museum opened on October 9, 2000, the 60th anniversary of Lennon's birth, and closed on September 30, 2010, when its exhibit contract with Yoko Ono expired. A tour of the museum began with a welcoming message and short film narrated by Yoko Ono (in Japanese with English headphones available), and ended at an avant-garde styled "reflection room" full of chairs facing a slide show of moving words and images. After this room there was a gift shop with John Lennon memorabilia available.\n\n"""
#         """Walls and Bridges\nWalls and Bridges is the fifth studio album by English musician John Lennon. It was issued by Apple Records on 26 September 1974 in the United States and on 4 October in the United Kingdom. Written, recorded and released during his 18-month separation from Yoko Ono, the album captured Lennon in the midst of his "Lost Weekend". "Walls and Bridges" was an American "Billboard" number-one album and featured two hit singles, "Whatever Gets You thru the Night" and "#9 Dream". The first of these was Lennon's first number-one hit in the United States as a solo artist, and his only chart-topping single in either the US or Britain during his lifetime.\n\n"""
#         """Nobody Loves You (When You're Down and Out)\n"Nobody Loves You (When You're Down and Out)" is a song written by John Lennon released on his 1974 album "Walls and Bridges". The song is included on the 1986 compilation "Menlove Ave.", the 1990 boxset "Lennon", the 1998 boxset "John Lennon Anthology", the 2005 two-disc compilation "", and the 2010 boxset "Gimme Some Truth".\n\n"""
#         """Give Peace a Chance\n"Give Peace a Chance" is an anti-war song written by John Lennon (credited to Lennon–McCartney), and performed with Yoko Ono in Montreal, Quebec, Canada. Released as a single in 1969 by the Plastic Ono Band on Apple Records (catalogue Apple 13 in the United Kingdom, Apple 1809 in the United States), it is the first solo single issued by Lennon, released when he was still a member of the Beatles, and became an anthem of the American anti-war movement during the 1970s. It peaked at number 14 on the "Billboard" Hot 100 and number 2 on the British singles chart.\n"""
#     )

#     multihop_input = (
#         f"{multihop_docs}"
#         "\n\nQuestion: "
#         "Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?"
#         "\nThought: "
#     )

#     multihop_output = (
#         "The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges. Nobody Loves You was written by John Lennon on Walls and Bridges album. "
#         "\nAnswer: Walls and Bridges."
#     )

#     # -------------------------------------------------------
#     # cls 이름을 기준으로 사용할 프롬프트 선택
#     # -------------------------------------------------------
#     cls_lower = cls.lower()
#     if "multihop" in cls_lower:
#         system_prompt = multihop_system
#         one_shot_input = multihop_input
#         one_shot_output = multihop_output
#     else:
#         # 기본값 hotpot (cls에 hotpot이 포함되어 있거나, 그 외 케이스)
#         system_prompt = hotpot_system
#         one_shot_input = hotpot_input
#         one_shot_output = hotpot_output

#     # -------------------------------------------------------
#     # 최종 유저 프롬프트 (retrieved context + query)
#     # -------------------------------------------------------
#     final_user_prompt = (
#         f"{context}"
#         f"\n\nQuestion: {query}"
#         "\nThought: "
#     )

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": one_shot_input},
#         {"role": "assistant", "content": one_shot_output},
#         {"role": "user", "content": final_user_prompt},
#     ]

#     try:
#         prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])

#         final_response = await global_config["llm_model_func"](
#             prompt=prompt,
#             stream=getattr(param, "stream", False),
#         )
#         def extract_answer(response: str) -> str:
#             if "Answer:" in response:
#                 return response.split("Answer:", 1)[1].strip()
#             return response.strip()

#         final_response = extract_answer(final_response)
#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return "Final generation failed."


#     # =======================================================
#     # Step 6. SAVE FULL JSON LOG
#     # =======================================================
#     try:
#         # cls는 위에서 이미 계산됨
#         save_dir = f"/workspace/AtomRAG/AtomRAG/retrieve"
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f"ablation_rerank_top{top}_ours_experiment1_{cls}_full_results.json")

#         full_output = {
#             "query": query,
#             "dataset_cls": cls,
#             "subqueries": queries,
#             "step1_retrieval": step1_retrieval_log,
#             "step2_mapping": None,
#             "step3_topk_chunks": step3_topk_log,
#             "final_context": context,
#             "messages_prompt_used": messages,
#             "final_answer": (
#                 final_response if isinstance(final_response, str)
#                 else str(final_response)
#             ),
#         }

#         if os.path.exists(save_path):
#             with open(save_path, "r", encoding="utf-8") as f:
#                 try:
#                     existing = json.load(f)
#                     if not isinstance(existing, list):
#                         existing = [existing]
#                 except Exception:
#                     existing = []
#         else:
#             existing = []

#         existing.append(full_output)

#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(existing, f, ensure_ascii=False, indent=4)

#         print(f"[INFO] JSON saved to {save_path}")

#     except Exception as e:
#         print(f"[ERROR] Saving JSON failed: {e}")

#     print(final_response)
    
#     return final_response


# ####### atomic fact를 여러 chunk와 연결하여 진행 + 일반 chunk를 넣어주고 cot방식의 generation을 사용한 방법#########
# async def ours_kg_query_experiment1(
#     query: str,
#     atomic_knowledge_graph_inst,
#     atomic_entities_vdb,
#     chunks_vdb,                      # ✅ 사용
#     text_triple_db,
#     text_atomics_db,
#     text_chunks_db,
#     param,
#     global_config: dict,
#     hashing_kv=None,
# ) -> str:

#     # =======================================================
#     # Step 0. Query decomposition
#     # =======================================================
#     query_modes = {
#         "experiment1": query_process_1,
#         "experiment2": query_process_2,
#         "experiment3": query_process_3,
#         "experiment4": query_process_4,
#     }
#     queries = await query_modes.get(param.query_mode, lambda *a: [query])(
#         query, param, global_config, hashing_kv
#     )
#     if not queries:
#         queries = [query]

#     print(f"[INFO] Step 0 complete — {len(queries)} subqueries generated.")

#     # =======================================================
#     # Step 1. Subquery → Atomic retrieval
#     # =======================================================
#     async def fetch_atomic_facts(sq):
#         try:
#             results = await atomic_entities_vdb.query(sq, top_k=10)
#             return sq, results
#         except Exception as e:
#             print(f"[ERROR] Atomic retrieval failed for '{sq}': {e}")
#             return sq, []

#     subquery_results = dict(
#         await asyncio.gather(*(fetch_atomic_facts(sq) for sq in queries))
#     )

#     # =======================================================
#     # Step 2. Load atomic records
#     # =======================================================
#     retrieved_items = []
#     for sq, results in subquery_results.items():
#         for r in results:
#             aid = r.get("id")
#             if aid:
#                 retrieved_items.append((sq, aid, float(r.get("distance", 0.0))))

#     if not retrieved_items:
#         return "No results."

#     unique_atomic_ids = list({aid for _, aid, _ in retrieved_items})
#     atomic_records = await text_atomics_db.get_by_ids(unique_atomic_ids)

#     atomic_info_map = {}
#     for aid, rec in zip(unique_atomic_ids, atomic_records or []):
#         if rec and isinstance(rec, dict):
#             atomic_info_map[aid] = {
#                 "content": rec.get("content", ""),
#                 "source_ids": rec.get("source_ids", []) or [],
#             }

#     # =======================================================
#     # Step 3. Chunk scoring (NEW LOGIC with chunks_vdb)
#     # =======================================================
#     chunk_map = defaultdict(lambda: {
#         "chunk_id": None,
#         "chunk_text": "",
#         "chunk_score": 0.0,
#         "evidences": [],
#     })

#     chunk_text_cache = {}

#     async def get_chunk_text(cid: str) -> str:
#         if cid in chunk_text_cache:
#             return chunk_text_cache[cid]
#         try:
#             cdata = await text_chunks_db.get_by_ids([cid])
#             txt = cdata[0].get("content", "") if cdata else ""
#         except Exception:
#             txt = ""
#         chunk_text_cache[cid] = txt
#         return txt

#     for sq, aid, atomic_score in retrieved_items:
#         ainfo = atomic_info_map.get(aid)
#         if not ainfo:
#             continue

#         src_chunks = ainfo["source_ids"]
#         atomic_text = ainfo["content"]

#         if not src_chunks:
#             continue

#         # ---------------------------
#         # Case 1: single source chunk
#         # ---------------------------
#         if len(src_chunks) == 1:
#             cid = src_chunks[0]
#             ctext = await get_chunk_text(cid)

#             if atomic_score > chunk_map[cid]["chunk_score"]:
#                 chunk_map[cid]["chunk_score"] = atomic_score

#             chunk_map[cid]["chunk_id"] = cid
#             chunk_map[cid]["chunk_text"] = ctext
#             chunk_map[cid]["evidences"].append({
#                 "rule": "single_source_use_atomic_score",
#                 "subquery": sq,
#                 "atomic_id": aid,
#                 "atomic_score": atomic_score,
#             })

#         # ---------------------------
#         # Case 2: multiple source chunks
#         # ---------------------------
#         else:
#             # 🔥 핵심: chunks_vdb로 query ↔ chunk scoring
#             def filter_lambda(dp):
#                 return dp.get("__id__") in src_chunks

#             try:
#                 chunk_results = await chunks_vdb.query(
#                     query=query,
#                     top_k=2,
#                     filter_lambda=filter_lambda,
#                 )
#             except Exception as e:
#                 print(f"[ERROR] Chunk VDB query failed: {e}")
#                 continue

#             for cr in chunk_results:
#                 cid = cr["id"]
#                 qchunk_score = float(cr.get("distance", 0.0))
#                 ctext = await get_chunk_text(cid)

#                 if qchunk_score > chunk_map[cid]["chunk_score"]:
#                     chunk_map[cid]["chunk_score"] = qchunk_score

#                 chunk_map[cid]["chunk_id"] = cid
#                 chunk_map[cid]["chunk_text"] = ctext
#                 chunk_map[cid]["evidences"].append({
#                     "rule": "multi_source_query_chunk_score",
#                     "subquery": sq,
#                     "atomic_id": aid,
#                     "atomic_score": atomic_score,
#                     "query_chunk_score": qchunk_score,
#                     "source_chunk_count": len(src_chunks),
#                 })

#     print(f"[INFO] Step 3 complete — {len(chunk_map)} chunks scored.")

#     # =======================================================
#     # Step 4. Top-K selection + context build
#     # =======================================================
#     chunk_results = list(chunk_map.values())
#     top_k = min(10, len(chunk_results))

#     chunk_results_sorted = sorted(
#         chunk_results,
#         key=lambda x: x["chunk_score"],
#         reverse=True
#     )[:top_k]

#     context_blocks = [c["chunk_text"].strip() for c in chunk_results_sorted]
#     context = "\n\n".join(context_blocks)

#     print(f"[INFO] Step 4 complete — Selected top-{top_k} chunks.")

#     # =======================================================
#     # Step 5. FINAL ANSWER GENERATION (HOTPOT vs MULTIHOP BY cls)
#     # =======================================================

#     # working_dir 기준으로 어떤 데이터셋인지 판단
#     cls = os.path.basename(global_config.get("working_dir", ""))

#     # ---------------------------
#     # HOTPOT one-shot prompt
#     # ---------------------------
#     hotpot_system = (
#         'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
#         'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
#         'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
#     )

#     hotpot_docs = (
#         """The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
#         """Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
#         """Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
#         """Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
#         """Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"""
#     )

#     hotpot_input = (
#         f"{hotpot_docs}"
#         "\n\nQuestion: "
#         "When was Neville A. Stanton's employer founded?"
#         "\nThought: "
#     )

#     hotpot_output = (
#         "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
#         "\nAnswer: 1862."
#     )

#     # ---------------------------
#     # MULTIHOP one-shot prompt
#     # ---------------------------
#     multihop_system = (
#         'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
#         'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
#         'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
#         'Use only the given Information; if it is insufficient, reply with "Answer: Insufficient information.".'
#         'If you need to answer like yes or no, use "Answer: Yes" or "Answer: No" only.'
#     )

#     multihop_docs = (
#         """Milk and Honey (album)\nMilk and Honey is an album by John Lennon and Yoko Ono released in 1984. Following the compilation "The John Lennon Collection", it is Lennon's eighth and final studio album, and the first posthumous release of new Lennon music, having been recorded in the last months of his life during and following the sessions for their 1980 album "Double Fantasy". It was assembled by Yoko Ono in association with the Geffen label.\n\n"""
#         """John Lennon Museum\nJohn Lennon Museum (ジョン・レノン・ミュージアム , Jon Renon Myūjiamu ) was a museum located inside the Saitama Super Arena in Chūō-ku, Saitama, Saitama Prefecture, Japan. It was established to preserve knowledge of John Lennon's life and musical career. It displayed Lennon's widow Yoko Ono's collection of his memorabilia as well as other displays. The museum opened on October 9, 2000, the 60th anniversary of Lennon's birth, and closed on September 30, 2010, when its exhibit contract with Yoko Ono expired. A tour of the museum began with a welcoming message and short film narrated by Yoko Ono (in Japanese with English headphones available), and ended at an avant-garde styled "reflection room" full of chairs facing a slide show of moving words and images. After this room there was a gift shop with John Lennon memorabilia available.\n\n"""
#         """Walls and Bridges\nWalls and Bridges is the fifth studio album by English musician John Lennon. It was issued by Apple Records on 26 September 1974 in the United States and on 4 October in the United Kingdom. Written, recorded and released during his 18-month separation from Yoko Ono, the album captured Lennon in the midst of his "Lost Weekend". "Walls and Bridges" was an American "Billboard" number-one album and featured two hit singles, "Whatever Gets You thru the Night" and "#9 Dream". The first of these was Lennon's first number-one hit in the United States as a solo artist, and his only chart-topping single in either the US or Britain during his lifetime.\n\n"""
#         """Nobody Loves You (When You're Down and Out)\n"Nobody Loves You (When You're Down and Out)" is a song written by John Lennon released on his 1974 album "Walls and Bridges". The song is included on the 1986 compilation "Menlove Ave.", the 1990 boxset "Lennon", the 1998 boxset "John Lennon Anthology", the 2005 two-disc compilation "", and the 2010 boxset "Gimme Some Truth".\n\n"""
#         """Give Peace a Chance\n"Give Peace a Chance" is an anti-war song written by John Lennon (credited to Lennon–McCartney), and performed with Yoko Ono in Montreal, Quebec, Canada. Released as a single in 1969 by the Plastic Ono Band on Apple Records (catalogue Apple 13 in the United Kingdom, Apple 1809 in the United States), it is the first solo single issued by Lennon, released when he was still a member of the Beatles, and became an anthem of the American anti-war movement during the 1970s. It peaked at number 14 on the "Billboard" Hot 100 and number 2 on the British singles chart.\n"""
#     )

#     multihop_input = (
#         f"{multihop_docs}"
#         "\n\nQuestion: "
#         "Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?"
#         "\nThought: "
#     )

#     multihop_output = (
#         "The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges. Nobody Loves You was written by John Lennon on Walls and Bridges album. "
#         "\nAnswer: Walls and Bridges."
#     )

#     # -------------------------------------------------------
#     # cls 이름을 기준으로 사용할 프롬프트 선택
#     # -------------------------------------------------------
#     cls_lower = cls.lower()
#     if "multihop" in cls_lower:
#         system_prompt = multihop_system
#         one_shot_input = multihop_input
#         one_shot_output = multihop_output
#     else:
#         # 기본값 hotpot (cls에 hotpot이 포함되어 있거나, 그 외 케이스)
#         system_prompt = hotpot_system
#         one_shot_input = hotpot_input
#         one_shot_output = hotpot_output

#     # -------------------------------------------------------
#     # 최종 유저 프롬프트 (retrieved context + query)
#     # -------------------------------------------------------
#     final_user_prompt = (
#         f"{context}"
#         f"\n\nQuestion: {query}"
#         "\nThought: "
#     )

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": one_shot_input},
#         {"role": "assistant", "content": one_shot_output},
#         {"role": "user", "content": final_user_prompt},
#     ]

#     try:
#         prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])

#         final_response = await global_config["llm_model_func"](
#             prompt=prompt,
#             stream=getattr(param, "stream", False),
#         )
#         def extract_answer(response: str) -> str:
#             if "Answer:" in response:
#                 return response.split("Answer:", 1)[1].strip()
#             return response.strip()

#         final_response = extract_answer(final_response)
#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return "Final generation failed."


#     # =======================================================
#     # Step 6. SAVE FULL JSON LOG
#     # =======================================================
#     try:
#         # cls는 위에서 이미 계산됨
#         save_dir = f"/workspace/AtomRAG/{cls}"
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f"{cls}_full_results.json")

#         full_output = {
#             "query": query,
#             "dataset_cls": cls,
#             "subqueries": queries,
#             "final_context": context,
#             "messages_prompt_used": messages,
#             "final_answer": (
#                 final_response if isinstance(final_response, str)
#                 else str(final_response)
#             ),
#         }

#         if os.path.exists(save_path):
#             with open(save_path, "r", encoding="utf-8") as f:
#                 try:
#                     existing = json.load(f)
#                     if not isinstance(existing, list):
#                         existing = [existing]
#                 except Exception:
#                     existing = []
#         else:
#             existing = []

#         existing.append(full_output)

#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(existing, f, ensure_ascii=False, indent=4)

#         print(f"[INFO] JSON saved to {save_path}")

#     except Exception as e:
#         print(f"[ERROR] Saving JSON failed: {e}")

#     print(final_response)
#     return final_response








# ####### 일반 chunk를 넣어주고 원래 query와 atomic에서 교집합 chunk만 그대로 나머지는 summary, cot방식의 generation을 사용한 방법#########
# async def ours_kg_query_experiment1(
#     query: str,
#     atomic_knowledge_graph_inst,
#     atomic_entities_vdb,
#     chunks_vdb,
#     text_triple_db,
#     text_atomics_db,
#     text_chunks_db,
#     param,
#     global_config: dict,
#     hashing_kv=None,
# ) -> str:
#     """
#     Simplified version:
#     - Run Step 0~3 (retrieval)
#     - Use top-K raw chunks as context
#     - Generate final answer using structured-CoT RAG format
#     - Save entire reasoning chain to JSON file
#     """

#     # =======================================================
#     # Step 0. Query decomposition
#     # =======================================================
#     query_modes = {
#         "experiment1": query_process_1,
#         "experiment2": query_process_2,
#         "experiment3": query_process_3,
#         "experiment4": query_process_4,
#     }
#     queries = await query_modes.get(param.query_mode, lambda *a: [query])(
#         query, param, global_config, hashing_kv
#     )
#     if not queries:
#         queries = [query]

#     print(f"[INFO] Step 0 complete — {len(queries)} subqueries generated.")


#     # =======================================================
#     # Step 1. Atomic fact retrieval
#     # =======================================================
#     print(f"[INFO] Retrieving atomic facts for {len(queries)} subqueries...")

#     async def fetch_atomic_facts(sq):
#         try:
#             results = await atomic_entities_vdb.query(sq, top_k=10)
#             return sq, results
#         except Exception as e:
#             print(f"[ERROR] Retrieval failed for '{sq}': {e}")
#             return sq, []

#     subquery_results = dict(
#         await asyncio.gather(*(fetch_atomic_facts(sq) for sq in queries))
#     )

#     step1_retrieval_log = {
#         sq: [
#             {
#                 "atomic_id": r.get("id"),
#                 "entity_name": r.get("entity_name", ""),
#                 "score": r.get("distance", 0.0),
#             }
#             for r in results
#         ]
#         for sq, results in subquery_results.items()
#     }


#     # =======================================================
#     # Step 2. Atomic → Chunk mapping
#     # =======================================================
#     chunk_to_atomics = defaultdict(lambda: {"atomic_facts": []})
#     chunk_cache = {}

#     retrieved_items = []
#     for sq, results in subquery_results.items():
#         for r in results:
#             rid = r.get("id")
#             if rid:
#                 retrieved_items.append((sq, rid, r.get("distance", 0.0)))

#     if not retrieved_items:
#         return "No results."

#     unique_atomic_ids = list({rid for _, rid, _ in retrieved_items})
#     atomic_records = await text_atomics_db.get_by_ids(unique_atomic_ids)

#     atomic_info_map = {}
#     for aid, rec in zip(unique_atomic_ids, atomic_records or []):
#         if rec and isinstance(rec, dict):
#             atomic_info_map[aid] = {
#                 "content": rec.get("content", ""),
#                 "source_ids": rec.get("source_ids", []) or [],
#             }

#     for sq, aid, score in retrieved_items:
#         ainfo = atomic_info_map.get(aid)
#         if not ainfo:
#             continue

#         atomic_text = ainfo["content"]
#         src_chunks = ainfo["source_ids"]

#         for cid in src_chunks:
#             if cid not in chunk_cache:
#                 try:
#                     cdata = await text_chunks_db.get_by_ids([cid])
#                     chunk_cache[cid] = cdata[0].get("content", "") if cdata else ""
#                 except Exception:
#                     chunk_cache[cid] = ""

#             chunk_to_atomics[cid]["chunk_id"] = cid
#             chunk_to_atomics[cid]["chunk_text"] = chunk_cache[cid]
#             chunk_to_atomics[cid]["atomic_facts"].append({
#                 "atomic_id": aid,
#                 "atomic_text": atomic_text,
#                 "sub_query": sq,
#                 "score": score,
#             })

#     for item in chunk_to_atomics.values():
#         seen = set()
#         uniq = []
#         for af in item["atomic_facts"]:
#             aid = af["atomic_id"]
#             if aid not in seen:
#                 seen.add(aid)
#                 uniq.append(af)
#         item["atomic_facts"] = uniq

#     chunk_results = list(chunk_to_atomics.values())
#     print(f"[INFO] Step 2 complete — {len(chunk_results)} chunks mapped.")


#     step2_mapping_log = []
#     for item in chunk_results:
#         txt = item.get("chunk_text", "")
#         step2_mapping_log.append({
#             "chunk_id": item["chunk_id"],
#             "chunk_text_preview": txt[:150] + ("..." if len(txt) > 150 else ""),
#             "num_atomics": len(item["atomic_facts"]),
#             "atomic_facts": item["atomic_facts"],
#         })


#     # =======================================================
#     # Step 3. Rank chunks using only max_atomic_score
#     # =======================================================
#     top = 10
#     top_k = min(top, len(chunk_results))

#     for item in chunk_results:
#         item["max_atomic_score"] = max(
#             [af["score"] for af in item["atomic_facts"]], default=0.0
#         )

#     atomic_top_chunks = sorted(
#         chunk_results,
#         key=lambda x: x["max_atomic_score"],
#         reverse=True
#     )[:top_k]

#     atomic_chunk_ids = {item["chunk_id"] for item in atomic_top_chunks}

#     print(f"[INFO] Step 3 complete — Atomic top-{top_k} chunks selected.")
    
#     # =======================================================
#     # Step 3-1. Retrieve chunks using original query
#     # =======================================================

#     print(f"[INFO] Retrieving top-{top_k} chunks using original query...")

#     query_chunk_results = await chunks_vdb.query(
#         query,
#         top_k=top
#     )

#     query_top_chunks = []
#     for r in query_chunk_results:
#         cid = r.get("id")
#         if not cid:
#             continue
#         query_top_chunks.append({
#             "chunk_id": cid,
#             "chunk_text": r.get("content", "")
#         })

#     query_chunk_ids = {item["chunk_id"] for item in query_top_chunks}

#     print(f"[INFO] Step 3-1 complete — Query top-{top_k} chunks selected.")




#     # =======================================================
#     # Step 4. Build hybrid context (raw + summary)
#     # =======================================================

#     context_blocks = []

#     # ---------------------------
#     # 1) Atomic ∩ Query → raw chunk
#     # ---------------------------
#     overlap_chunk_ids = atomic_chunk_ids & query_chunk_ids

#     for item in atomic_top_chunks:
#         cid = item["chunk_id"]
#         if cid in overlap_chunk_ids:
#             context_blocks.append(item.get("chunk_text", "").strip())

#     # ---------------------------
#     # 2) Query-only → summarized chunk
#     # ---------------------------
#     query_only_chunks = [
#         item for item in query_top_chunks
#         if item["chunk_id"] not in atomic_chunk_ids
#     ]

#     async def summarize_chunk(context: str) -> str:
#         messages = [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {
#                 "role": "user",
#                 "content": (
#                     "Write a summary of the following, including as many key details as possible:\n\n"
#                     f"{context}"
#                 ),
#             },
#         ]

#         prompt = "\n".join(
#             [f"{m['role'].upper()}: {m['content']}" for m in messages]
#         )

#         summary = await global_config["llm_model_func"](
#             prompt=prompt,
#             stream=False,
#         )

#         return summary.strip()


#     for item in query_only_chunks:
#         chunk_text = item.get("chunk_text", "").strip()
#         if not chunk_text:
#             continue

#         try:
#             summary = await summarize_chunk(chunk_text)
#             context_blocks.append(summary.strip())
#         except Exception as e:
#             print(f"[WARN] Summary failed for chunk {item['chunk_id']}: {e}")
#             context_blocks.append(chunk_text[:300])  # fallback

#     context = "\n\n".join(context_blocks)
#     print("[INFO] Step 4 complete — Hybrid context built (raw + summary).")


#     # =======================================================
#     # Step 5. FINAL ANSWER GENERATION (HOTPOT vs MULTIHOP BY cls)
#     # =======================================================

#     # working_dir 기준으로 어떤 데이터셋인지 판단
#     cls = os.path.basename(global_config.get("working_dir", ""))

#     # ---------------------------
#     # HOTPOT one-shot prompt
#     # ---------------------------
#     hotpot_system = (
#         'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
#         'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
#         'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
#     )

#     hotpot_docs = (
#         """The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
#         """Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
#         """Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
#         """Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
#         """Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"""
#     )

#     hotpot_input = (
#         f"{hotpot_docs}"
#         "\n\nQuestion: "
#         "When was Neville A. Stanton's employer founded?"
#         "\nThought: "
#     )

#     hotpot_output = (
#         "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
#         "\nAnswer: 1862."
#     )

#     # ---------------------------
#     # MULTIHOP one-shot prompt
#     # ---------------------------
#     multihop_system = (
#         'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
#         'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
#         'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
#         'Use only the given Information; if it is insufficient, reply with "Answer: Insufficient information.".'
#         'If you need to answer like yes or no, use "Answer: Yes" or "Answer: No" only.'
#     )

#     multihop_docs = (
#         """Milk and Honey (album)\nMilk and Honey is an album by John Lennon and Yoko Ono released in 1984. Following the compilation "The John Lennon Collection", it is Lennon's eighth and final studio album, and the first posthumous release of new Lennon music, having been recorded in the last months of his life during and following the sessions for their 1980 album "Double Fantasy". It was assembled by Yoko Ono in association with the Geffen label.\n\n"""
#         """John Lennon Museum\nJohn Lennon Museum (ジョン・レノン・ミュージアム , Jon Renon Myūjiamu ) was a museum located inside the Saitama Super Arena in Chūō-ku, Saitama, Saitama Prefecture, Japan. It was established to preserve knowledge of John Lennon's life and musical career. It displayed Lennon's widow Yoko Ono's collection of his memorabilia as well as other displays. The museum opened on October 9, 2000, the 60th anniversary of Lennon's birth, and closed on September 30, 2010, when its exhibit contract with Yoko Ono expired. A tour of the museum began with a welcoming message and short film narrated by Yoko Ono (in Japanese with English headphones available), and ended at an avant-garde styled "reflection room" full of chairs facing a slide show of moving words and images. After this room there was a gift shop with John Lennon memorabilia available.\n\n"""
#         """Walls and Bridges\nWalls and Bridges is the fifth studio album by English musician John Lennon. It was issued by Apple Records on 26 September 1974 in the United States and on 4 October in the United Kingdom. Written, recorded and released during his 18-month separation from Yoko Ono, the album captured Lennon in the midst of his "Lost Weekend". "Walls and Bridges" was an American "Billboard" number-one album and featured two hit singles, "Whatever Gets You thru the Night" and "#9 Dream". The first of these was Lennon's first number-one hit in the United States as a solo artist, and his only chart-topping single in either the US or Britain during his lifetime.\n\n"""
#         """Nobody Loves You (When You're Down and Out)\n"Nobody Loves You (When You're Down and Out)" is a song written by John Lennon released on his 1974 album "Walls and Bridges". The song is included on the 1986 compilation "Menlove Ave.", the 1990 boxset "Lennon", the 1998 boxset "John Lennon Anthology", the 2005 two-disc compilation "", and the 2010 boxset "Gimme Some Truth".\n\n"""
#         """Give Peace a Chance\n"Give Peace a Chance" is an anti-war song written by John Lennon (credited to Lennon–McCartney), and performed with Yoko Ono in Montreal, Quebec, Canada. Released as a single in 1969 by the Plastic Ono Band on Apple Records (catalogue Apple 13 in the United Kingdom, Apple 1809 in the United States), it is the first solo single issued by Lennon, released when he was still a member of the Beatles, and became an anthem of the American anti-war movement during the 1970s. It peaked at number 14 on the "Billboard" Hot 100 and number 2 on the British singles chart.\n"""
#     )

#     multihop_input = (
#         f"{multihop_docs}"
#         "\n\nQuestion: "
#         "Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?"
#         "\nThought: "
#     )

#     multihop_output = (
#         "The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges. Nobody Loves You was written by John Lennon on Walls and Bridges album. "
#         "\nAnswer: Walls and Bridges."
#     )

#     # -------------------------------------------------------
#     # cls 이름을 기준으로 사용할 프롬프트 선택
#     # -------------------------------------------------------
#     cls_lower = cls.lower()
#     if "multihop" in cls_lower:
#         system_prompt = multihop_system
#         one_shot_input = multihop_input
#         one_shot_output = multihop_output
#     else:
#         # 기본값 hotpot (cls에 hotpot이 포함되어 있거나, 그 외 케이스)
#         system_prompt = hotpot_system
#         one_shot_input = hotpot_input
#         one_shot_output = hotpot_output

#     # -------------------------------------------------------
#     # 최종 유저 프롬프트 (retrieved context + query)
#     # -------------------------------------------------------
#     final_user_prompt = (
#         f"{context}"
#         f"\n\nQuestion: {query}"
#         "\nThought: "
#     )

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": one_shot_input},
#         {"role": "assistant", "content": one_shot_output},
#         {"role": "user", "content": final_user_prompt},
#     ]

#     try:
#         prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])

#         final_response = await global_config["llm_model_func"](
#             prompt=prompt,
#             stream=getattr(param, "stream", False),
#         )
#         def extract_answer(response: str) -> str:
#             if "Answer:" in response:
#                 return response.split("Answer:", 1)[1].strip()
#             return response.strip()

#         final_response = extract_answer(final_response)
#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return "Final generation failed."


#     # =======================================================
#     # Step 6. SAVE FULL JSON LOG
#     # =======================================================
#     try:
#         # cls는 위에서 이미 계산됨
#         save_dir = f"/workspace/AtomRAG/{cls}"
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f"{cls}_full_results.json")

#         full_output = {
#             "query": query,
#             "dataset_cls": cls,
#             "subqueries": queries,
#             "step1_retrieval": step1_retrieval_log,
#             "step2_mapping": step2_mapping_log,
#             "final_context": context,
#             "messages_prompt_used": messages,
#             "final_answer": (
#                 final_response if isinstance(final_response, str)
#                 else str(final_response)
#             ),
#         }

#         if os.path.exists(save_path):
#             with open(save_path, "r", encoding="utf-8") as f:
#                 try:
#                     existing = json.load(f)
#                     if not isinstance(existing, list):
#                         existing = [existing]
#                 except Exception:
#                     existing = []
#         else:
#             existing = []

#         existing.append(full_output)

#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(existing, f, ensure_ascii=False, indent=4)

#         print(f"[INFO] JSON saved to {save_path}")

#     except Exception as e:
#         print(f"[ERROR] Saving JSON failed: {e}")

#     print(final_response)
#     return final_response


# ####### atom reranking +  cot방식의 generation을 사용한 방법#########
# async def ours_kg_query_experiment1(
#     query: str,
#     atomic_knowledge_graph_inst,
#     atomic_entities_vdb,
#     text_triple_db,
#     text_atomics_db,
#     text_chunks_db,
#     param,
#     global_config: dict,
#     hashing_kv=None,
# ) -> str:
#     """
#     Simplified version:
#     - Run Step 0~3 (retrieval + combined scoring)
#     - Use top-K raw chunks as context
#     - Generate final answer using structured-CoT RAG format
#     - Save entire reasoning chain to JSON file
#     """

#     import os
#     import json
#     import asyncio
#     from collections import defaultdict

#     # =======================================================
#     # Step 0. Query decomposition  (첫 번째 코드 버전)
#     # =======================================================
#     query_modes = {
#         "experiment1": query_process_1,
#         "experiment2": query_process_2,
#         "experiment3": query_process_3,
#         "experiment4": query_process_4,
#     }
#     queries = await query_modes.get(param.query_mode, lambda *a: [query])(
#         query, param, global_config, hashing_kv
#     )
#     if not queries:
#         queries = [query]

#     print(f"[INFO] Step 0 complete — {len(queries)} subqueries generated.")

#     # =======================================================
#     # Step 1. Atomic fact retrieval  (첫 번째 코드 버전)
#     # =======================================================
#     print(f"[INFO] Retrieving atomic facts for {len(queries)} subqueries...")

#     async def fetch_atomic_facts(sq):
#         try:
#             results = await atomic_entities_vdb.query(sq, top_k=10)
#             return sq, results
#         except Exception as e:
#             print(f"[ERROR] Retrieval failed for '{sq}': {e}")
#             return sq, []

#     subquery_results = dict(
#         await asyncio.gather(*(fetch_atomic_facts(sq) for sq in queries))
#     )

#     # store logs
#     step1_retrieval_log = {
#         sq: [
#             {
#                 "atomic_id": r.get("id"),
#                 "entity_name": r.get("entity_name", ""),
#                 "score": r.get("distance", 0.0),
#             }
#             for r in results
#         ]
#         for sq, results in subquery_results.items()
#     }

#     # =======================================================
#     # Step 2. Atomic → Chunk mapping  (첫 번째 코드 버전)
#     # =======================================================
#     chunk_to_atomics = defaultdict(lambda: {"atomic_facts": []})
#     chunk_cache = {}

#     retrieved_items = []
#     for sq, results in subquery_results.items():
#         for r in results:
#             rid = r.get("id")
#             if rid:
#                 retrieved_items.append((sq, rid, r.get("distance", 0.0)))

#     if not retrieved_items:
#         return "No results."

#     unique_atomic_ids = list({rid for _, rid, _ in retrieved_items})
#     atomic_records = await text_atomics_db.get_by_ids(unique_atomic_ids)

#     atomic_info_map = {}
#     for aid, rec in zip(unique_atomic_ids, atomic_records or []):
#         if rec and isinstance(rec, dict):
#             atomic_info_map[aid] = {
#                 "content": rec.get("content", ""),
#                 "source_ids": rec.get("source_ids", []) or [],
#             }

#     # mapping
#     for sq, aid, score in retrieved_items:
#         ainfo = atomic_info_map.get(aid)
#         if not ainfo:
#             continue

#         atomic_text = ainfo["content"]
#         src_chunks = ainfo["source_ids"]

#         for cid in src_chunks:
#             # load chunk on demand
#             if cid not in chunk_cache:
#                 try:
#                     cdata = await text_chunks_db.get_by_ids([cid])
#                     chunk_cache[cid] = cdata[0].get("content", "") if cdata else ""
#                 except Exception:
#                     chunk_cache[cid] = ""

#             chunk_to_atomics[cid]["chunk_id"] = cid
#             chunk_to_atomics[cid]["chunk_text"] = chunk_cache[cid]
#             chunk_to_atomics[cid]["atomic_facts"].append({
#                 "atomic_id": aid,
#                 "atomic_text": atomic_text,
#                 "sub_query": sq,
#                 "score": score,
#             })

#     # remove duplicates
#     for item in chunk_to_atomics.values():
#         seen = set()
#         uniq = []
#         for af in item["atomic_facts"]:
#             aid = af["atomic_id"]
#             if aid not in seen:
#                 seen.add(aid)
#                 uniq.append(af)
#         item["atomic_facts"] = uniq

#     chunk_results = list(chunk_to_atomics.values())
#     print(f"[INFO] Step 2 complete — {len(chunk_results)} chunks mapped.")

#     # Log step2
#     step2_mapping_log = []
#     for item in chunk_results:
#         txt = item.get("chunk_text", "")
#         step2_mapping_log.append({
#             "chunk_id": item["chunk_id"],
#             "chunk_text_preview": txt[:150] + ("..." if len(txt) > 150 else ""),
#             "num_atomics": len(item["atomic_facts"]),
#             "atomic_facts": item["atomic_facts"],
#         })

#     # =======================================================
#     # Step 3. Rank chunks (첫 번째 코드의 combined scoring 사용)
#     # =======================================================

#     # 1) 각 chunk에 대해 atomic fact 개수와 max atomic score 계산
#     for item in chunk_results:
#         num_atomics = len(item["atomic_facts"])
#         max_score = max([af["score"] for af in item["atomic_facts"]], default=0.0)

#         item["num_atomics"] = num_atomics
#         item["max_atomic_score"] = max_score

#     # 2) normalization (min-max)
#     def safe_norm(val, min_v, max_v):
#         if max_v - min_v < 1e-9:
#             return 0.0
#         return (val - min_v) / (max_v - min_v)

#     all_nums = [x["num_atomics"] for x in chunk_results]
#     all_scores = [x["max_atomic_score"] for x in chunk_results]

#     min_num, max_num = min(all_nums), max(all_nums)
#     min_score, max_score = min(all_scores), max(all_scores)

#     # 3) Combined Score
#     w1 = 0.6  # atomic_fact_count weight
#     w2 = 0.4  # max_atomic_score weight

#     for item in chunk_results:
#         norm_num = safe_norm(item["num_atomics"], min_num, max_num)
#         norm_score = safe_norm(item["max_atomic_score"], min_score, max_score)
#         item["chunk_score"] = w1 * norm_num + w2 * norm_score

#     # top-10 선택
#     top_k = min(5, len(chunk_results))
#     chunk_results_sorted = sorted(
#         chunk_results,
#         key=lambda x: x["chunk_score"],
#         reverse=True
#     )[:top_k]

#     print(f"[INFO] Step 3 complete — Selected top-{top_k} chunks (combined scoring).")

#     # Log update
#     step3_topk_log = [
#         {
#             "chunk_id": item["chunk_id"],
#             "chunk_score": item["chunk_score"],
#             "num_atomics": item["num_atomics"],
#             "max_atomic_score": item["max_atomic_score"],
#             "atomic_ids": [af["atomic_id"] for af in item["atomic_facts"]],
#         }
#         for item in chunk_results_sorted
#     ]

#     # =======================================================
#     # Step 4. Build simple context (structured-CoT 버전의 Step4 그대로)
#     #        → label 없이 chunk 텍스트만 이어붙임
#     # =======================================================

#     context_blocks = []
#     for item in chunk_results_sorted:
#         chunk_text = item.get("chunk_text", "").strip()
#         context_blocks.append(chunk_text)

#     context = "\n\n".join(context_blocks)
#     print("[INFO] Step 4 complete — Context built (no labels).")

#     # =======================================================
#     # Step 5. FINAL ANSWER GENERATION (HOTPOT vs MULTIHOP BY cls)
#     #        (structured-CoT 버전 그대로)
#     # =======================================================

#     # working_dir 기준으로 어떤 데이터셋인지 판단
#     cls = os.path.basename(global_config.get("working_dir", ""))

#     # ---------------------------
#     # HOTPOT one-shot prompt
#     # ---------------------------
#     hotpot_system = (
#         'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
#         'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
#         'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
#     )

#     hotpot_docs = (
#         """The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
#         """Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
#         """Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
#         """Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
#         """Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"""
#     )

#     hotpot_input = (
#         f"{hotpot_docs}"
#         "\n\nQuestion: "
#         "When was Neville A. Stanton's employer founded?"
#         "\nThought: "
#     )

#     hotpot_output = (
#         "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
#         "\nAnswer: 1862."
#     )

#     # ---------------------------
#     # MULTIHOP one-shot prompt
#     # ---------------------------
#     multihop_system = (
#         'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
#         'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
#         'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
#         'Use only the given Information; if it is insufficient, reply with "Answer: Insufficient information.".'
#         'If you need to answer like yes or no, use "Answer: Yes" or "Answer: No" only.'
#     )

#     multihop_docs = (
#         """Milk and Honey (album)\nMilk and Honey is an album by John Lennon and Yoko Ono released in 1984. Following the compilation "The John Lennon Collection", it is Lennon's eighth and final studio album, and the first posthumous release of new Lennon music, having been recorded in the last months of his life during and following the sessions for their 1980 album "Double Fantasy". It was assembled by Yoko Ono in association with the Geffen label.\n\n"""
#         """John Lennon Museum\nJohn Lennon Museum (ジョン・レノン・ミュージアム , Jon Renon Myūjiamu ) was a museum located inside the Saitama Super Arena in Chūō-ku, Saitama, Saitama Prefecture, Japan. It was established to preserve knowledge of John Lennon's life and musical career. It displayed Lennon's widow Yoko Ono's collection of his memorabilia as well as other displays. The museum opened on October 9, 2000, the 60th anniversary of Lennon's birth, and closed on September 30, 2010, when its exhibit contract with Yoko Ono expired. A tour of the museum began with a welcoming message and short film narrated by Yoko Ono (in Japanese with English headphones available), and ended at an avant-garde styled "reflection room" full of chairs facing a slide show of moving words and images. After this room there was a gift shop with John Lennon memorabilia available.\n\n"""
#         """Walls and Bridges\nWalls and Bridges is the fifth studio album by English musician John Lennon. It was issued by Apple Records on 26 September 1974 in the United States and on 4 October in the United Kingdom. Written, recorded and released during his 18-month separation from Yoko Ono, the album captured Lennon in the midst of his "Lost Weekend". "Walls and Bridges" was an American "Billboard" number-one album and featured two hit singles, "Whatever Gets You thru the Night" and "#9 Dream". The first of these was Lennon's first number-one hit in the United States as a solo artist, and his only chart-topping single in either the US or Britain during his lifetime.\n\n"""
#         """Nobody Loves You (When You're Down and Out)\n"Nobody Loves You (When You're Down and Out)" is a song written by John Lennon released on his 1974 album "Walls and Bridges". The song is included on the 1986 compilation "Menlove Ave.", the 1990 boxset "Lennon", the 1998 boxset "John Lennon Anthology", the 2005 two-disc compilation "", and the 2010 boxset "Gimme Some Truth".\n\n"""
#         """Give Peace a Chance\n"Give Peace a Chance" is an anti-war song written by John Lennon (credited to Lennon–McCartney), and performed with Yoko Ono in Montreal, Quebec, Canada. Released as a single in 1969 by the Plastic Ono Band on Apple Records (catalogue Apple 13 in the United Kingdom, Apple 1809 in the United States), it is the first solo single issued by Lennon, released when he was still a member of the Beatles, and became an anthem of the American anti-war movement during the 1970s. It peaked at number 14 on the "Billboard" Hot 100 and number 2 on the British singles chart.\n"""
#     )

#     multihop_input = (
#         f"{multihop_docs}"
#         "\n\nQuestion: "
#         "Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?"
#         "\nThought: "
#     )

#     multihop_output = (
#         "The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges. Nobody Loves You was written by John Lennon on Walls and Bridges album. "
#         "\nAnswer: Walls and Bridges."
#     )

#     # -------------------------------------------------------
#     # cls 이름을 기준으로 사용할 프롬프트 선택
#     # -------------------------------------------------------
#     cls_lower = cls.lower()
#     if "multihop" in cls_lower:
#         system_prompt = multihop_system
#         one_shot_input = multihop_input
#         one_shot_output = multihop_output
#     else:
#         # 기본값 hotpot
#         system_prompt = hotpot_system
#         one_shot_input = hotpot_input
#         one_shot_output = hotpot_output

#     # -------------------------------------------------------
#     # 최종 유저 프롬프트 (retrieved context + query)
#     # -------------------------------------------------------
#     final_user_prompt = (
#         f"{context}"
#         f"\n\nQuestion: {query}"
#         "\nThought: "
#     )

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": one_shot_input},
#         {"role": "assistant", "content": one_shot_output},
#         {"role": "user", "content": final_user_prompt},
#     ]

#     try:
#         prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])

#         final_response = await global_config["llm_model_func"](
#             prompt=prompt,
#             stream=getattr(param, "stream", False),
#         )

#         def extract_answer(response: str) -> str:
#             if "Answer:" in response:
#                 return response.split("Answer:", 1)[1].strip()
#             return response.strip()

#         final_response = extract_answer(final_response)

#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return "Final generation failed."

#     # =======================================================
#     # Step 6. SAVE FULL JSON LOG (structured-CoT 버전 그대로)
#     # =======================================================
#     try:
#         save_dir = f"/workspace/AtomRAG/{cls}"
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f"{cls}_full_results.json")

#         full_output = {
#             "query": query,
#             "dataset_cls": cls,
#             "subqueries": queries,
#             "step1_retrieval": step1_retrieval_log,
#             "step2_mapping": step2_mapping_log,
#             "step3_topk_chunks": step3_topk_log,
#             "final_context": context,
#             "messages_prompt_used": messages,
#             "final_answer": (
#                 final_response if isinstance(final_response, str)
#                 else str(final_response)
#             ),
#         }

#         if os.path.exists(save_path):
#             with open(save_path, "r", encoding="utf-8") as f:
#                 try:
#                     existing = json.load(f)
#                     if not isinstance(existing, list):
#                         existing = [existing]
#                 except Exception:
#                     existing = []
#         else:
#             existing = []

#         existing.append(full_output)

#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(existing, f, ensure_ascii=False, indent=4)

#         print(f"[INFO] JSON saved to {save_path}")

#     except Exception as e:
#         print(f"[ERROR] Saving JSON failed: {e}")

#     print(final_response)
#     return final_response




# ####### 일반 chunk를 넣어주고 새로운 cot방식의 generation을 사용한 방법#########
# import asyncio
# import os
# import json
# from collections import defaultdict
# import spacy


# async def ours_kg_query_experiment1(
#     query: str,
#     atomic_knowledge_graph_inst,
#     atomic_entities_vdb,
#     text_triple_db,
#     text_atomics_db,
#     text_chunks_db,
#     param,
#     global_config: dict,
#     hashing_kv=None,
# ) -> dict:
#     """
#     Retrieval + Evidence Sentence Extraction only (No Generation)
#     Step 0~4:
#         - Query decomposition
#         - Atomic retrieval
#         - Atomic → Chunk mapping
#         - Sentence-level evidence extraction with indexing
#     Return dictionary for next-stage generation
#     """

#     # =======================================================
#     # Step 0. Query decomposition
#     # =======================================================
#     query_modes = {
#         "experiment1": query_process_1,
#         "experiment2": query_process_2,
#         "experiment3": query_process_3,
#         "experiment4": query_process_4,
#     }

#     queries = await query_modes.get(param.query_mode, lambda *a: [query])(
#         query, param, global_config, hashing_kv
#     )
#     if not queries:
#         queries = [query]

#     print(f"[INFO] Step 0 complete — {len(queries)} subqueries generated.")


#     # =======================================================
#     # Step 1. Atomic fact retrieval
#     # =======================================================
#     print(f"[INFO] Retrieving atomic facts for {len(queries)} subqueries...]")

#     async def fetch_atomic(sq):
#         try:
#             results = await atomic_entities_vdb.query(sq, top_k=10)
#             return sq, results
#         except Exception as e:
#             print(f"[ERROR] Retrieval failed for '{sq}': {e}")
#             return sq, []

#     subquery_results = dict(
#         await asyncio.gather(*(fetch_atomic(sq) for sq in queries))
#     )

#     step1_retrieval_log = {
#         sq: [
#             {
#                 "atomic_id": r.get("id"),
#                 "score": r.get("distance", 0.0),
#             }
#             for r in results
#         ]
#         for sq, results in subquery_results.items()
#     }


#     # =======================================================
#     # Step 2. Atomic → Chunk mapping
#     # =======================================================
#     chunk_to_atomics = defaultdict(lambda: {"atomic_facts": []})
#     chunk_cache = {}

#     retrieved = []
#     for sq, results in subquery_results.items():
#         for r in results:
#             rid = r.get("id")
#             if rid:
#                 retrieved.append((sq, rid, r.get("distance", 0.0)))

#     if not retrieved:
#         return {"status": "no_result"}

#     unique_atomic_ids = list({rid for _, rid, _ in retrieved})
#     atomic_records = await text_atomics_db.get_by_ids(unique_atomic_ids)

#     atomic_info = {}
#     for aid, rec in zip(unique_atomic_ids, atomic_records or []):
#         if rec and isinstance(rec, dict):
#             atomic_info[aid] = {
#                 "text": rec.get("content", ""),
#                 "source_ids": rec.get("source_ids", []) or [],
#             }

#     for sq, aid, score in retrieved:
#         info = atomic_info.get(aid)
#         if not info:
#             continue

#         for cid in info["source_ids"]:
#             if cid not in chunk_cache:
#                 cdata = await text_chunks_db.get_by_ids([cid])
#                 chunk_cache[cid] = cdata[0].get("content", "") if cdata else ""

#             chunk_to_atomics[cid]["chunk_id"] = cid
#             chunk_to_atomics[cid]["chunk_text"] = chunk_cache[cid]
#             chunk_to_atomics[cid]["atomic_facts"].append(
#                 {"atomic_id": aid, "score": score}
#             )

#     chunk_results = list(chunk_to_atomics.values())
#     print(f"[INFO] Step 2 complete — {len(chunk_results)} chunks mapped.")


#     # =======================================================
#     # Step 3. Rank chunks (max atomic score)
#     # =======================================================
#     for item in chunk_results:
#         item["max_score"] = max(
#             [af["score"] for af in item["atomic_facts"]],
#             default=0.0
#         )

#     top_k = min(10, len(chunk_results))
#     top_chunks = sorted(
#         chunk_results,
#         key=lambda x: x["max_score"],
#         reverse=True
#     )[:top_k]

#     print(f"[INFO] Step 3 complete — Selected top-{top_k} chunks.")


#     # =======================================================
#     # Step 4. Sentence-level evidence extraction
#     # =======================================================
#     print("[INFO] Step 4 — Splitting into indexed sentences...")

#     # Create SpaCy sentencizer once
#     if "spacy_nlp" not in globals():
#         global spacy_nlp
#         spacy_nlp = spacy.blank("en")
#         spacy_nlp.add_pipe("sentencizer")

#     evidence_by_chunk = []
#     chunk_idx = 1

#     for item in top_chunks:
#         doc = spacy_nlp(item["chunk_text"].strip())
#         sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

#         indexed_sents = []
#         for s_idx, s in enumerate(sentences, start=1):
#             indexed_sents.append(f"[{chunk_idx}-{s_idx}] {s}")

#         evidence_by_chunk.append({
#             "chunk_id": chunk_idx,
#             "sentences": indexed_sents
#         })
#         chunk_idx += 1

#     print(f"[INFO] Step 4 complete — {sum(len(c['sentences']) for c in evidence_by_chunk)} sentences extracted.")


#     # Format evidence as CHUNKS like one-shot prompt
#     formatted_context = "Retrieved chunks:\n"
#     for chunk_obj in evidence_by_chunk:
#         cid = chunk_obj["chunk_id"]
#         formatted_context += f"\nchunk {cid}\n"
#         for sent in chunk_obj["sentences"]:
#             formatted_context += f"{sent}\n"

#     # =======================================================
#     # Step 5. FINAL ANSWER GENERATION (HOTPOT vs MULTIHOP BY cls)
#     # =======================================================

#     # working_dir 기준으로 어떤 데이터셋인지 판단
#     cls = os.path.basename(global_config.get("working_dir", ""))

#     # ---------------------------
#     # HOTPOT one-shot prompt
#     # ---------------------------
#     hotpot_system = (
#         'As an advanced reading comprehension assistant, your task is to analyze text chunk and corresponding questions meticulously. '
#         'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
#         'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
#     )

#     hotpot_docs = (
#         """chunk 1
#         [1-1] The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n
#         [1-2] Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students.\n 
#         [1-3] The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010.\n
#         [1-4] In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world.\n 
#         [1-5] The university considers itself one of the top 5 research universities in the UK.\n
#         [1-6] The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.)\n
#         [1-7] It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.
#         """
#         """chunk 2
#         [2-1] Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA.\n
#         [2-2] As of the 2010 census, its population was 505 and it contained 202 housing units.\n
#         [2-3] Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton.\n 
#         [2-4] Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF).\n 
#         [2-5] He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject.\n 
#         [2-6] Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology.\n 
#         [2-7] He has been published in academic journals including "Nature".\n 
#         [2-8] He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.
#         """
#         """chunk 3
#         [3-1] Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million
#         """
#     )

#     hotpot_input = (
#         f"{hotpot_docs}"
#         "\n\nQuestion: "
#         "When was Neville A. Stanton's employer founded?"
#         "\nThought: "
#     )

#     hotpot_output = (
#         "[2-3] Neville A. Stanton is a British Professor\n"
#         "[1-2] The University of Southampton, which was founded in 1862."
#         "\nAnswer: 1862."
#     )

#     # ---------------------------
#     # MULTIHOP one-shot prompt
#     # ---------------------------
#     multihop_system = (
#         'As an advanced reading comprehension assistant, your task is to analyze text chunk and corresponding questions meticulously. '
#         'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
#         'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
#         'Use only the given Information; if it is insufficient, reply with "Answer: Insufficient information.".'
#         'If you need to answer like yes or no, use "Answer: Yes" or "Answer: No" only.'
#     )

#     multihop_docs = (
#         """chunk 1
#         [1-1] Milk and Honey (album)\nMilk and Honey is an album by John Lennon and Yoko Ono released in 1984. 
#         [1-2] Following the compilation "The John Lennon Collection", it is Lennon's eighth and final studio album, and the first posthumous release of new Lennon music, having been recorded in the last months of his life during and following the sessions for their 1980 album "Double Fantasy". 
#         [1-3] It was assembled by Yoko Ono in association with the Geffen label.\n\n
#         [1-4] John Lennon Museum\nJohn Lennon Museum (ジョン・レノン・ミュージアム , Jon Renon Myūjiamu ) was a museum located inside the Saitama Super Arena in Chūō-ku, Saitama, Saitama Prefecture, Japan. 
#         [1-5] It was established to preserve knowledge of John Lennon's life and musical career. 
#         [1-6] It displayed Lennon's widow Yoko Ono's collection of his memorabilia as well as other displays. 
#         [1-7] The museum opened on October 9, 2000, the 60th anniversary of Lennon's birth, and closed on September 30, 2010, when its exhibit contract with Yoko Ono expired. 
#         [1-8] A tour of the museum began with a welcoming message and short film narrated by Yoko Ono (in Japanese with English headphones available), and ended at an avant-garde styled "reflection room" full of chairs facing a slide show of moving words and images. 
#         [1-9] After this room there was a gift shop with John Lennon memorabilia available.\n\
#         """
#         """chunk 2
#         [2-1] Walls and Bridges\nWalls and Bridges is the fifth studio album by English musician John Lennon. 
#         [2-2] It was issued by Apple Records on 26 September 1974 in the United States and on 4 October in the United Kingdom. 
#         [2-3] Written, recorded and released during his 18-month separation from Yoko Ono, the album captured Lennon in the midst of his "Lost Weekend". 
#         [2-4] "Walls and Bridges" was an American "Billboard" number-one album and featured two hit singles, "Whatever Gets You thru the Night" and "#9 Dream". 
#         [2-5] The first of these was Lennon's first number-one hit in the United States as a solo artist, and his only chart-topping single in either the US or Britain during his lifetime.\n\n
#         """
#         """chunk 3
#         [3-1] Nobody Loves You (When You're Down and Out)\n"Nobody Loves You (When You're Down and Out)" is a song written by John Lennon released on his 1974 album "Walls and Bridges". 
#         [3-2] The song is included on the 1986 compilation "Menlove Ave.", the 1990 boxset "Lennon", the 1998 boxset "John Lennon Anthology", the 2005 two-disc compilation "", and the 2010 boxset "Gimme Some Truth".\n\n
#         [3-3] Give Peace a Chance\n"Give Peace a Chance" is an anti-war song written by John Lennon (credited to Lennon–McCartney), and performed with Yoko Ono in Montreal, Quebec, Canada. Released as a single in 1969 by the Plastic Ono Band on Apple Records (catalogue Apple 13 in the United Kingdom, Apple 1809 in the United States), it is the first solo single issued by Lennon, released when he was still a member of the Beatles, and became an anthem of the American anti-war movement during the 1970s. 
#         [3-4] It peaked at number 14 on the "Billboard" Hot 100 and number 2 on the British singles chart.\n
#         """
#     )

#     multihop_input = (
#         f"{multihop_docs}"
#         "\n\nQuestion: "
#         "Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?"
#         "\nThought: "
#     )

#     multihop_output = (
#         "[3-1] \"Nobody Loves You (When You're Down and Out)\" is a song written by John Lennon and released on the 1974 album \"Walls and Bridges.\" \n"
#         "[2-3] The album \"Walls and Bridges\" was written, recorded, and released during John Lennon's 18-month separation from Yoko Ono. \n"
#         "[2-2] The album \"Walls and Bridges\" was issued by Apple Records on 26 September 1974 in the United States and on 4 October in the United Kingdom. \n"
#         "Answer: Walls and Bridges."
# )

#     # -------------------------------------------------------
#     # cls 이름을 기준으로 사용할 프롬프트 선택
#     # -------------------------------------------------------
#     cls_lower = cls.lower()
#     if "multihop" in cls_lower:
#         system_prompt = multihop_system
#         one_shot_input = multihop_input
#         one_shot_output = multihop_output
#     else:
#         # 기본값 hotpot (cls에 hotpot이 포함되어 있거나, 그 외 케이스)
#         system_prompt = hotpot_system
#         one_shot_input = hotpot_input
#         one_shot_output = hotpot_output

#     # -------------------------------------------------------
#     # 최종 유저 프롬프트 (retrieved context + query)
#     # -------------------------------------------------------
#     final_user_prompt = (
#         f"{formatted_context}"
#         f"\n\nQuestion: {query}"
#         "\nThought: "
#     )

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": one_shot_input},
#         {"role": "assistant", "content": one_shot_output},
#         {"role": "user", "content": final_user_prompt},
#     ]

#     try:
#         prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])

#         final_response = await global_config["llm_model_func"](
#             prompt=prompt,
#             stream=getattr(param, "stream", False),
#         )

#         def extract_answer(response: str) -> str:
#             return response.split("Answer:", 1)[1].strip() if "Answer:" in response else response.strip()

#         final_response = extract_answer(final_response)

#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return "Final generation failed."


#     # =======================================================
#     # Step 6. SAVE FULL JSON LOG
#     # =======================================================
#     try:
#         save_dir = f"/workspace/AtomRAG/{cls}"
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f"{cls}_full_results.json")

#         step2_mapping_log = atomic_info  # 필요시 위에서 가져오세요
#         step3_topk_log = top_chunks     # 필요시 수정 가능

#         full_output = {
#             "query": query,
#             "dataset_cls": cls,
#             "subqueries": queries,
#             "step1_retrieval": step1_retrieval_log,
#             "step2_mapping": step2_mapping_log,
#             "step3_topk_chunks": step3_topk_log,
#             "final_context": formatted_context,
#             "messages_prompt_used": messages,
#             "final_answer": (
#                 final_response if isinstance(final_response, str)
#                 else str(final_response)
#             ),
#         }

#         if os.path.exists(save_path):
#             with open(save_path, "r", encoding="utf-8") as f:
#                 try:
#                     existing = json.load(f)
#                     if not isinstance(existing, list):
#                         existing = [existing]
#                 except Exception:
#                     existing = []
#         else:
#             existing = []

#         existing.append(full_output)

#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(existing, f, ensure_ascii=False, indent=4)

#         print(f"[INFO] JSON saved to {save_path}")

#     except Exception as e:
#         print(f"[ERROR] Saving JSON failed: {e}")

#     print(final_response)
#     return final_response



# ####### 일반 chunk + bge-large-reranker로 reranking 방식#########

# import os, json, copy, asyncio
# from collections import defaultdict
# from sentence_transformers import CrossEncoder

# reranker = CrossEncoder("BAAI/bge-reranker-large")

# async def ours_kg_query_experiment1(
#     query: str,
#     atomic_knowledge_graph_inst,
#     atomic_entities_vdb,
#     text_triple_db,
#     text_atomics_db,
#     text_chunks_db,
#     param,
#     global_config: dict,
#     hashing_kv=None,
# ) -> str:
#     # =======================================================
#     # Step 0. Query decomposition
#     # =======================================================
#     query_modes = {
#         "experiment1": query_process_1,
#         "experiment2": query_process_2,
#         "experiment3": query_process_3,
#         "experiment4": query_process_4,
#     }
#     queries = await query_modes.get(param.query_mode, lambda *a: [query])(
#         query, param, global_config, hashing_kv
#     )
#     if not queries:
#         queries = [query]

#     print(f"[INFO] Step 0 complete — {len(queries)} subqueries generated.")

#     # =======================================================
#     # Step 1. Atomic fact retrieval
#     # =======================================================
#     print(f"[INFO] Retrieving atomic facts for {len(queries)} subqueries...")

#     async def fetch_atomic_facts(sq):
#         try:
#             results = await atomic_entities_vdb.query(sq, top_k=10)
#             return sq, results
#         except Exception as e:
#             print(f"[ERROR] Retrieval failed for '{sq}': {e}")
#             return sq, []

#     subquery_results = dict(
#         await asyncio.gather(*(fetch_atomic_facts(sq) for sq in queries))
#     )

#     step1_retrieval_log = {
#         sq: [
#             {
#                 "atomic_id": r.get("id"),
#                 "entity_name": r.get("entity_name", ""),
#                 "score": r.get("distance", 0.0),
#             }
#             for r in results
#         ]
#         for sq, results in subquery_results.items()
#     }

#     # =======================================================
#     # Step 2. Atomic → Chunk mapping
#     # =======================================================

#     chunk_to_atomics = defaultdict(lambda: {"atomic_facts": []})
#     chunk_cache = {}

#     retrieved_items = []
#     for sq, results in subquery_results.items():
#         for r in results:
#             rid = r.get("id")
#             if rid:
#                 retrieved_items.append((sq, rid, r.get("distance", 0.0)))

#     if not retrieved_items:
#         return "No results."

#     unique_atomic_ids = list({rid for _, rid, _ in retrieved_items})
#     atomic_records = await text_atomics_db.get_by_ids(unique_atomic_ids)

#     atomic_info_map = {}
#     for aid, rec in zip(unique_atomic_ids, atomic_records or []):
#         if rec and isinstance(rec, dict):
#             atomic_info_map[aid] = {
#                 "content": rec.get("content", ""),
#                 "source_ids": rec.get("source_ids", []) or [],
#             }

#     for sq, aid, score in retrieved_items:
#         ainfo = atomic_info_map.get(aid)
#         if not ainfo:
#             continue

#         atomic_text = ainfo["content"]
#         src_chunks = ainfo["source_ids"]

#         for cid in src_chunks:
#             if cid not in chunk_cache:
#                 try:
#                     cdata = await text_chunks_db.get_by_ids([cid])
#                     chunk_cache[cid] = cdata[0].get("content", "") if cdata else ""
#                 except:
#                     chunk_cache[cid] = ""

#             chunk_to_atomics[cid]["chunk_id"] = cid
#             chunk_to_atomics[cid]["chunk_text"] = chunk_cache[cid]
#             chunk_to_atomics[cid]["atomic_facts"].append({
#                 "atomic_id": aid,
#                 "atomic_text": atomic_text,
#                 "sub_query": sq,
#                 "score": score,
#             })

#     for item in chunk_to_atomics.values():
#         seen = set()
#         uniq = []
#         for af in item["atomic_facts"]:
#             aid = af["atomic_id"]
#             if aid not in seen:
#                 seen.add(aid)
#                 uniq.append(af)
#         item["atomic_facts"] = uniq

#     chunk_results = list(chunk_to_atomics.values())
#     print(f"[INFO] Step 2 complete — {len(chunk_results)} chunks mapped.")

#     step2_mapping_log = []
#     for item in chunk_results:
#         txt = item.get("chunk_text", "")
#         step2_mapping_log.append({
#             "chunk_id": item["chunk_id"],
#             "chunk_text_preview": txt[:150] + ("..." if len(txt) > 150 else ""),
#             "num_atomics": len(item["atomic_facts"]),
#             "atomic_facts": item["atomic_facts"],
#         })

#     # =======================================================
#     # Step 3. BGE-reranker ranking
#     # =======================================================

#     print("[INFO] Step 3 — Reranking with BGE-reranker...")

#     pairs = [
#         [query, c["chunk_text"]] for c in chunk_results
#     ]
#     scores = reranker.predict(pairs)

#     for c, s in zip(chunk_results, scores):
#         c["rerank_score"] = float(s)

#     chunk_results_sorted = sorted(
#         chunk_results,
#         key=lambda x: x["rerank_score"],
#         reverse=True
#     )

#     top_k = min(10, len(chunk_results_sorted))
#     chunk_results_sorted = chunk_results_sorted[:top_k]

#     print(f"[INFO] Step 3 complete — Selected top-{top_k} chunks (reranker).")

#     step3_topk_log = [
#         {
#             "chunk_id": item["chunk_id"],
#             "rerank_score": item["rerank_score"],
#             "atomic_ids": [af["atomic_id"] for af in item["atomic_facts"]],
#         }
#         for item in chunk_results_sorted
#     ]

#     # =======================================================
#     # Step 4. Build context
#     # =======================================================

#     context_blocks = []
#     for idx, item in enumerate(chunk_results_sorted):
#         chunk_text = item.get("chunk_text", "").strip()
#         context_blocks.append(
#             f"### Retrieved Chunk {idx+1}\n{chunk_text}\n"
#         )

#     context = "\n\n".join(context_blocks)
#     print("[INFO] Step 4 complete — Context built.")

#     # =======================================================
#     # Step 5. Answer generation (same)
#     # =======================================================

#     cls = os.path.basename(global_config.get("working_dir", ""))
#     sys_prompt = PROMPTS["rag_response_origin"].format(context_data=context)

#     try:
#         final_response = await global_config["llm_model_func"](
#             query,
#             system_prompt=sys_prompt,
#             stream=param.stream,
#         )
#     except Exception as e:
#         return f"Final generation failed: {e}"

#     # =======================================================
#     # Step 6. Save FULL JSON LOG
#     # =======================================================

#     try:
#         save_dir = f"/workspace/AtomRAG/{cls}"
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f"{cls}_full_results.json")

#         full_output = {
#             "query": query,
#             "subqueries": queries,
#             "step1_retrieval": step1_retrieval_log,
#             "step2_mapping": step2_mapping_log,
#             "step3_topk_chunks": step3_topk_log,
#             "final_context": context,
#             "final_answer": final_response if isinstance(final_response, str) else str(final_response),
#             "system_prompt": sys_prompt,
#         }

#         if os.path.exists(save_path):
#             with open(save_path, "r", encoding="utf-8") as f:
#                 try:
#                     existing = json.load(f)
#                     if not isinstance(existing, list):
#                         existing = [existing]
#                 except:
#                     existing = []
#         else:
#             existing = []

#         existing.append(full_output)

#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(existing, f, ensure_ascii=False, indent=4)

#         print(f"[INFO] JSON saved to {save_path}")

#     except Exception as e:
#         print(f"[ERROR] Saving JSON failed: {e}")

#     print(final_response)
#     return final_response


# ####### 일반 chunk + sentence 단위로 monoT5를 이용해서 prunning하는 방식#########
# import torch
# from sentence_transformers import CrossEncoder
# import spacy
# nlp = spacy.blank("en")
# nlp.add_pipe("sentencizer")

# # =======================================================
# # monoT5 Reranker 로딩 (전역에서 한 번만)
# # =======================================================
# # 예시: 다국어 monoT5 reranker (필요에 따라 모델 이름 바꿔도 됨)
# MONOT5_MODEL_NAME = "castorini/monot5-base-msmarco"

# device = "cuda" if torch.cuda.is_available() else "cpu"
# monoT5_model = CrossEncoder(MONOT5_MODEL_NAME, device=device)

# def monoT5_score(query: str, sentence: str) -> float:
#     """
#     query와 sentence 쌍에 대해 monoT5 score를 계산.
#     0~1 사이 (혹은 모델이 주는 score)를 float으로 반환한다고 가정.
#     """
#     if not sentence.strip():
#         return 0.0
#     # CrossEncoder는 입력을 (query, passage) pair 리스트 형식으로 받음
#     score = monoT5_model.predict([(query, sentence)])[0]
#     return float(score)


# # =======================================================
# # 메인 함수
# # =======================================================
# async def ours_kg_query_experiment1(
#     query: str,
#     atomic_knowledge_graph_inst,
#     atomic_entities_vdb,
#     text_triple_db,
#     text_atomics_db,
#     text_chunks_db,
#     param,
#     global_config: dict,
#     hashing_kv=None,
# ) -> str:

#     # =======================================================
#     # Step 0. Query decomposition
#     # =======================================================
#     query_modes = {
#         "experiment1": query_process_1,
#         "experiment2": query_process_2,
#         "experiment3": query_process_3,
#         "experiment4": query_process_4,
#     }
#     queries = await query_modes.get(param.query_mode, lambda *a: [query])(
#         query, param, global_config, hashing_kv
#     )
#     if not queries:
#         queries = [query]

#     print(f"[INFO] Step 0 complete — {len(queries)} subqueries generated.")

#     # =======================================================
#     # Step 1. Atomic fact retrieval
#     # =======================================================
#     print(f"[INFO] Retrieving atomic facts for {len(queries)} subqueries...")

#     async def fetch_atomic_facts(sq):
#         try:
#             results = await atomic_entities_vdb.query(sq, top_k=10)
#             return sq, results
#         except Exception as e:
#             print(f"[ERROR] Retrieval failed for '{sq}': {e}")
#             return sq, []

#     subquery_results = dict(
#         await asyncio.gather(*(fetch_atomic_facts(sq) for sq in queries))
#     )

#     step1_retrieval_log = {
#         sq: [
#             {
#                 "atomic_id": r.get("id"),
#                 "entity_name": r.get("entity_name", ""),
#                 "score": r.get("distance", 0.0),
#             }
#             for r in results
#         ]
#         for sq, results in subquery_results.items()
#     }

#     # =======================================================
#     # Step 2. Atomic → Chunk mapping
#     # =======================================================

#     chunk_to_atomics = defaultdict(lambda: {"atomic_facts": []})
#     chunk_cache = {}

#     retrieved_items = []
#     for sq, results in subquery_results.items():
#         for r in results:
#             rid = r.get("id")
#             if rid:
#                 retrieved_items.append((sq, rid, r.get("distance", 0.0)))

#     if not retrieved_items:
#         return "No results."

#     unique_atomic_ids = list({rid for _, rid, _ in retrieved_items})
#     atomic_records = await text_atomics_db.get_by_ids(unique_atomic_ids)

#     atomic_info_map = {}
#     for aid, rec in zip(unique_atomic_ids, atomic_records or []):
#         if rec and isinstance(rec, dict):
#             atomic_info_map[aid] = {
#                 "content": rec.get("content", ""),
#                 "source_ids": rec.get("source_ids", []) or [],
#             }

#     for sq, aid, score in retrieved_items:
#         ainfo = atomic_info_map.get(aid)
#         if not ainfo:
#             continue

#         atomic_text = ainfo["content"]
#         src_chunks = ainfo["source_ids"]

#         for cid in src_chunks:
#             if cid not in chunk_cache:
#                 try:
#                     cdata = await text_chunks_db.get_by_ids([cid])
#                     chunk_cache[cid] = cdata[0].get("content", "") if cdata else ""
#                 except Exception:
#                     chunk_cache[cid] = ""

#             chunk_to_atomics[cid]["chunk_id"] = cid
#             chunk_to_atomics[cid]["chunk_text"] = chunk_cache[cid]
#             chunk_to_atomics[cid]["atomic_facts"].append({
#                 "atomic_id": aid,
#                 "atomic_text": atomic_text,
#                 "sub_query": sq,
#                 "score": score,
#             })

#     # atomic_facts 중복 제거
#     for item in chunk_to_atomics.values():
#         seen = set()
#         uniq = []
#         for af in item["atomic_facts"]:
#             aid = af["atomic_id"]
#             if aid not in seen:
#                 seen.add(aid)
#                 uniq.append(af)
#         item["atomic_facts"] = uniq

#     chunk_results = list(chunk_to_atomics.values())
#     print(f"[INFO] Step 2 complete — {len(chunk_results)} chunks mapped.")

#     step2_mapping_log = []
#     for item in chunk_results:
#         txt = item.get("chunk_text", "")
#         step2_mapping_log.append({
#             "chunk_id": item["chunk_id"],
#             "chunk_text_preview": txt[:150] + ("..." if len(txt) > 150 else ""),
#             "num_atomics": len(item["atomic_facts"]),
#             "atomic_facts": item["atomic_facts"],
#         })

#     # =======================================================
#     # Step 3. Rank chunks (ONLY max_atomic_score)
#     # =======================================================

#     for item in chunk_results:
#         item["max_atomic_score"] = max(
#             [af["score"] for af in item["atomic_facts"]],
#             default=0.0
#         )

#     chunk_results_sorted = sorted(
#         chunk_results,
#         key=lambda x: x["max_atomic_score"],
#         reverse=True,
#     )

#     top_k = min(10, len(chunk_results_sorted))
#     chunk_results_sorted = chunk_results_sorted[:top_k]

#     print(f"[INFO] Step 3 complete — Selected top-{top_k} chunks (by max_atomic_score).")

#     step3_topk_log = [
#         {
#             "chunk_id": item["chunk_id"],
#             "max_atomic_score": item["max_atomic_score"],
#             "atomic_ids": [af["atomic_id"] for af in item["atomic_facts"]],
#         }
#         for item in chunk_results_sorted
#     ]

#     # =======================================================
#     # Step 3.5. Sentence-level pruning using monoT5 + spaCy
#     # DSLR official pipeline:
#     #   1. score all sentences
#     #   2. sort by relevance
#     #   3. threshold filtering
#     #   4. restore original order
#     # =======================================================

#     THRESHOLD = 0.6
#     pruned_chunk_results = []

#     for item in chunk_results_sorted:
#         chunk_text = item.get("chunk_text", "").strip()

#         # sentence segmentation
#         doc = nlp(chunk_text)
#         sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

#         # (1) scoring
#         scored_sentences = []
#         for sent in sentences:
#             try:
#                 score = monoT5_score(query, sent)
#             except:
#                 score = 0.0
#             scored_sentences.append((sent, score))

#         # (2) sort by score (descending)
#         scored_sorted = sorted(scored_sentences, key=lambda x: x[1], reverse=True)

#         # (3) threshold filtering
#         filtered_sorted = [s for s, sc in scored_sorted if sc >= THRESHOLD]

#         # (4) restore original order
#         filtered_original_order = [s for s in sentences if s in filtered_sorted]

#         # nothing left? keep at least 1
#         if not filtered_original_order and sentences:
#             filtered_original_order = [sentences[0]]

#         pruned_text = " ".join(filtered_original_order)

#         item["pruned_chunk_text"] = pruned_text
#         pruned_chunk_results.append(item)

#     print("[INFO] Step 3.5 complete — DSLR-style pruning + reorder applied.")



#     # =======================================================
#     # Step 4. Build context (using pruned chunks)
#     # =======================================================

#     context_blocks = []
#     for idx, item in enumerate(pruned_chunk_results):
#         chunk_text = item.get("pruned_chunk_text", "").strip()
#         context_blocks.append(
#             f"### Retrieved Chunk {idx+1}\n{chunk_text}\n"
#         )

#     context = "\n\n".join(context_blocks)
#     print("[INFO] Step 4 complete — Context built (sentence-pruned).")

#     # =======================================================
#     # Step 5. Final answer generation
#     # =======================================================

#     cls = os.path.basename(global_config.get("working_dir", ""))
#     sys_prompt_template = PROMPTS.get("rag_response_origin")
#     sys_prompt = sys_prompt_template.format(context_data=context)

#     try:
#         final_response = await global_config["llm_model_func"](
#             query,
#             system_prompt=sys_prompt,
#             stream=getattr(param, "stream", False),
#         )
#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return "Final generation failed."

#     # =======================================================
#     # Step 6. Save FULL JSON LOG
#     # =======================================================

#     try:
#         save_dir = f"/workspace/AtomRAG/{cls}"
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f"{cls}_full_results.json")

#         full_output = {
#             "query": query,
#             "subqueries": queries,
#             "step1_retrieval": step1_retrieval_log,
#             "step2_mapping": step2_mapping_log,
#             "step3_topk_chunks": step3_topk_log,
#             "final_context": context,
#             "final_answer": (
#                 final_response if isinstance(final_response, str) else str(final_response)
#             ),
#             "system_prompt": sys_prompt,
#         }

#         if os.path.exists(save_path):
#             with open(save_path, "r", encoding="utf-8") as f:
#                 try:
#                     existing = json.load(f)
#                     if not isinstance(existing, list):
#                         existing = [existing]
#                 except Exception:
#                     existing = []
#         else:
#             existing = []

#         existing.append(full_output)

#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(existing, f, ensure_ascii=False, indent=4)

#         print(f"[INFO] JSON saved to {save_path}")

#     except Exception as e:
#         print(f"[ERROR] Saving JSON failed: {e}")

#     print(final_response)
#     return final_response




# ######### 중요 chunk는 원문 그대로 덜 중요한 chunk는 summary #########
# async def ours_kg_query_experiment1(
#     query: str,
#     atomic_knowledge_graph_inst,
#     atomic_entities_vdb,
#     text_triple_db,
#     text_atomics_db,
#     text_chunks_db,
#     param,
#     global_config: dict,
#     hashing_kv=None,
# ) -> str:

#     import os, json, asyncio
#     from collections import defaultdict

#     # =====================================================================
#     # Summarization helper
#     # =====================================================================
#     async def summarize_chunk(text: str, level: int):
#         """
#         level=1 → HIGH (원문 그대로)
#         level=2 → MID (부분 요약)
#         level=3 → LOW (강한 요약)
#         """

#         # 🔹 HIGH: summarization 없이 원문 그대로 사용
#         if level == 1:
#             return text

#         llm = global_config["llm_model_func"]

#         # 🔹 MID summarization
#         if level == 2:
#             prompt = f"""
# You are an expert summarizer. Summarize the following passage in **2~3 sentences**, 
# keeping all important facts and avoiding hallucination.

# Passage:
# {text}
# """
#             try:
#                 result = await llm(prompt, system_prompt="Summarization Assistant", stream=False)
#                 return result
#             except:
#                 return text  # fallback

#         # 🔹 LOW summarization — 핵심 fact만 bullet로 정리
#         if level == 3:
#             prompt = f"""
# Extract the **3 most important factual statements** from the passage. 
# Output as concise bullet points.

# Passage:
# {text}
# """
#             try:
#                 result = await llm(prompt, system_prompt="Extractive Summarizer", stream=False)
#                 return result
#             except:
#                 return text

#         return text

#     # =====================================================================
#     # Step 0. Query decomposition
#     # =====================================================================
#     query_modes = {
#         "experiment1": query_process_1,
#         "experiment2": query_process_2,
#         "experiment3": query_process_3,
#         "experiment4": query_process_4,
#     }

#     queries = await query_modes.get(param.query_mode, lambda *a: [query])(
#         query, param, global_config, hashing_kv
#     )
#     if not queries:
#         queries = [query]

#     print(f"[INFO] Step 0 complete — {len(queries)} subqueries generated.")

#     # =====================================================================
#     # Step 1. Atomic fact retrieval
#     # =====================================================================

#     async def fetch_atomic_facts(sq):
#         try:
#             res = await atomic_entities_vdb.query(sq, top_k=10)
#             return sq, res
#         except Exception as e:
#             print(f"[ERROR] Retrieval failed for '{sq}': {e}")
#             return sq, []

#     subquery_results = dict(await asyncio.gather(*(fetch_atomic_facts(sq) for sq in queries)))

#     # logging
#     step1_retrieval_log = {
#         sq: [
#             {
#                 "atomic_id": r.get("id"),
#                 "entity_name": r.get("entity_name", ""),
#                 "score": r.get("distance", 0.0),
#             }
#             for r in results
#         ]
#         for sq, results in subquery_results.items()
#     }

#     # =====================================================================
#     # Step 2. Atomic → Chunk mapping
#     # =====================================================================
#     chunk_to_atomics = defaultdict(lambda: {"atomic_facts": []})
#     chunk_cache = {}

#     retrieved_items = []
#     for sq, results in subquery_results.items():
#         for r in results:
#             rid = r.get("id")
#             if rid:
#                 retrieved_items.append((sq, rid, r.get("distance", 0.0)))

#     if not retrieved_items:
#         return "No results."

#     unique_atomic_ids = list({rid for _, rid, _ in retrieved_items})
#     atomic_records = await text_atomics_db.get_by_ids(unique_atomic_ids)

#     atomic_info_map = {}
#     for aid, rec in zip(unique_atomic_ids, atomic_records or []):
#         if rec and isinstance(rec, dict):
#             atomic_info_map[aid] = {
#                 "content": rec.get("content", ""),
#                 "source_ids": rec.get("source_ids", []) or [],
#             }

#     # mapping
#     for sq, aid, score in retrieved_items:
#         ainfo = atomic_info_map.get(aid)
#         if not ainfo:
#             continue

#         atomic_text = ainfo["content"]
#         src_chunks = ainfo["source_ids"]

#         for cid in src_chunks:
#             # chunk caching
#             if cid not in chunk_cache:
#                 try:
#                     cdata = await text_chunks_db.get_by_ids([cid])
#                     chunk_cache[cid] = cdata[0].get("content", "") if cdata else ""
#                 except:
#                     chunk_cache[cid] = ""

#             chunk_to_atomics[cid]["chunk_id"] = cid
#             chunk_to_atomics[cid]["chunk_text"] = chunk_cache[cid]
#             chunk_to_atomics[cid]["atomic_facts"].append({
#                 "atomic_id": aid,
#                 "atomic_text": atomic_text,
#                 "sub_query": sq,
#                 "score": score,
#             })

#     # remove dup
#     for item in chunk_to_atomics.values():
#         seen = set()
#         uniq = []
#         for af in item["atomic_facts"]:
#             aid = af["atomic_id"]
#             if aid not in seen:
#                 seen.add(aid)
#                 uniq.append(af)
#         item["atomic_facts"] = uniq

#     chunk_results = list(chunk_to_atomics.values())
#     print(f"[INFO] Step 2 complete — {len(chunk_results)} chunks mapped.")

#     # mapping log
#     step2_mapping_log = []
#     for item in chunk_results:
#         txt = item.get("chunk_text", "")
#         step2_mapping_log.append({
#             "chunk_id": item["chunk_id"],
#             "chunk_text_preview": txt[:150] + ("..." if len(txt) > 150 else ""),
#             "num_atomics": len(item["atomic_facts"]),
#             "atomic_facts": item["atomic_facts"],
#         })

#     # =====================================================================
#     # Step 3. Combined scoring
#     # =====================================================================

#     for item in chunk_results:
#         item["num_atomics"] = len(item["atomic_facts"])
#         item["max_atomic_score"] = max([af["score"] for af in item["atomic_facts"]], default=0.0)

#     # normalization
#     def norm(v, mn, mx):
#         if mx - mn < 1e-9:
#             return 0.0
#         return (v - mn) / (mx - mn)

#     nums = [x["num_atomics"] for x in chunk_results]
#     scores = [x["max_atomic_score"] for x in chunk_results]

#     min_n, max_n = min(nums), max(nums)
#     min_s, max_s = min(scores), max(scores)

#     w1 = 0.6
#     w2 = 0.4

#     for item in chunk_results:
#         item["chunk_score"] = (
#             w1 * norm(item["num_atomics"], min_n, max_n)
#             + w2 * norm(item["max_atomic_score"], min_s, max_s)
#         )

#     top_k = min(10, len(chunk_results))
#     chunk_results_sorted = sorted(chunk_results, key=lambda x: x["chunk_score"], reverse=True)[:top_k]

#     print(f"[INFO] Step 3 complete — Selected top-{top_k} chunks.")

#     step3_topk_log = [
#         {
#             "chunk_id": item["chunk_id"],
#             "chunk_score": item["chunk_score"],
#             "num_atomics": item["num_atomics"],
#             "max_atomic_score": item["max_atomic_score"],
#         }
#         for item in chunk_results_sorted
#     ]

#     # =====================================================================
#     # Step 4. Build context WITH summarization by relevance
#     # =====================================================================

#     def relevance_level(rank):
#         if rank <= 3:
#             return 1  # HIGH
#         elif rank <= 6:
#             return 2  # MID
#         else:
#             return 3  # LOW

#     context_blocks = []
#     for idx, item in enumerate(chunk_results_sorted):
#         level = relevance_level(idx + 1)
#         original = item["chunk_text"]

#         summarized = await summarize_chunk(original, level)

#         label = (
#             "[HIGHLY RELEVANT]" if level == 1 else
#             "[MODERATELY RELEVANT]" if level == 2 else
#             "[LIGHTLY RELEVANT]"
#         )

#         context_blocks.append(
#             f"### Retrieved Chunk {idx+1} {label}\n{summarized}\n"
#         )

#     context = "\n\n".join(context_blocks)
#     print("[INFO] Step 4 complete — Context with summarization applied.")

#     # =====================================================================
#     # Step 5. Final answer generation
#     # =====================================================================
#     cls = os.path.basename(global_config.get("working_dir", ""))
#     sys_prompt_template = PROMPTS.get("rag_response_origin")
#     sys_prompt = sys_prompt_template.format(context_data=context)

#     try:
#         final_answer = await global_config["llm_model_func"](
#             query,
#             system_prompt=sys_prompt,
#             stream=param.stream,
#         )
#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         final_answer = "Final generation failed."

#     # =====================================================================
#     # Step 6. Save full JSON log
#     # =====================================================================
#     try:
#         save_dir = f"/workspace/AtomRAG/{cls}"
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f"{cls}_full_results.json")

#         full_output = {
#             "query": query,
#             "subqueries": queries,
#             "step1_retrieval": step1_retrieval_log,
#             "step2_mapping": step2_mapping_log,
#             "step3_topk_chunks": step3_topk_log,
#             "final_context": context,
#             "final_answer": final_answer,
#             "system_prompt": sys_prompt,
#         }

#         if os.path.exists(save_path):
#             with open(save_path, "r", encoding="utf-8") as f:
#                 try:
#                     existing = json.load(f)
#                     if not isinstance(existing, list):
#                         existing = [existing]
#                 except:
#                     existing = []
#         else:
#             existing = []

#         existing.append(full_output)

#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(existing, f, ensure_ascii=False, indent=4)

#         print(f"[INFO] JSON saved to {save_path}")

#     except Exception as e:
#         print(f"[ERROR] Saving JSON failed: {e}")

#     print(final_answer)
#     return final_answer


# ###### query distribution(직렬/병렬) + chunk filtering + (개선된)summary ######
# async def ours_kg_query_experiment1(
#     query,
#     atomic_knowledge_graph_inst: BaseGraphStorage,
#     atomic_entities_vdb: BaseVectorStorage,
#     text_atomics_db: BaseKVStorage,
#     text_chunks_db: BaseKVStorage,
#     param: QueryParam,
#     global_config: dict,
#     hashing_kv: BaseKVStorage = None,
# ) -> str:
#     # Step 0. Query decomposition
#     if param.query_mode == "experiment1":
#         queries = await query_process_1(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment2":
#         queries = await query_process_2(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment3":
#         queries = await query_process_3(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment4":
#         queries = await query_process_4(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment5":
#         reasoning_type, queries = await query_process_5(query, param, global_config, hashing_kv)
#     else:
#         queries = [query]

#     # Step 1. subquery별 atomic fact 검색
#     subquery_results = {}
#     try:
#         # (1) 병렬형 또는 reasoning_type이 없는 경우 → 기존 방식 유지
#         if not reasoning_type or reasoning_type == "parallel":
#             for sq in queries:
#                 res = await atomic_entities_vdb.query(sq, top_k=10)
#                 subquery_results[sq] = res

#         # (2) Sequential reasoning인 경우
#         elif reasoning_type == "sequential":
#             prev_atomic_facts = []  # 이전 단계에서 추출된 atomic fact 텍스트 저장

#             for idx, sq in enumerate(queries):
#                 # [Answer from sub-query N]이 없는 첫 단계
#                 if "[Answer from sub-query" not in sq:
#                     res = await atomic_entities_vdb.query(sq, top_k=5)
#                     subquery_results[sq] = res

#                     # 다음 단계에 넣을 top-3 atomic fact 텍스트 수집
#                     prev_atomic_facts = []
#                     for r in res:
#                         try:
#                             atom_id = r.get("id")
#                             atomic_data = await text_atomics_db.get_by_ids([atom_id])
#                             atomic_text = (
#                                 atomic_data[0].get("content", "")
#                                 if atomic_data and isinstance(atomic_data, list)
#                                 else ""
#                             )
#                             if atomic_text:
#                                 prev_atomic_facts.append(atomic_text.strip())
#                         except Exception:
#                             continue

#                     print(f"[SEQ] Step {idx+1}: Retrieved {len(prev_atomic_facts)} atomic facts")
#                     continue

#                 # (이후 단계) [Answer from sub-query N] 포함된 sub-query 처리
#                 if not prev_atomic_facts:
#                     print(f"[SEQ] Step {idx+1}: No previous facts found, skipping dependent expansion.")
#                     continue

#                 for fact in prev_atomic_facts:
#                     # 이전 단계의 개별 atomic fact를 대입하여 새로운 query 생성
#                     filled_query = sq.replace(f"[Answer from sub-query {idx}]", fact)
#                     filled_query = filled_query.replace("[Answer from sub-query 1]", fact)  # 안전 처리

#                     res = await atomic_entities_vdb.query(filled_query, top_k=5)
#                     subquery_results[filled_query] = res

#                     print(f"[SEQ] Step {idx+1}: Query='{filled_query}' → Retrieved {len(res)} results")

#                     # 이 단계에서도 다음 단계를 위한 atomic fact 후보를 새로 수집 가능
#                     new_facts = []
#                     for r in res:
#                         try:
#                             atom_id = r.get("id")
#                             atomic_data = await text_atomics_db.get_by_ids([atom_id])
#                             atomic_text = (
#                                 atomic_data[0].get("content", "")
#                                 if atomic_data and isinstance(atomic_data, list)
#                                 else ""
#                             )
#                             if atomic_text:
#                                 new_facts.append(atomic_text.strip())
#                         except Exception:
#                             continue

#                     # 다음 step에서 사용할 fact 리스트를 업데이트 (중복 제거)
#                     prev_atomic_facts = list(set(new_facts))

#         else:
#             print(f"[WARN] Unknown reasoning type '{reasoning_type}', using default parallel mode.")
#             for sq in queries:
#                 res = await atomic_entities_vdb.query(sq, top_k=10)
#                 subquery_results[sq] = res

#     except Exception as e:
#         print(f"[ERROR] atomic_entities_vdb query failed: {e}")
#         return []



#     # Step 2. chunk 단위로 묶기
#     G_atomic = getattr(atomic_knowledge_graph_inst, "_graph", None)
#     chunk_to_atomics = {}

#     if G_atomic:
#         for sq, res_list in subquery_results.items():
#             for r in res_list:
#                 atom_id = r.get("id")
#                 entity_name = r.get("entity_name")
#                 score = r.get("distance", 0.0)

#                 if not entity_name or entity_name not in G_atomic:
#                     continue

#                 try:
#                     node_data = await atomic_knowledge_graph_inst.get_node(entity_name)
#                 except Exception:
#                     node_data = None
#                 if not isinstance(node_data, dict):
#                     continue

#                 src_id = node_data.get("source_id")  # 연결된 chunk id들
#                 atomic_id = node_data.get("atomic_id") or atom_id
#                 if not src_id or not atomic_id:
#                     continue

#                 # atomic fact text 가져오기
#                 try:
#                     atomic_data = await text_atomics_db.get_by_ids([atomic_id])
#                     atomic_text = atomic_data[0].get("content", "") if atomic_data else ""
#                 except Exception:
#                     atomic_text = ""

#                 # 여러 chunk에 연결될 수 있음
#                 chunk_ids = [cid.strip() for cid in src_id.split("<SEP>") if cid.strip()]
#                 for cid in chunk_ids:
#                     try:
#                         chunk_data = await text_chunks_db.get_by_ids([cid])
#                         chunk_text = chunk_data[0].get("content", "") if chunk_data else ""
#                     except Exception:
#                         chunk_text = ""

#                     if cid not in chunk_to_atomics:
#                         chunk_to_atomics[cid] = {
#                             "chunk_id": cid,
#                             "chunk_text": chunk_text,
#                             "atomic_facts": []
#                         }

#                     chunk_to_atomics[cid]["atomic_facts"].append({
#                         "atomic_id": atomic_id,
#                         "atomic_text": atomic_text,
#                         "sub_query": sq,
#                         "score": score
#                     })
#         for cid, item in chunk_to_atomics.items():
#             seen_ids = set()
#             unique_facts = []
#             for af in item["atomic_facts"]:
#                 aid = af.get("atomic_id")
#                 if aid not in seen_ids:
#                     seen_ids.add(aid)
#                     unique_facts.append(af)
#             item["atomic_facts"] = unique_facts

#     # Step 3. 결과 리스트 정리
#     results = list(chunk_to_atomics.values())

#     # Step 3-1. Chunk 대표 score 기반 Top-K Filtering
#     top_k_chunks = 10  # 상위 K개 chunk만 선택

#     scored_results = []
#     for item in results:
#         atomic_facts = item.get("atomic_facts", [])
#         if not atomic_facts:
#             continue

#         # atomic fact들의 cosine similarity를 그대로 사용
#         similarities = []
#         for af in atomic_facts:
#             d = af.get("score", None)
#             if isinstance(d, (int, float)):
#                 sim = d  # 이미 cosine similarity 값임 → 변환 불필요
#                 similarities.append(sim)

#         # chunk의 대표 score = 해당 chunk 내 가장 높은 similarity
#         chunk_score = max(similarities) if similarities else 0.0

#         scored_results.append({
#             **item,
#             "chunk_score": chunk_score
#         })

#     # 점수 기준으로 정렬 후 상위 K개만 선택
#     scored_results.sort(key=lambda x: x["chunk_score"], reverse=True)
#     filtered_results = scored_results[:top_k_chunks]

#     print(f"[INFO] Filtered top-{top_k_chunks} chunks out of {len(results)} total "
#         f"based on max cosine similarity per chunk.")
#     results = filtered_results

#     # Step 4. JSON 저장 (중간 결과)
#     cls = os.path.basename(global_config["working_dir"])
#     output_path = f"/workspace/AtomRAG/{cls}/query_atomic_chunk_mapping.json"

#     output_entry = {
#         "query": query,
#         "results": results
#     }

#     try:
#         if os.path.exists(output_path):
#             with open(output_path, "r", encoding="utf-8") as f:
#                 old_data = json.load(f)
#                 if not isinstance(old_data, list):
#                     old_data = [old_data]
#         else:
#             old_data = []
#         old_data.append(output_entry)

#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(old_data, f, ensure_ascii=False, indent=2)
#         print(f"[INFO] Results for query='{query}' saved to {output_path}")
#     except Exception as e:
#         print(f"[ERROR] Failed to save results: {e}")

#     # Step 5. Chunk-level Summarization (Parallel)
#     use_model_func = global_config["llm_model_func"]
#     results = output_entry["results"]

#     if not results:
#         print("[INFO] No chunk results found. Returning empty response.")
#         return PROMPTS.get("fail_response", "No results found.")

#     print(f"[INFO] Generating {len(results)} chunk-level summaries in parallel...")

#     async def summarize_chunk(i, item):
#         chunk_text = item.get("chunk_text", "")
#         atomic_facts = item.get("atomic_facts", [])
#         atomic_list_str = "\n".join(
#             [f"- {af['atomic_text']}" for af in atomic_facts if af.get("atomic_text")]
#         )
        
#         summary_prompt_template = PROMPTS["chunk_rewriting"]
        
#         prompt = summary_prompt_template.format(
#             main_query=query,
#             chunk_text=chunk_text,
#             atomic_facts=atomic_list_str
#         )

#         try:
#             response = await use_model_func(
#                 prompt,
#                 stream=param.stream,
#             )
#             if isinstance(response, str):
#                 response = response.strip()
#             return {
#                 "chunk_id": item.get("chunk_id", f"chunk_{i}"),
#                 "summary_prompt": prompt,
#                 "summary": response
#             }
#         except Exception as e:
#             print(f"[ERROR] Summary generation failed for chunk {i}: {e}")
#             return None

#     # 병렬 실행
#     tasks = [summarize_chunk(i, item) for i, item in enumerate(results)]
#     summaries_raw = await asyncio.gather(*tasks)
#     summaries = [s for s in summaries_raw if s is not None]

#     if not summaries:
#         print("[INFO] No summaries generated. Returning fail response.")
#         return PROMPTS.get("fail_response", "Summary generation failed.")

#     # Step 6. Summaries concat → Final Answer Prompt (atomic facts removed)
#     context_blocks = []
#     for i, summ in enumerate(summaries):
#         summary_text = summ.get("summary", "").strip()
        
#         if not summary_text or summary_text.lower() == "empty response":
#             continue
        
#         if "[Relation]:" in summary_text:
#             first_line = summary_text.splitlines()[0].strip().upper()
#             if "UNRELATED" in first_line:
#                 continue

#         block_text = f"[Chunk {i+1}]\n{summary_text}\n"
#         context_blocks.append(block_text)

#     # context가 완전히 비어있으면 기본 문자열로 대체
#     if not context_blocks:
#         print("[INFO] All summaries are empty. Using default 'no information' context.")
#         context = "no information"
#     else:
#         context = "\n\n".join(context_blocks)


#     # Step 7. Final answer generation
#     if cls == "hotpot":
#         sys_prompt_temp = PROMPTS["rag_response_hotpot"]
#     elif cls == "multihoprag":
#         sys_prompt_temp = PROMPTS["rag_response_multihoprag"]
#     else:
#         sys_prompt_temp = PROMPTS["ours_rag_response"]

#     sys_prompt = sys_prompt_temp.format(context_data=context)

#     print(f"[INFO] Generating final answer using summarized context...")

#     try:
#         final_response = await use_model_func(query, system_prompt=sys_prompt, stream=param.stream)
#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return PROMPTS.get("fail_response", "Final generation failed.")

#     print("final prompt length:", len(sys_prompt))
#     print(final_response)

#     # Step 8. 결과 저장
#     final_output_path = f"/workspace/AtomRAG/{cls}/query_final_answer.json"
#     final_entry = {
#         "query": query,
#         "summaries": summaries,
#         "final_prompt": sys_prompt,
#         "final_answer": final_response,
#     }

#     try:
#         if os.path.exists(final_output_path):
#             with open(final_output_path, "r", encoding="utf-8") as f:
#                 old_data = json.load(f)
#                 if not isinstance(old_data, list):
#                     old_data = [old_data]
#         else:
#             old_data = []
#         old_data.append(final_entry)

#         with open(final_output_path, "w", encoding="utf-8") as f:
#             json.dump(old_data, f, ensure_ascii=False, indent=2)
#         print(f"[INFO] Final answer for query='{query}' saved to {final_output_path}")
#     except Exception as e:
#         print(f"[ERROR] Failed to save final answer: {e}")

#     return final_response

######### varaible-grounded reasoning query decomposition #########
# async def ours_kg_query_experiment1(
#     query,
#     atomic_knowledge_graph_inst,
#     atomic_entities_vdb,
#     text_atomics_db,
#     text_chunks_db,
#     param,
#     global_config,
#     hashing_kv=None,
# ) -> str:
#     """
#     강건 버전 ours_kg_query_experiment1
#     모든 병렬/직렬 reasoning 케이스 (Case 1~9) 완전 커버
#     """
#     # -------------------------------
#     # Step 0. Query decomposition
#     # -------------------------------
#     reasoning_type = None
#     if param.query_mode == "experiment1":
#         queries = await query_process_1(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment2":
#         queries = await query_process_2(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment3":
#         queries = await query_process_3(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment4":
#         queries = await query_process_4(query, param, global_config, hashing_kv)
#     elif param.query_mode == "experiment5":
#         reasoning_type, queries = await query_process_5(query, param, global_config, hashing_kv)
#     else:
#         queries = [query]

#     use_model_func = global_config["llm_model_func"]

#     subquery_results = {}
#     final_atomic_facts = []

#     # -------------------------------
#     # Step 1. Atomic fact retrieval
#     # -------------------------------
#     try:
#         if not reasoning_type or reasoning_type == "parallel":
#             # 병렬형 → 기본 방식
#             for sq in queries:
#                 res = await atomic_entities_vdb.query(sq, top_k=10)
#                 subquery_results[sq] = res

#         # 직렬형 reasoning
#         elif reasoning_type == "sequential":
#             answer_map = {}  # {index: [answers]}
#             final_query_indices = set()

#             for idx, sq in enumerate(queries, start=1):
#                 sq = sq.replace("\\[", "[")  # escape 문자 보정
#                 print(f"\n[SEQ] Step {idx}: {sq}")

#                 # placeholder 탐색
#                 refs = re.findall(r"\[Answer from sub-query (\d+)\]", sq)

#                 # Case A: 독립적 sub-query (첫 단계)
#                 if not refs:
#                     res = await atomic_entities_vdb.query(sq, top_k=5)
#                     subquery_results[sq] = res

#                     atomic_texts = []
#                     for r in res:
#                         atom_id = r.get("id")
#                         data = await text_atomics_db.get_by_ids([atom_id])
#                         if data and isinstance(data, list):
#                             txt = data[0].get("content", "").strip()
#                             if txt:
#                                 atomic_texts.append(txt)

#                     llm_input = (
#                         f"---Instruction---\n"
#                         f"Answer concisely in one word or short phrase.\n\n"
#                         f"---Sub-query---\n{sq}\n\n"
#                         f"---Relevant Atomic Facts---\n" + "\n".join(f"- {a}" for a in atomic_texts)
#                     )
#                     try:
#                         ans = await use_model_func(llm_input, stream=param.stream)
#                         if isinstance(ans, str):
#                             canonical = ans.strip().lower()
#                             answer_map[idx] = [canonical]
#                     except Exception as e:
#                         print(f"[ERROR] LLM failed on step {idx}: {e}")
#                     continue

#                 # Case B: 의존적 sub-query
#                 ref_indices = [int(r) for r in refs]
#                 missing_refs = [r for r in ref_indices if r not in answer_map]

#                 if missing_refs:
#                     print(f"[WARN] Missing reference answers for {missing_refs}, skipping those.")
#                     ref_indices = [r for r in ref_indices if r in answer_map]
#                     if not ref_indices:
#                         continue

#                 # 각 placeholder 조합 확장 (Cartesian product)
#                 ref_answers = [answer_map[r] for r in ref_indices]
#                 combinations = list(itertools.product(*ref_answers))

#                 generated_answers = set()
#                 for combo in combinations:
#                     filled_sq = sq
#                     for r_idx, ans_val in zip(ref_indices, combo):
#                         filled_sq = filled_sq.replace(f"[Answer from sub-query {r_idx}]", ans_val)
#                     filled_sq = filled_sq.replace("\\[", "[")  # 보정
#                     print(f"[SEQ] Filled query: {filled_sq}")

#                     res = await atomic_entities_vdb.query(filled_sq, top_k=5)
#                     subquery_results[filled_sq] = res

#                     # 마지막 단계 판단
#                     if idx == len(queries):
#                         final_query_indices.add(idx)
#                         final_atomic_facts.extend(res)
#                         continue

#                     # 중간 reasoning 단계 처리
#                     atomic_texts = []
#                     for r in res:
#                         atom_id = r.get("id")
#                         data = await text_atomics_db.get_by_ids([atom_id])
#                         if data and isinstance(data, list):
#                             txt = data[0].get("content", "").strip()
#                             if txt:
#                                 atomic_texts.append(txt)

#                     llm_input = (
#                         f"---Instruction---\n"
#                         f"Answer concisely in one word or short phrase.\n\n"
#                         f"---Sub-query---\n{filled_sq}\n\n"
#                         f"---Relevant Atomic Facts---\n" + "\n".join(f"- {a}" for a in atomic_texts)
#                     )
#                     try:
#                         ans_text = await use_model_func(llm_input, stream=param.stream)
#                         if isinstance(ans_text, str):
#                             canonical = ans_text.strip().lower()
#                             generated_answers.add(canonical)
#                     except Exception as e:
#                         print(f"[ERROR] LLM reasoning failed on step {idx}: {e}")

#                 # 이번 단계의 모든 생성 답변 저장
#                 if generated_answers:
#                     answer_map[idx] = list(generated_answers)

#             # 마지막 단계 식별이 안 된 경우 (Case 9: multi-final)
#             if not final_atomic_facts and answer_map:
#                 max_idx = max(answer_map.keys())
#                 final_query_indices.add(max_idx)
#                 print(f"[INFO] Inferred final step index: {max_idx}")

#         else:
#             print(f"[WARN] Unknown reasoning type '{reasoning_type}', using default parallel mode.")
#             for sq in queries:
#                 res = await atomic_entities_vdb.query(sq, top_k=10)
#                 subquery_results[sq] = res

#     except Exception as e:
#         print(f"[ERROR] atomic_entities_vdb query failed: {e}")
#         return "Retrieval failed."

#     # -------------------------------
#     # Step 2. Chunk-level grouping
#     # -------------------------------
#     G_atomic = getattr(atomic_knowledge_graph_inst, "_graph", None)
#     chunk_to_atomics = {}

#     if G_atomic:
#         # Sequential이면 마지막 단계 atomic fact만 사용
#         if reasoning_type == "sequential":
#             target_results = {"final_subquery": final_atomic_facts}
#         else:
#             target_results = subquery_results

#         for sq, res_list in target_results.items():
#             for r in res_list:
#                 atom_id = r.get("id")
#                 entity_name = r.get("entity_name")
#                 score = r.get("distance", 0.0)

#                 if not entity_name or entity_name not in G_atomic:
#                     continue

#                 try:
#                     node_data = await atomic_knowledge_graph_inst.get_node(entity_name)
#                 except Exception:
#                     node_data = None
#                 if not isinstance(node_data, dict):
#                     continue

#                 src_id = node_data.get("source_id")
#                 atomic_id = node_data.get("atomic_id") or atom_id
#                 if not src_id or not atomic_id:
#                     continue

#                 # atomic fact text
#                 try:
#                     atomic_data = await text_atomics_db.get_by_ids([atomic_id])
#                     atomic_text = atomic_data[0].get("content", "") if atomic_data else ""
#                 except Exception:
#                     atomic_text = ""

#                 # 연결된 chunk
#                 chunk_ids = [cid.strip() for cid in src_id.split("<SEP>") if cid.strip()]
#                 for cid in chunk_ids:
#                     try:
#                         chunk_data = await text_chunks_db.get_by_ids([cid])
#                         chunk_text = chunk_data[0].get("content", "") if chunk_data else ""
#                     except Exception:
#                         chunk_text = ""

#                     if cid not in chunk_to_atomics:
#                         chunk_to_atomics[cid] = {
#                             "chunk_id": cid,
#                             "chunk_text": chunk_text,
#                             "atomic_facts": []
#                         }

#                     chunk_to_atomics[cid]["atomic_facts"].append({
#                         "atomic_id": atomic_id,
#                         "atomic_text": atomic_text,
#                         "sub_query": sq,
#                         "score": score
#                     })

#         # 중복 제거
#         for cid, item in chunk_to_atomics.items():
#             seen = set()
#             unique = []
#             for af in item["atomic_facts"]:
#                 aid = af.get("atomic_id")
#                 if aid not in seen:
#                     seen.add(aid)
#                     unique.append(af)
#             item["atomic_facts"] = unique

#     results = list(chunk_to_atomics.values())

#     # -------------------------------
#     # Step 3. Chunk scoring (skip for sequential)
#     # -------------------------------
#     if reasoning_type != "sequential":
#         top_k_chunks = 10
#         scored_results = []
#         for item in results:
#             similarities = [
#                 af.get("score", 0.0)
#                 for af in item.get("atomic_facts", [])
#                 if isinstance(af.get("score"), (int, float))
#             ]
#             chunk_score = max(similarities) if similarities else 0.0
#             scored_results.append({**item, "chunk_score": chunk_score})
#         scored_results.sort(key=lambda x: x["chunk_score"], reverse=True)
#         results = scored_results[:top_k_chunks]
#         print(f"[INFO] Filtered top-{top_k_chunks} chunks based on max cosine similarity.")
#     else:
#         print("[SEQ] Skipping chunk scoring and filtering.")

#     # -------------------------------
#     # Step 4. Context generation
#     # -------------------------------
#     if reasoning_type == "sequential":
#         print("[SEQ] Building context for final subqueries...")
#         last_subqueries = [queries[i - 1] for i in final_query_indices]
#         executed_subqueries = "\n".join(f"- {sq}" for sq in subquery_results.keys())

#         atomic_str = "\n".join([
#             f"- {af.get('atomic_text','')}"
#             for item in results for af in item["atomic_facts"]
#         ])
#         chunk_texts = "\n\n".join([
#             item.get("chunk_text", "")
#             for item in results
#         ])

#         rewrite_prompt = PROMPTS["chunk_rewriting"].format(
#             main_query=f"{'; '.join(last_subqueries)}\n\n[Executed Subqueries]\n{executed_subqueries}",
#             chunk_text=chunk_texts,
#             atomic_facts=atomic_str
#         )

#         try:
#             context = await use_model_func(rewrite_prompt, stream=param.stream)
#         except Exception as e:
#             print(f"[ERROR] Context generation failed: {e}")
#             context = "no information"
#     else:
#         # 기존 병렬형 context 처리
#         context_blocks = []
#         for i, item in enumerate(results):
#             atomic_list_str = "\n".join(
#                 [f"- {af['atomic_text']}" for af in item.get("atomic_facts", []) if af.get("atomic_text")]
#             )
#             chunk_text = item.get("chunk_text", "")
#             summary_prompt = global_config["PROMPTS"]["chunk_rewriting"].format(
#                 main_query=query,
#                 chunk_text=chunk_text,
#                 atomic_facts=atomic_list_str
#             )
#             summary = await use_model_func(summary_prompt, stream=param.stream)
#             if isinstance(summary, str):
#                 summary = summary.strip()
#             if summary and "UNRELATED" not in summary:
#                 context_blocks.append(f"[Chunk {i+1}]\n{summary}\n")
#         context = "\n\n".join(context_blocks) if context_blocks else "no information"

#     # -------------------------------
#     # Step 5. Final Answer Generation
#     # -------------------------------
#     cls = os.path.basename(global_config["working_dir"])
#     if cls == "hotpot":
#         sys_prompt_temp = global_config["PROMPTS"]["rag_response_hotpot"]
#     elif cls == "multihoprag":
#         sys_prompt_temp = global_config["PROMPTS"]["rag_response_multihoprag"]
#     else:
#         sys_prompt_temp = global_config["PROMPTS"]["ours_rag_response"]

#     sys_prompt = sys_prompt_temp.format(context_data=context)

#     if reasoning_type == "sequential":
#         final_query = f"Sub-query: {'; '.join(last_subqueries)}\n\nContext:\n{context}"
#     else:
#         final_query = query

#     print(f"[INFO] Generating final answer...")
#     try:
#         final_response = await use_model_func(final_query, system_prompt=sys_prompt, stream=param.stream)
#     except Exception as e:
#         print(f"[ERROR] Final answer generation failed: {e}")
#         return "Final generation failed."

#     print("final prompt length:", len(sys_prompt))
#     print(final_response)
#     return final_response


async def ours_kg_query_experiment2(
    query,
    atomic_entity_relation_graph: BaseGraphStorage,
    triple_entity_relation_graph: BaseGraphStorage,
    triple_entities_vdb: BaseVectorStorage,
    text_atomics_db: BaseKVStorage,
    text_chunks_db: BaseKVStorage,
    param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:
    # Step 0: query decomposition
    if param.query_mode in ["experiment1"]:
        queries = await query_process_1(query, param, global_config, hashing_kv)
    elif param.query_mode in ["experiment2"]:
        queries = await query_process_2(query, param, global_config, hashing_kv)
    elif param.query_mode in ["experiment3"]:
        queries = await query_process_3(query, param, global_config, hashing_kv)
    elif param.query_mode in ["experiment4"]:
        queries = await query_process_4(query, param, global_config, hashing_kv)
    else:
        queries = [query]
        
    # Step 1: triple_entities_vdb에서 query 관련 triple 검색
    try:
        vdb_res = []
        if param.query_mode in ["experiment1", "experiment2", "experiment3", "experiment4"]:
            for q in queries:
                res = await triple_entities_vdb.query(q, top_k=10)
                for r in res:
                    r["sub_query"] = q
                vdb_res.extend(res)
        else:
            res = await triple_entities_vdb.query(query, top_k=10)
            for r in res:
                r["sub_query"] = query
            vdb_res.extend(res)
    except Exception as e:
        print(f"[ERROR] triple_entities_vdb query failed: {e}")
        vdb_res = []
        
    if not vdb_res:
        print("[INFO] No triples retrieved.")
        return []

    # subquery별 결과 저장
    subquery_results = {}
    G_triple = getattr(triple_entity_relation_graph, "_graph", None)
    G_atomic = getattr(atomic_entity_relation_graph, "_graph", None)

    if G_triple and G_atomic:
        for r in vdb_res:
            sq = r.get("sub_query", query)
            triple_id = r.get("id")
            triple_text = r.get("entity_name")
            score = r.get("distance")

            if not triple_id or not triple_text:
                continue

            # Step 2: triple graph에서 triple_text 매칭 → atomic_id(s) 찾기
            try:
                triple_node = await triple_entity_relation_graph.get_node(triple_text)
            except Exception:
                triple_node = None

            if not isinstance(triple_node, dict):
                continue

            src_ids = triple_node.get("source_id")
            if not src_ids:
                continue

            atomic_ids = [a.strip() for a in src_ids.split("<SEP>") if a.strip()]

            for atomic_id in atomic_ids:
                # Step 3: atomic fact 텍스트 가져오기
                try:
                    atomic_data = await text_atomics_db.get_by_ids([atomic_id])
                    atomic_text = atomic_data[0].get("content", "") if atomic_data else ""
                except Exception:
                    atomic_text = ""

                if not atomic_text:
                    continue

                # Step 4: atomic_entity_relation_graph에서 atomic_text로 검색 → chunk id(s) 찾기
                try:
                    atomic_node = await atomic_entity_relation_graph.get_node(atomic_text)
                except Exception:
                    atomic_node = None

                if not isinstance(atomic_node, dict):
                    continue

                chunk_ids = [
                    c.strip() for c in atomic_node.get("source_id", "").split("<SEP>")
                    if c.strip()
                ]

                # Step 5: chunk text 가져오기
                if chunk_ids:
                    try:
                        chunk_datas = await text_chunks_db.get_by_ids(chunk_ids)
                    except Exception:
                        chunk_datas = []
                else:
                    chunk_datas = []

                for cid, chunk_data in zip(chunk_ids, chunk_datas):
                    chunk_text = chunk_data.get("content", "") if chunk_data else ""
                    entry = {
                        "query": query,
                        "sub_query": sq,
                        "triple_id": triple_id,
                        "triple_text": triple_text,
                        "atomic_id": atomic_id,
                        "atomic_text": atomic_text,
                        "chunk_id": cid,
                        "chunk_text": chunk_text,
                        "score": score,
                    }
                    subquery_results.setdefault(sq, []).append(entry)

    # Step 6: 라운드 로빈 + 중복 제거 방식으로 결과 합치기
    results = []
    seen = set()
    pointers = {sq: 0 for sq in subquery_results}  # 각 subquery 포인터

    while True:
        progress = False
        for sq, res_list in subquery_results.items():
            idx = pointers[sq]
            # 중복 건너뛰기
            while idx < len(res_list) and res_list[idx]["chunk_id"] in seen:
                idx += 1
            if idx < len(res_list):
                r = res_list[idx]
                results.append(r)
                seen.add(r["chunk_id"])
                pointers[sq] = idx + 1
                progress = True

        if not progress:  # 더 이상 뽑을 게 없으면 종료
            break
    
    results = results[:10]
    
    # Step 7: JSON 저장
    cls = os.path.basename(global_config["working_dir"])
    if param.query_mode in ["experiment1"]:
        output_path = f"/workspace/AtomRAG/{cls}/query_decom1_triple_mapping.json"
    elif param.query_mode in ["experiment2"]:
        output_path = f"/workspace/AtomRAG/{cls}/query_decom2_triple_mapping.json"
    elif param.query_mode in ["experiment3"]:
        output_path = f"/workspace/AtomRAG/{cls}/query_decom3_triple_mapping.json"
    elif param.query_mode in ["experiment4"]:
        output_path = f"/workspace/AtomRAG/{cls}/query_decom4_triple_mapping.json"
    else:
        output_path = f"/workspace/AtomRAG/{cls}/query_triple_mapping.json"

    output_entry = {
        "query": query,
        "results": results
    }

    try:
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                if not isinstance(old_data, list):
                    old_data = [old_data]
        else:
            old_data = []

        old_data.append(output_entry)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(old_data, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Results for query='{query}' appended to {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")

    return output_entry

async def ours_kg_query_experiment3(
    query,
    atomic_knowledge_graph_inst: BaseGraphStorage,
    triple_knowledge_graph_inst: BaseGraphStorage,
    atomic_entities_vdb: BaseVectorStorage,
    text_atomics_db: BaseKVStorage,
    text_chunks_db: BaseKVStorage,
    param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:
    # Step 0: query decomposition
    if param.query_mode in ["experiment1"]:
        queries = await query_process_1(query, param, global_config, hashing_kv)
    elif param.query_mode in ["experiment2"]:
        queries = await query_process_2(query, param, global_config, hashing_kv)
    elif param.query_mode in ["experiment3"]:
        queries = await query_process_3(query, param, global_config, hashing_kv)
    elif param.query_mode in ["experiment4"]:
        queries = await query_process_4(query, param, global_config, hashing_kv)
    else:
        queries = [query]
        
    # Step 1: subquery별 atomic fact 검색
    try:
        subquery_results = {}
        if param.query_mode in ["experiment1", "experiment2", "experiment3", "experiment4"]:
            for q in queries:
                res = await atomic_entities_vdb.query(q, top_k=10)
                subquery_results[q] = res
        else:
            res = await atomic_entities_vdb.query(query, top_k=10)
            subquery_results[query] = res
    except Exception as e:
        print(f"[ERROR] atomic_entities_vdb query failed: {e}")
        subquery_results = {}

    if not subquery_results:
        print("[INFO] No atomic facts retrieved.")
        return []

    # Step 2: subquery별 chunk 리스트 생성
    subquery_chunk_results = {}
    G_atomic = getattr(atomic_knowledge_graph_inst, "_graph", None)

    if G_atomic:
        for sq, res_list in subquery_results.items():
            chunk_list = []
            for r in res_list:
                atom_id = r.get("id")
                entity_name = r.get("entity_name")
                score = r.get("distance")

                if not entity_name or entity_name not in G_atomic:
                    continue

                try:
                    node_data = await atomic_knowledge_graph_inst.get_node(entity_name)
                except Exception:
                    node_data = None

                if not isinstance(node_data, dict):
                    continue

                src_id = node_data.get("source_id")
                atomic_id = node_data.get("atomic_id") or atom_id
                if not src_id or not atomic_id:
                    continue

                chunk_ids = [c.strip() for c in src_id.split("<SEP>") if c.strip()]

                try:
                    atomic_data = await text_atomics_db.get_by_ids([atomic_id])
                    atomic_text = atomic_data[0].get("content", "") if atomic_data else ""
                except Exception:
                    atomic_text = ""

                try:
                    chunk_datas = await text_chunks_db.get_by_ids(chunk_ids) if chunk_ids else []
                except Exception:
                    chunk_datas = []

                for cid, chunk_data in zip(chunk_ids, chunk_datas):
                    chunk_text = chunk_data.get("content", "") if chunk_data else ""
                    chunk_list.append({
                        "query": query,
                        "sub_query": sq,
                        "atomic_id": atomic_id,
                        "atomic_text": atomic_text,
                        "chunk_id": cid,
                        "chunk_text": chunk_text,
                        "score": score
                    })
            subquery_chunk_results[sq] = chunk_list

    # Step 3: 라운드 로빈 + 중복 제거
    results = []
    seen = set()
    pointers = {sq: 0 for sq in subquery_chunk_results}

    while True:
        progress = False
        for sq, res_list in subquery_chunk_results.items():
            idx = pointers[sq]
            # 중복 제거
            while idx < len(res_list) and res_list[idx]["chunk_id"] in seen:
                idx += 1
            if idx < len(res_list):
                r = res_list[idx]
                if r["chunk_id"] not in seen:
                    results.append(r)
                    seen.add(r["chunk_id"])
                pointers[sq] = idx + 1
                progress = True
        if not progress:
            break

    results = results[:10]

    # Step 4: JSON 저장
    cls = os.path.basename(global_config["working_dir"])
    if param.query_mode in ["experiment1"]:
        output_path = f"/workspace/AtomRAG/{cls}/query_decom1_atomic(1-hop)_mapping.json"
    elif param.query_mode in ["experiment2"]:
        output_path = f"/workspace/AtomRAG/{cls}/query_decom2_atomic(1-hop)_mapping.json"
    elif param.query_mode in ["experiment3"]:
        output_path = f"/workspace/AtomRAG/{cls}/query_decom3_atomic(1-hop)_mapping.json"
    elif param.query_mode in ["experiment4"]:
        output_path = f"/workspace/AtomRAG/{cls}/query_decom4_atomic(1-hop)_mapping.json"
    else:
        output_path = f"/workspace/AtomRAG/{cls}/query_atomic(1-hop)_mapping.json"
    output_entry = {
        "query": query,
        "results": results   # 위에서 만든 atomic/chunk 리스트
    }

    try:
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                if not isinstance(old_data, list):
                    old_data = [old_data]
        else:
            old_data = []

        old_data.append(output_entry)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(old_data, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Results for query='{query}' appended to {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")

    return output_entry

async def query_process_1(
    text: str,
    param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> tuple[list[str], list[str]]:
    """
    Extract query extension from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extend query (extended query).
    """

    # 1. check cache(extended queries)
    args_hash = compute_args_hash(param.mode, text, cache_type="extended queries")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="extended queries"
    )
    if cached_response is not None:
        try:
            extended_datas = json.loads(cached_response)
            return extended_datas["extended queries"]
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for extended queries, proceeding with extension"
            )

    # 2. Build the query-process prompt
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["query_process_examples1"]):
        examples = "\n".join(
            PROMPTS["query_process_examples1"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["query_process_examples1"])

    kw_prompt = PROMPTS["query_process1"].format(
        query=text, examples=examples,
    )

    # 3. Call the LLM for query_process
    use_model_func = global_config["llm_model_func"]
    result = await use_model_func(kw_prompt, query_extension=True)
    
    # 4. Parse out JSON from the LLM response
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("No JSON-like structure found in the LLM respond.")
        return [], []
    try:
        process_datas = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return [], []

    subqueries = process_datas.get("sub_queries", [])
    
    # 5. Cache only the processed subqueries with cache type
    if subqueries:
        cache_data = {
            "subqueries": subqueries
        }
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=json.dumps(cache_data),
                prompt=text,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=param.mode,
                cache_type="subqueries",
            ),
        )

    return subqueries

async def query_process_2(
    text: str,
    param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> tuple[list[str], list[str]]:
    """
    Extract query extension from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extend query (extended query).
    """

    # 1. check cache(extended queries)
    args_hash = compute_args_hash(param.mode, text, cache_type="extended queries")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="extended queries"
    )
    if cached_response is not None:
        try:
            extended_datas = json.loads(cached_response)
            return extended_datas["extended queries"]
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for extended queries, proceeding with extension"
            )

    # 2. Build the query-process prompt
    # 2-1. Build the examples
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["query_process_examples2"]):
        examples = "\n".join(
            PROMPTS["query_process_examples2"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["query_process_examples2"])

    # 2-2. Process conversation history
    kw_prompt = PROMPTS["query_process2"].format(
        query=text, examples=examples,
    )

    # 3. Call the LLM for query_process
    use_model_func = global_config["llm_model_func"]
    result = await use_model_func(kw_prompt, query_extension=True)
    
    # 4. Parse out JSON from the LLM response
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("No JSON-like structure found in the LLM respond.")
        return [], []
    try:
        process_datas = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return [], []

    keywords_nested = process_datas.get("keywords", [])
    flat_keywords = [kw for sublist in keywords_nested for kw in sublist]
    keywords = list(OrderedDict.fromkeys(flat_keywords))
    
    # 5. Cache only the processed keywords with cache type
    if keywords:
        cache_data = {
            "keywords": keywords
        }
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=json.dumps(cache_data),
                prompt=text,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=param.mode,
                cache_type="keywords",
            ),
        )

    return keywords

async def query_process_3(
    text: str,
    param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> tuple[list[str], list[str]]:
    #1. check cache(extended queries)
    args_hash = compute_args_hash(param.mode, text, cache_type="extended queries")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="extended queries"
    )
    if cached_response is not None:
        try:
            extended_datas = json.loads(cached_response)
            return extended_datas["extended queries"]
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for extended queries, proceeding with extension"
            )

    # 2. Build the query-process prompt
    # 2-1. Build the examples
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["query_process_examples3"]):
        examples = "\n".join(
            PROMPTS["query_process_examples3"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["query_process_examples3"])

    # 2-2. Process conversation history
    kw_prompt = PROMPTS["query_process3"].format(
        query=text, examples=examples,
    )

    # 3. Call the LLM for query_process
    use_model_func = global_config["llm_model_func"]
    result = await use_model_func(kw_prompt, query_extension=True)
    
    # 4. Parse out JSON from the LLM response
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("No JSON-like structure found in the LLM respond.")
        return [], []
    try:
        process_datas = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return [], []

    sub_queries = process_datas.get("sub_queries", [])
    
    # 5. Cache only the processed sub_queries with cache type
    if sub_queries:
        cache_data = {
            "sub_queries": sub_queries
        }
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=json.dumps(cache_data),
                prompt=text,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=param.mode,
                cache_type="sub_queries",
            ),
        )

    return sub_queries

async def query_process_4(
    text: str,
    param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> tuple[list[str], list[str]]:
    """
    Extract query extension from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extend query (extended query).
    """

    # 1. check cache(extended queries)
    args_hash = compute_args_hash(param.mode, text, cache_type="extended queries")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="extended queries"
    )
    if cached_response is not None:
        try:
            extended_datas = json.loads(cached_response)
            return extended_datas["extended queries"]
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for extended queries, proceeding with extension"
            )

    # 2. Build the query-process prompt
    # 2-1. Build the examples
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["query_process_examples"]):
        examples = "\n".join(
            PROMPTS["query_process_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["query_process_examples"])

    # 2-2. Process conversation history
    kw_prompt = PROMPTS["query_process"].format(
        query=text, examples=examples,
    )

    # 3. Call the LLM for query_process
    use_model_func = global_config["llm_model_func"]
    result = await use_model_func(kw_prompt, query_extension=True)
    
    # 4. Parse out JSON from the LLM response
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("No JSON-like structure found in the LLM respond.")
        return [], []
    try:
        process_datas = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return [], []

    keywords_nested = process_datas.get("keywords", [])
    flat_keywords = [kw for sublist in keywords_nested for kw in sublist]
    keywords = list(OrderedDict.fromkeys(flat_keywords))
    
    # 5. Cache only the processed keywords with cache type
    if keywords:
        cache_data = {
            "keywords": keywords
        }
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=json.dumps(cache_data),
                prompt=text,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=param.mode,
                cache_type="keywords",
            ),
        )

    return keywords

async def query_process_5(
    text: str,
    param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> tuple[list[str], list[str]]:
    """
    Extract query extension from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extend query (extended query).
    """

    # 1. check cache(extended queries)
    args_hash = compute_args_hash(param.mode, text, cache_type="extended queries")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="extended queries"
    )
    if cached_response is not None:
        try:
            extended_datas = json.loads(cached_response)
            return extended_datas["extended queries"]
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for extended queries, proceeding with extension"
            )

    # 2. Build the query-process prompt
    # 2-1. Build the examples
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["query_process_examples5"]):
        examples = "\n".join(
            PROMPTS["query_process_examples5"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["query_process_examples5"])

    # 2-2. Process conversation history
    kw_prompt = PROMPTS["query_process5"].format(
        query=text, examples=examples,
    )

    # 3. Call the LLM for query_process
    use_model_func = global_config["llm_model_func"]
    result = await use_model_func(kw_prompt, query_extension=True)
    
    # 4. Parse out JSON from the LLM response
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("No JSON-like structure found in the LLM response.")
        return "", [], []
    try:
        process_datas = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return "", [], []

    # Extract reasoning structure
    atomic_facts = process_datas.get("atomic_facts", [])
    sub_queries = process_datas.get("sub_queries", [])

    # 5. Cache the processed reasoning info
    if sub_queries:
        cache_data = {
            "atomic_facts": atomic_facts,
            "sub_queries": sub_queries
        }
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=json.dumps(cache_data),
                prompt=text,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=param.mode,
                cache_type="extended queries",
            ),
        )

    return sub_queries



async def query_process(
    text: str,
    param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> tuple[list[str], list[str]]:
    """
    Extract query extension from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extend query (extended query).
    """

    # 1. check cache(extended queries)
    args_hash = compute_args_hash(param.mode, text, cache_type="extended queries")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="extended queries"
    )
    if cached_response is not None:
        try:
            extended_datas = json.loads(cached_response)
            return extended_datas["extended queries"]
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for extended queries, proceeding with extension"
            )

    # 2. Build the query-process prompt
    # 2-1. Build the examples
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["query_process_examples"]):
        examples = "\n".join(
            PROMPTS["query_process_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["query_process_examples"])

    # 2-2. Process conversation history
    kw_prompt = PROMPTS["query_process"].format(
        query=text, examples=examples,
    )

    # 3. Call the LLM for query_process
    use_model_func = global_config["llm_model_func"]
    result = await use_model_func(kw_prompt, query_extension=True)
    
    # 4. Parse out JSON from the LLM response
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("No JSON-like structure found in the LLM respond.")
        return [], []
    try:
        process_datas = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return [], []

    keywords = process_datas.get("keywords", [])
    
    # 5. Cache only the processed keywords with cache type
    if keywords:
        cache_data = {
            "keywords": keywords
        }
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=json.dumps(cache_data),
                prompt=text,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=param.mode,
                cache_type="keywords",
            ),
        )

    return keywords

async def ours_extract_keywords_only(
    text: str,
    param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> tuple[list[str], list[str]]:
    """
    Extract ours keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (ours_keywords).
    """

    # 1. Handle cache if needed - add cache type for keywords
    args_hash = compute_args_hash(param.mode, text, cache_type="ours_keywords")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="ours_keywords"
    )

    if cached_response is not None:
        try:
            keywords_data = json.loads(cached_response)
            return keywords_data["ours_keywords"]
        
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )
    
    # 2. Build the examples
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["ours_keywords_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["ours_keywords_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["ours_keywords_extraction_examples"])
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    # 3. Process conversation history
    history_context = ""
    if param.conversation_history:
        history_context = get_conversation_turns(
            param.conversation_history, param.history_turns
        )

    # 4. Build the keyword-extraction prompt
    kw_prompt = PROMPTS["ours_keywords_extraction"].format(
        query=text, examples=examples, language=language, history=history_context
    )

    # 5. Call the LLM for keyword extraction
    use_model_func = global_config["llm_model_func"]
    result = await use_model_func(kw_prompt, keyword_extraction=True)

    # 6. Parse out JSON from the LLM response
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("No JSON-like structure found in the LLM respond.")
        return [], []
    try:
        keywords_data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return [], []

    ours_keywords = keywords_data.get("keywords", [])

    # 7. Cache only the processed keywords with cache type
    if ours_keywords:
        cache_data = {
            "ours_keywords": ours_keywords,
        }
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=json.dumps(cache_data),
                prompt=text,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=param.mode,
                cache_type="ours_keywords",
            ),
        )
    return ours_keywords

async def _build_query_context(
    query: list,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    # ll_entities_context, ll_relations_context, ll_text_units_context = "", "", ""
    # hl_entities_context, hl_relations_context, hl_text_units_context = "", "", ""

    ll_keywords, hl_keywords = query[0], query[1]

    if query_param.mode == "local":
        entities_context, relations_context, text_units_context = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    elif query_param.mode == "global":
        entities_context, relations_context, text_units_context = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )
    else:  # hybrid mode
        ll_data, hl_data = await asyncio.gather(
            _get_node_data(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
            ),
            _get_edge_data(
                hl_keywords,
                knowledge_graph_inst,
                relationships_vdb,
                text_chunks_db,
                query_param,
            ),
        )

        (
            ll_entities_context,
            ll_relations_context,
            ll_text_units_context,
        ) = ll_data

        (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
        ) = hl_data

        entities_context, relations_context, text_units_context = combine_contexts(
            [hl_entities_context, ll_entities_context],
            [hl_relations_context, ll_relations_context],
            [hl_text_units_context, ll_text_units_context],
        )
    # not necessary to use LLM to generate a response
    if not entities_context.strip() and not relations_context.strip():
        return None

    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""

async def ours_build_query_context(
    query,
    keywords: list,
    atomic_knowledge_graph_inst: BaseGraphStorage,
    triple_knowledge_graph_inst: BaseGraphStorage,
    atomic_entities_vdb: BaseVectorStorage,
    atomic_relationships_vdb: BaseVectorStorage,
    triple_entities_vdb: BaseVectorStorage,
    triple_relationships_vdb: BaseVectorStorage,
    text_atomics_db: BaseKVStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    context = ""
    
    if query_param.mode == "ours":
        results = await ours_hierarchical_search(
                query,
                keywords,
                atomic_knowledge_graph_inst,
                triple_knowledge_graph_inst,
                atomic_entities_vdb,
                atomic_relationships_vdb,
                triple_entities_vdb,
                triple_relationships_vdb,
                text_atomics_db,
                text_chunks_db,
                query_param,
            )
        
        # results에서 context 필드만 추출
        context_parts = [res.get("context", "") for res in results if res.get("context")]
        context = "\n".join(context_parts)

    return context
        
async def _get_node_data(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    # get similar entities
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return "", "", ""
    
    # get entity information
    node_datas, node_degrees = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
        ),
        asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
        ),
    )

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    
    # get entitytext chunk
    use_text_units, use_relations = await asyncio.gather(
        _find_most_related_text_unit_from_entities(
            node_datas, query_param, text_chunks_db, knowledge_graph_inst
        ),
        _find_most_related_edges_from_entities(
            node_datas, query_param, knowledge_graph_inst
        ),
    )
    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )

    # build prompt
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        [
            "id",
            "source",
            "target",
            "description",
            "keywords",
            "weight",
            "rank",
            "created_at",
        ]
    ]
    for i, e in enumerate(use_relations):
        created_at = e.get("created_at", "UNKNOWN")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
                created_at,
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context
        
def mean_pool(vecs):
    return np.mean(vecs, axis=0)

def cos_sim(a, b):
    a = np.ravel(a) 
    b = np.ravel(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

async def ours_hierarchical_search(
    query,
    keywords,
    atomic_knowledge_graph_inst: BaseGraphStorage,
    triple_knowledge_graph_inst: BaseGraphStorage,
    atomic_entities_vdb: BaseVectorStorage,
    atomic_relationships_vdb: BaseVectorStorage,
    triple_entities_vdb: BaseVectorStorage,
    triple_relationships_vdb: BaseVectorStorage,
    text_atomics_db: BaseKVStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    selected_entities = []   # 전체 keyword별 top-1 entity
    
    async def query_entity(kw):
        try:
            vdb_res = await triple_entities_vdb.query(kw, top_k=1)
        except Exception:
            vdb_res = []

        if not vdb_res:
            return {
                "entity_id": None,
                "entity_name": None,
                "score": None,
            }

        top = vdb_res[0]
        return {
            "entity_id": top.get("id"),
            "entity_name": top.get("entity_name"),
            "score": top.get("distance"),
        }

    # Step 1: 각 keyword 단어별 top-1 entity 검색 (병렬 실행)
    tasks = []
    for group_idx, kw_group in enumerate(keywords):
        for kw in kw_group:
            tasks.append(query_entity(kw))

    results = await asyncio.gather(*tasks)
    selected_entities.extend(results)
            
    unique_entities = {}
    
    for ent in selected_entities:
        ent_id = ent.get("entity_id")
        if ent_id:  # entity_id가 있는 경우
            if ent_id not in unique_entities:
                unique_entities[ent_id] = ent

    selected_entities = list(unique_entities.values())
    
    # Step 2: graph에서 가져온 entity + 1-hop neighbor 전부 저장
    graph_entities = []  

    for ent in selected_entities:
        entity_id = ent.get("entity_id")
        entity_name = ent.get("entity_name")

        if not entity_id and not entity_name:
            continue

        try:
            G = getattr(triple_knowledge_graph_inst, "_graph", None)
            node_key = None
            if G:
                if entity_name and entity_name in G:
                    node_key = entity_name
                elif entity_id and entity_id in G:
                    node_key = entity_id

            if node_key:
                # 1) 기준 entity 자체를 graph에서 가져옴
                try:
                    base_node = await triple_knowledge_graph_inst.get_node(node_key)
                except Exception:
                    base_node = None
                if isinstance(base_node, dict):
                    base_node["entity_name"] = node_key
                    graph_entities.append(base_node)
                else:
                    graph_entities.append({"entity_name": node_key})

                # 2) 기준 entity의 1-hop neighbor들 가져오기
                for neighbor in G.neighbors(node_key):
                    try:
                        n = await triple_knowledge_graph_inst.get_node(neighbor)
                    except Exception:
                        n = None
                    if isinstance(n, dict):
                        n["entity_name"] = neighbor
                        graph_entities.append(n)
                    else:
                        graph_entities.append({"entity_name": neighbor})
        except Exception:
            continue

    unique_graph_entities = {}
    for e in graph_entities:
        name = e.get("entity_name")
        if name and name not in unique_graph_entities:
            unique_graph_entities[name] = e

    graph_entities = list(unique_graph_entities.values())
    
    # Step 3: source_id에서 "atom-"으로 시작하는 id 정리
    atom_ids_collection = []  # 전체 atom id 리스트
    for e in graph_entities:
        src_id = e.get("source_id", "")
        if not src_id:
            continue
        
        # 여러 개면 <SEP> 기준으로 분리
        parts = src_id.split("<SEP>")
        for p in parts:
            p = p.strip()
            if p.startswith("atom-"):
                atom_ids_collection.append(p)

    atom_ids_collection = list(set(atom_ids_collection))

    # Step 4: text_atomics_db에서 atomic_id 기반으로 문장 가져오기
    atomic_base_entities = []

    try:
        # 여러 개 한 번에 가져오기
        nodes_data = await text_atomics_db.get_by_ids(atom_ids_collection)  
        # 반환은 dict 형태라고 가정: {atom_id: {"content": "..."}}
    except Exception:
        nodes_data = {}

    for atom_id, node_data in zip(atom_ids_collection, nodes_data):
        if node_data and "content" in node_data:
            atomic_base_entities.append({
                "entity_name": node_data["content"],  # atomic fact 문장
                "atom_id": atom_id                    # 원래 id 같이 저장
            })

    unique_atomic_base_entities = {}
    for e in atomic_base_entities:
        name = e.get("entity_name")
        if name and name not in unique_atomic_base_entities:
            unique_atomic_base_entities[name] = e

    atomic_base_entities = list(unique_atomic_base_entities.values())

    # Step 5: atomic_base_entities 기준으로 graph에서 본인 + 1-hop neighbors 가져오기
    atomic_graph_entities = []

    try:
        G_atomic = getattr(atomic_knowledge_graph_inst, "_graph", None)
    except Exception:
        G_atomic = None

    if G_atomic:
        for ent in atomic_base_entities:
            atom_id = ent.get("atom_id")
            base_name = ent.get("entity_name")  # 문장

            # atomic graph에서는 node_id가 문장(string)인 구조니까 base_name 사용
            if not base_name or base_name not in G_atomic:
                continue

            # 1) 기준 노드 정보 가져오기
            try:
                base_node = await atomic_knowledge_graph_inst.get_node(base_name)
            except Exception:
                base_node = None

            if isinstance(base_node, dict):
                base_node["entity_name"] = base_name
                base_node["atom_id"] = atom_id
                atomic_graph_entities.append(base_node)
            else:
                atomic_graph_entities.append({
                    "entity_name": base_name,
                    "atom_id": atom_id
                })

            # 2) 기준 노드의 1-hop neighbor 정보 가져오기
            for neighbor in G_atomic.neighbors(base_name):
                try:
                    n = await atomic_knowledge_graph_inst.get_node(neighbor)
                except Exception:
                    n = None

                if isinstance(n, dict):
                    n["entity_name"] = neighbor
                    atomic_graph_entities.append(n)
                else:
                    atomic_graph_entities.append({"entity_name": neighbor})

    unique_atomic_graph_entities = {}
    for e in atomic_graph_entities:
        name = e.get("entity_name")
        if name and name not in unique_atomic_graph_entities:
            unique_atomic_graph_entities[name] = e

    atomic_graph_entities = list(unique_atomic_graph_entities.values())

    # Step 6: atomic_graph_entities에서 atomic_id 수집
    atom_ids_for_chunks = []
    for e in atomic_graph_entities:
        atom_id = e.get("atomic_id") or e.get("atom_id")
        if atom_id:
            atom_ids_for_chunks.append(atom_id)

    atom_ids_for_chunks = list(set(atom_ids_for_chunks))  # 중복 제거

    # Step 7: atomic_id -> chunk_id 매핑 (source_id가 chunk id임)
    atom_to_chunk = {}
    for e in atomic_graph_entities:
        atom_id = e.get("atomic_id") or e.get("atom_id")
        src_id = e.get("source_id")
        if atom_id and src_id and src_id.startswith("chunk-"):
            atom_to_chunk[atom_id] = src_id

    raw_chunk_ids = list(set(atom_to_chunk.values()))
    chunk_ids = []
    for cid in raw_chunk_ids:
        if "<SEP>" in cid:
            chunk_ids.extend([c.strip() for c in cid.split("<SEP>") if c.strip()])
        else:
            chunk_ids.append(cid.strip())

    chunk_ids = list(set(chunk_ids))
    chunk_datas = await text_chunks_db.get_by_ids(chunk_ids)

    chunk_to_atomics = {}
    for cid, chunk_data in zip(chunk_ids, chunk_datas):  # zip으로 id와 data 매핑
        content = chunk_data.get("content", "")
        chunk_to_atomics[cid] = {
            "content": content,
            "atomics": []
        }

    for e in atomic_graph_entities:
        atom_id = e.get("atomic_id") or e.get("atom_id")
        atom_name = e.get("entity_name")
        chunk_id = atom_to_chunk.get(atom_id)

        # <SEP> 가 들어간 경우도 고려해야 함
        if chunk_id:
            for cid in chunk_id.split("<SEP>"):
                cid = cid.strip()
                if cid and cid in chunk_to_atomics:
                    chunk_to_atomics[cid]["atomics"].append(atom_name)

    from openai import AsyncOpenAI

    client = AsyncOpenAI()

    async def rewrite_context(cid, chunk_content, facts, token_budget):
        if not facts:
            return None

        prompt = f"""
        You are given a chunk of text and extracted atomic facts.
        Summarize this chunk based ONLY on the atomic facts.
        The result should be concise, faithful to the facts,
        and no longer than {token_budget} tokens.

        Chunk:
        {chunk_content}

        Atomic facts:
        {facts}

        Rewritten context (≤ {token_budget} tokens):
        """
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            rewritten_context = response.choices[0].message.content.strip()
        except Exception:
            rewritten_context = " ".join(facts)

        return {
            "chunk_id": cid,
            "facts": facts,
            "context": rewritten_context
        }


    async def process_in_batches(chunk_items, token_budget, batch_size=10):
        contexts = []
        for i in range(0, len(chunk_items), batch_size):
            batch = chunk_items[i:i+batch_size]
            tasks = [
                rewrite_context(cid, data["content"], data["atomics"], token_budget)
                for cid, data in batch
            ]
            results = await asyncio.gather(*tasks)
            contexts.extend([c for c in results if c])
        return contexts


    # Step 8: chunk selection
    chunk_items = list(chunk_to_atomics.items())

    if len(chunk_items) > 30:
        chunk_items = sorted(
            chunk_items,
            key=lambda x: len(x[1]["atomics"]),
            reverse=True
        )[:30]

    num_chunks = len(chunk_items)
    total_budget = 5000
    token_budget = total_budget // max(1, num_chunks)

    # Step 9: run batch processing (10개씩)
    contexts = await process_in_batches(chunk_items, token_budget, batch_size=10)

    return contexts


    
    
async def get_text_embedding(text):
    # Ensure input is a list (API supports batch requests)
    if isinstance(text, str):
        text = [text]  # Convert to a list
        
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    
    # Extract embeddings and convert to NumPy array
    embeddings = np.array([res.embedding for res in response.data])
    
    return embeddings

async def rerank_by_cosine_similarity(query, items, key="description"):
    if not items:
        return []
    
    descriptions = [item[key] for item in items]  # Extract descriptions

    # Get embeddings
    query_embedding = await get_text_embedding(query)  # Shape: (1, embedding_dim)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    item_embeddings = await get_text_embedding(descriptions)
    item_embeddings = np.array(item_embeddings)
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, item_embeddings).flatten()

    # Sort items based on similarity scores (descending order)
    sorted_items = sorted(zip(similarities, items), key=lambda x: x[0], reverse=True)
    
    # Return only the sorted items
    return [item for _, item in sorted_items]

async def filter_node_query(knowledge_graph_inst: BaseGraphStorage, results, query, top_k):
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    
    descriptions = [item["entity_name"]+info["description"] for item, info in zip(results, node_datas)]  # Extract descriptions

    # Get embeddings
    query_embedding = await get_text_embedding(query)  # Shape: (1, embedding_dim)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    item_embeddings = await get_text_embedding(descriptions)
    item_embeddings = np.array(item_embeddings)
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, item_embeddings).flatten()

    # Sort items based on similarity scores (descending order)
    sorted_results = sorted(zip(similarities, results), key=lambda x: x[0], reverse=True)
    
    # Return only the top_k results
    return [item for _, item in sorted_results[:top_k]]

async def filter_edge_query(knowledge_graph_inst: BaseGraphStorage, results, query, top_k):
    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )
    
    descriptions = [info["keywords"]+item["src_id"]+item["tgt_id"]+info["description"] for item, info in zip(results, edge_datas)]  # Extract descriptions

    # Get embeddings
    query_embedding = await get_text_embedding(query)  # Shape: (1, embedding_dim)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    item_embeddings = await get_text_embedding(descriptions)
    item_embeddings = np.array(item_embeddings)
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, item_embeddings).flatten()

    # Sort items based on similarity scores (descending order)
    sorted_results = sorted(zip(similarities, results), key=lambda x: x[0], reverse=True)
    
    # Return only the top_k results
    return [item for _, item in sorted_results[:top_k]]

async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    tasks = []
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                tasks.append((c_id, index, this_edges))

    results = await asyncio.gather(
        *[text_chunks_db.get_by_id(c_id) for c_id, _, _ in tasks]
    )

    for (c_id, index, this_edges), data in zip(tasks, results):
        all_text_units_lookup[c_id] = {
            "data": data,
            "order": index,
            "relation_counts": 0,
        }

        if this_edges:
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    all_edges_pack, all_edges_degree = await asyncio.gather(
        asyncio.gather(*[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]),
        asyncio.gather(
            *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
        ),
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return "", "", ""

    edge_datas, edge_degree = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
        ),
        asyncio.gather(
            *[
                knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"])
                for r in results
            ]
        ),
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")

    edge_datas = [
        {
            "src_id": k["src_id"],
            "tgt_id": k["tgt_id"],
            "rank": d,
            "created_at": k.get("__created_at__", None),  # 从 KV 存储中获取时间元数据
            **v,
        }
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    use_entities, use_text_units = await asyncio.gather(
        _find_most_related_entities_from_relationships(
            edge_datas, query_param, knowledge_graph_inst
        ),
        _find_related_text_unit_from_relationships(
            edge_datas, query_param, text_chunks_db, knowledge_graph_inst
        ),
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )
    

    relations_section_list = [
        [
            "id",
            "source",
            "target",
            "description",
            "keywords",
            "weight",
            "rank",
            "created_at",
        ]
    ]
    for i, e in enumerate(edge_datas):
        created_at = e.get("created_at", "Unknown")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
                created_at,
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context

async def ours_get_edge_data(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    if query_param.mode in ["ours", "ours1"]:
        results = await relationships_vdb.query(query, top_k=query_param.top_k)

        if not len(results):
            return "", "", ""
    elif query_param.mode in ["ours2", "ours3", "ours6", "ours7"]:
        if query_param.mode == "ours2":
            tasks = [relationships_vdb.query(q, top_k=20) for q in query]
        elif query_param.mode in ["ours3", "ours6", "ours7"]:
            tasks = [relationships_vdb.query(q, top_k=40) for q in query]
            
        results_list = await asyncio.gather(*tasks) 
        
        results = []
        unique_edges = set()
        edge_map = {}
        
        for idx, results_per_query in enumerate(results_list):
            step_type = f"{query_param.mode}_step{idx+1}_edge"
            for item in results_per_query:
                src, tgt = item["src_id"], item["tgt_id"]
                edge_key = (src, tgt) if src < tgt else (tgt, src)
                
                if edge_key not in unique_edges:
                    unique_edges.add(edge_key)
                    item["type"] = step_type
                    edge_map[edge_key] = item
                else:
                    edge_map[edge_key]["type"] = f"{query_param.mode}_duplication_edge"
        
        results = list(edge_map.values())
        
        if not len(results):
            return "", "", ""
    elif query_param.mode in ["ours4"]:
        type_queries = {}
        type_queries["main_query"] = query[0][0]  # 메인 쿼리 저장

        # Step-by-step query 저장
        for i in range(len(query[1])):  # query[1]의 길이만큼 반복
            step_key = f"step{i+1}_query"  
            step_queries = [query[1][i]]  # step-by-step의 메인 질문 추가
            
            # 각 step-by-step에서 생성된 query 추가 (3개씩)
            for j in range(3):
                step_queries.append(query[2][3 * i + j])
            
            # 리스트를 하나의 문자열로 변환 (쉼표로 구분)
            type_queries[step_key] = ", ".join(step_queries)

        results = []
        for i in range(0, len(query[1])+1):
            if i == 0:
                results = await relationships_vdb.query(type_queries["main_query"], top_k=60*(len(query[1])+1))
            else:
                results = await filter_edge_query(knowledge_graph_inst, results, type_queries[f"step{i}_query"], top_k=60*(len(query[1])+1-i))
                print(f"step{i}_filter_edge_query, top_k: {60*(len(query[1])+1-i)}")
    elif query_param.mode in ["ours8"]:
        type_queries = {}
        type_queries["main_query"] = query[0][0]  # 메인 쿼리 저장

        # Step-by-step query 저장
        for i in range(len(query[1])):  # query[1]의 길이만큼 반복
            step_key = f"step{i+1}_query"  
            step_queries = [query[1][i]]  # step-by-step의 메인 질문 추가
            
            # 각 step-by-step에서 생성된 query 추가 (3개씩)
            for j in range(3):
                step_queries.append(query[2][3 * i + j])
            
            # 리스트를 하나의 문자열로 변환 (쉼표로 구분)
            type_queries[step_key] = ", ".join(step_queries)

        results = []
        for i in range(0, len(query[1])+1):
            if i == 0:
                results = await relationships_vdb.query(type_queries["main_query"], top_k=60*2**(len(query[1])))
            else:
                results = await filter_edge_query(knowledge_graph_inst, results, type_queries[f"step{i}_query"], top_k=60*2**(len(query[1])-i))
                print(f"step{i}_filter_edge_query, top_k: {60*2**(len(query[1])-i)}")
    elif query_param.mode in ["ours5"]:
        type_queries = {}
        type_queries["main_query"] = query[0][0]  # 메인 쿼리 저장

        # Step-by-step query 저장
        for i in range(len(query[1])):  # query[1]의 길이만큼 반복
            step_key = f"step{i+1}_query"  
            step_queries = [query[1][i]]  # step-by-step의 메인 질문 추가
            
            # 각 step-by-step에서 생성된 query 추가 (3개씩)
            for j in range(3):
                step_queries.append(query[2][3 * i + j])
            
            # 리스트를 하나의 문자열로 변환 (쉼표로 구분)
            type_queries[step_key] = ", ".join(step_queries)

        results = []
        for i in range(0, len(query[1])):
            if i == 0:
                results = await relationships_vdb.query(type_queries["main_query"]+type_queries[f"step{i+1}_query"], top_k=60*(2**(len(query[1])-(i+1))))
                print(f"step{i+1}_filter_edge_query, top_k: {60*(2**(len(query[1])-(i+1)))}")
            else:
                results = await filter_edge_query(knowledge_graph_inst, results, type_queries["main_query"]+type_queries[f"step{i+1}_query"], top_k=60*(2**(len(query[1])-(i+1))))
                print(f"step{i+1}_filter_edge_query, top_k: {60*(2**(len(query[1])-(i+1)))}")

    edge_datas, edge_degree = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
        ),
        asyncio.gather(
            *[
                knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"])
                for r in results
            ]
        ),
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")

    if query_param.mode in ["ours", "ours1", "ours4", "ours5", "ours8"]:
        edge_datas = [
            {
                "src_id": k["src_id"],
                "tgt_id": k["tgt_id"],
                "rank": d,
                "created_at": k.get("__created_at__", None),  # 从 KV 存储中获取时间元数据
                **v,
            }
            for k, v, d in zip(results, edge_datas, edge_degree)
            if v is not None
        ]
    elif query_param.mode in ["ours2", "ours3", "ours6", "ours7"]:
        edge_datas = [
            {
                "src_id": k["src_id"],
                "tgt_id": k["tgt_id"],
                "type":k["type"],
                "rank": d,
                "created_at": k.get("__created_at__", None),  # 从 KV 存储中获取时间元数据
                **v,
            }
            for k, v, d in zip(results, edge_datas, edge_degree)
            if v is not None
        ]
    
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    use_entities, use_text_units = await asyncio.gather(
        _find_most_related_entities_from_relationships(
            edge_datas, query_param, knowledge_graph_inst
        ),
        _find_related_text_unit_from_relationships(
            edge_datas, query_param, text_chunks_db, knowledge_graph_inst
        ),
    )
    logger.info(
        f"Global query uses {len(use_entities)} entities, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )
    
    # Rerank entities, relations, and text units using cosine similarity
    if query_param.mode in ["ours", "ours1"]:
        edge_datas = await rerank_by_cosine_similarity(query, edge_datas, key="description")
        use_entities = await rerank_by_cosine_similarity(query, use_entities, key="description")
        use_text_units = await rerank_by_cosine_similarity(query, use_text_units, key="content")
    elif query_param.mode in ["ours2", "ours3"]:
        pass

    relations_section_list = [
        [
            "id",
            "source",
            "target",
            "description",
            "keywords",
            "weight",
            "rank",
            "created_at",
        ]
    ]
    for i, e in enumerate(edge_datas):
        created_at = e.get("created_at", "Unknown")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
                created_at,
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    node_datas, node_degrees = await asyncio.gather(
        asyncio.gather(
            *[
                knowledge_graph_inst.get_node(entity_name)
                for entity_name in entity_names
            ]
        ),
        asyncio.gather(
            *[
                knowledge_graph_inst.node_degree(entity_name)
                for entity_name in entity_names
            ]
        ),
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_units_lookup = {}

    async def fetch_chunk_data(c_id, index):
        if c_id not in all_text_units_lookup:
            chunk_data = await text_chunks_db.get_by_id(c_id)
            # Only store valid data
            if chunk_data is not None and "content" in chunk_data:
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                }

    tasks = []
    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            tasks.append(fetch_chunk_data(c_id, index))

    await asyncio.gather(*tasks)

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units


def combine_contexts(entities, relationships, sources):
    # Function to extract entities, relationships, and sources from context strings
    hl_entities, ll_entities = entities[0], entities[1]
    hl_relationships, ll_relationships = relationships[0], relationships[1]
    hl_sources, ll_sources = sources[0], sources[1]
    # Combine and deduplicate the entities
    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )

    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    return combined_entities, combined_relationships, combined_sources

def ours_combine_contexts(entities, relationships, sources):
    # Function to extract entities, relationships, and sources from context strings
    node_entities, edge_entities = entities[0], entities[1]
    node_relationships, edge_relationships = relationships[0], relationships[1]
    node_sources, edge_sources = sources[0], sources[1]
    # Combine and deduplicate the entities
    combined_entities = process_combine_contexts(node_entities, edge_entities)

    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        node_relationships, edge_relationships
    )

    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(node_sources, edge_sources)

    return combined_entities, combined_relationships, combined_sources


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
):
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]

    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    # Filter out invalid chunks
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

    if not valid_chunks:
        logger.warning("No valid chunks found after filtering")
        return PROMPTS["fail_response"]

    maybe_trun_chunks = truncate_list_by_token_size(
        valid_chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    if not maybe_trun_chunks:
        logger.warning("No chunks left after truncation")
        return PROMPTS["fail_response"]

    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "\n--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])

    if query_param.only_need_context:
        return section

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )
    
    cls = os.path.basename(global_config["working_dir"])
        
    if cls == "novelqa":
        option = get_options_by_query("/workspace/datasets/novelqa.json", query)
        sys_prompt_temp = PROMPTS["rag_response_novelqa"]
        sys_prompt = sys_prompt_temp.format(
            context_data=section,
            response_type=query_param.response_type,
            history=history_context,
            options=option,
        )
    elif cls == "infiniteqa":
        sys_prompt_temp = PROMPTS["rag_response_infiniteqa"]
        sys_prompt = sys_prompt_temp.format(
            context_data=section,
            response_type=query_param.response_type,
            history=history_context,
        )
    elif cls == "infinitechoice":
        option = get_options_by_query_infinite("/workspace/datasets/infinitechoice.jsonl", query)
        sys_prompt_temp = PROMPTS["rag_response_infinitechoice"]
        sys_prompt = sys_prompt_temp.format(
            context_data=section,
            response_type=query_param.response_type,
            history=history_context,
            options=option,
        )
    else:
        sys_prompt_temp = PROMPTS["naive_rag_response"]
        sys_prompt = sys_prompt_temp.format(
            content_data=section,
            response_type=query_param.response_type,
            history=history_context,
        )

    if query_param.only_need_prompt:
        return sys_prompt

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    
    print(response)

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )

    return response


async def kg_query_with_keywords(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:
    """
    Refactored kg_query that does NOT extract keywords by itself.
    It expects hl_keywords and ll_keywords to be set in query_param, or defaults to empty.
    Then it uses those to build context and produce a final LLM response.
    """

    # ---------------------------
    # 1) Handle potential cache for query results
    # ---------------------------
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # ---------------------------
    # 2) RETRIEVE KEYWORDS FROM query_param
    # ---------------------------

    # If these fields don't exist, default to empty lists/strings.
    hl_keywords = getattr(query_param, "hl_keywords", []) or []
    ll_keywords = getattr(query_param, "ll_keywords", []) or []

    # If neither has any keywords, you could handle that logic here.
    if not hl_keywords and not ll_keywords:
        logger.warning(
            "No keywords found in query_param. Could default to global mode or fail."
        )
        return PROMPTS["fail_response"]
    if not ll_keywords and query_param.mode in ["local", "hybrid"]:
        logger.warning("low_level_keywords is empty, switching to global mode.")
        query_param.mode = "global"
    if not hl_keywords and query_param.mode in ["global", "hybrid"]:
        logger.warning("high_level_keywords is empty, switching to local mode.")
        query_param.mode = "local"

    # Flatten low-level and high-level keywords if needed
    ll_keywords_flat = (
        [item for sublist in ll_keywords for item in sublist]
        if any(isinstance(i, list) for i in ll_keywords)
        else ll_keywords
    )
    hl_keywords_flat = (
        [item for sublist in hl_keywords for item in sublist]
        if any(isinstance(i, list) for i in hl_keywords)
        else hl_keywords
    )

    # Join the flattened lists
    ll_keywords_str = ", ".join(ll_keywords_flat) if ll_keywords_flat else ""
    hl_keywords_str = ", ".join(hl_keywords_flat) if hl_keywords_flat else ""

    keywords = [ll_keywords_str, hl_keywords_str]

    logger.info("Using %s mode for query processing", query_param.mode)

    # ---------------------------
    # 3) BUILD CONTEXT
    # ---------------------------
    context = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
    )
    if not context:
        return PROMPTS["fail_response"]

    # If only context is needed, return it
    if query_param.only_need_context:
        return context

    # ---------------------------
    # 4) BUILD THE SYSTEM PROMPT + CALL LLM
    # ---------------------------

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )
    return response