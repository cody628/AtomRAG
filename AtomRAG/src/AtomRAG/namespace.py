from typing import Iterable


class NameSpace:
    KV_STORE_FULL_DOCS = "full_docs"
    KV_STORE_TEXT_CHUNKS = "text_chunks"
    KV_STORE_TEXT_ATOMICS = "text_atomics"
    KV_STORE_TEXT_TRIPLES = "text_triples"
    KV_STORE_LLM_RESPONSE_CACHE = "llm_response_cache"

    atomic_VECTOR_STORE_ENTITIES = "atomic_entities"
    atomic_VECTOR_STORE_RELATIONSHIPS = "atomic_relationships"
    triple_VECTOR_STORE_ENTITIES = "triple_entities"
    triple_VECTOR_STORE_RELATIONSHIPS = "triple_relationships"
    VECTOR_STORE_CHUNKS = "chunks"

    GRAPH_STORE_atomic_ENTITY_RELATION = "atomic_entity_relation"
    GRAPH_STORE_triple_ENTITY_RELATION = "triple_entity_relation"

    DOC_STATUS = "doc_status"


def make_namespace(prefix: str, base_namespace: str):
    return prefix + base_namespace


def is_namespace(namespace: str, base_namespace: str | Iterable[str]):
    if isinstance(base_namespace, str):
        return namespace.endswith(base_namespace)
    return any(is_namespace(namespace, ns) for ns in base_namespace)
