"""
RAG 检索模块 v3.0
===================
简化架构，提升可靠性：
  1. 向量检索为主，关键词过滤为辅
  2. 不用 rerank（额外API调用不稳定），用简单的分数融合
  3. 降低阈值，多返回内容，让Agent自己判断
  4. 返回完整文档内容，不截断
"""

import chromadb
import os
import re
from dotenv import load_dotenv
from zhipu_api import create_embedding, chat_completions, _get_api_key

load_dotenv()

CHROMA_DB_PATH = "./db"
COLLECTION_NAME = "hunnu_school_knowledge"
EMBEDDING_MODEL = "embedding-2"
SIM_THRESHOLD = 0.25  # 降低阈值，多召回
DEFAULT_TOP_K = 8     # 多返回一些，让Agent有足够上下文

_chroma_client = None
_collection = None


def get_zhipu_client():
    return None


def get_chroma_collection():
    global _chroma_client, _collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def reset_clients():
    global _chroma_client, _collection
    _chroma_client = None
    _collection = None


reset_zhipu_client = reset_clients


def get_embedding(text: str) -> list:
    return create_embedding(text, model=EMBEDDING_MODEL)


def get_embeddings_batch(texts: list) -> list:
    embeddings = []
    for i, text in enumerate(texts):
        vec = get_embedding(text)
        embeddings.append(vec)
        if (i + 1) % 50 == 0:
            print(f"   已生成 {i+1}/{len(texts)} 条向量")
    return embeddings


def rewrite_query(query: str, chat_history: list = None) -> str:
    """查询改写：有对话历史时才调用，消除指代"""
    if not chat_history or len(chat_history) < 2:
        return query

    history_text = ""
    recent = chat_history[-6:] if len(chat_history) > 6 else chat_history
    for msg in recent:
        role = "用户" if msg.get("role") == "user" else "助手"
        history_text += f"{role}：{msg.get('content', '')[:150]}\n"

    prompt = f"""根据对话历史，将用户最新问题改写成适合文档检索的独立查询语句。
要求：消除指代词，补充完整，只输出改写后的查询，不要解释。

对话历史：
{history_text}
用户最新问题：{query}

改写后的查询："""

    try:
        response = chat_completions(
            messages=[{"role": "user", "content": prompt}],
            model="glm-4-flash",
            temperature=0.1,
            max_tokens=80
        )
        rewritten = response["choices"][0]["message"]["content"].strip().strip('"').strip("'")
        return rewritten if rewritten else query
    except Exception:
        return query


def search_test(query: str, top_k: int = None, chat_history: list = None,
                use_rewrite: bool = True, use_rerank: bool = True) -> list:
    """
    主检索函数 - 向量检索 + 关键词加分
    简化版：不用 rerank API，用简单的分数融合
    """
    if top_k is None:
        top_k = DEFAULT_TOP_K

    # Step 1: 查询改写
    search_query = query
    if use_rewrite and chat_history:
        search_query = rewrite_query(query, chat_history)

    # Step 2: 向量检索（多召回）
    collection = get_chroma_collection()
    query_vec = get_embedding(search_query)
    result = collection.query(
        query_embeddings=[query_vec],
        n_results=max(top_k * 12, 50),  # 多召回候选，确保关键词重排有足够素材
        include=["documents", "metadatas", "distances"]
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    # Step 3: 关键词加分（增强版：标题匹配+内容匹配+短语匹配）
    stopwords = set("的 了 和 是 在 有 我 你 他 这 那 什么 怎么 如何 请问 吗 呢 啊 吧 要 有哪些 一下 可以 多少".split())
    words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', search_query)
    keywords = [w for w in words if w not in stopwords and len(w) > 1]

    results = []
    for idx in range(len(docs)):
        cos_sim = 1 - dists[idx]
        title = metas[idx].get("title", "")
        content_lower = docs[idx]

        # 1. 标题关键词匹配（权重高，因为标题代表文档主题）
        title_bonus = 0
        if keywords and title:
            title_match_count = sum(1 for kw in keywords if kw in title)
            title_bonus = title_match_count * 0.20  # 每个关键词加0.20
            if title_match_count == len(keywords):
                title_bonus += 0.15  # 全部命中额外加0.15
            if title_match_count >= 2:
                title_bonus += 0.1  # 多关键词命中再加0.1

        # 2. 内容关键词匹配
        kw_bonus = 0
        if keywords:
            match_count = sum(1 for kw in keywords if kw in content_lower)
            kw_bonus = match_count * 0.04

        # 3. 整句/短语匹配（当查询短语出现在内容中时加分）
        phrase_bonus = 0
        if len(search_query) >= 3 and search_query in content_lower:
            phrase_bonus = 0.08
        if title and len(search_query) >= 3 and search_query in title:
            phrase_bonus += 0.15

        final_score = cos_sim + title_bonus + kw_bonus + phrase_bonus

        if final_score >= SIM_THRESHOLD:
            results.append({
                "content": docs[idx],
                "title": title,
                "score": round(final_score, 4),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def search_simple(query: str, top_k: int = 5) -> list:
    return search_test(query, top_k=top_k, use_rewrite=False, use_rerank=False)


def get_collection_count() -> int:
    try:
        return get_chroma_collection().count()
    except Exception:
        return 0


if __name__ == "__main__":
    print("=" * 60)
    print("RAG 检索模块 v3.0 测试")
    print("=" * 60)
    test_queries = [
        "奖学金评定条件是什么？",
        "考试作弊会怎么样？",
        "挂科了怎么办",
        "学校有哪些专业？",
    ]
    for q in test_queries:
        print(f"\n🔍 查询：{q}")
        results = search_test(q, top_k=3, use_rewrite=False)
        print(f"   返回 {len(results)} 条结果：")
        for i, r in enumerate(results):
            print(f"   [{i+1}] {r['title']} (score: {r['score']})")
            print(f"        {r['content'][:80]}...")
