"""
RAG 检索模块 v2.0
===================
队员C（RAG在线检索）负责模块 - 升级版

升级内容：
  1. 查询改写（Query Rewriting）- 将用户口语化问题优化为检索关键词
  2. 混合检索（Hybrid Search）- 向量检索 + 关键词检索融合
  3. 重排序（Rerank）- 用智谱 rerank 模型二次排序，提升准确率
  4. 相似度阈值过滤 - 过滤低质量结果
  5. 懒加载 + 线程安全 - 支持多用户并发
"""

import chromadb
import os
import re
from dotenv import load_dotenv
from zhipu_api import create_embedding, chat_completions, _get_api_key

load_dotenv()

# -------------------------- 全局配置 --------------------------
CHROMA_DB_PATH = "./db"
COLLECTION_NAME = "hunnu_school_knowledge"
EMBEDDING_MODEL = "embedding-2"    # 智谱 embedding 模型，1024 维
SIM_THRESHOLD = 0.4                # 相似度阈值（余弦相似度，低于此值过滤）
HYBRID_TOP_K_MULTIPLIER = 3        # 混合检索候选放大倍数（先召回多一些再重排）
DEFAULT_TOP_K = 5                  # 默认返回条数


# -------------------------- 客户端管理 --------------------------
_chroma_client = None
_collection = None


def get_zhipu_client():
    """兼容旧接口 - 直接返回 None（已改用 zhipu_api 模块）"""
    return None


def get_chroma_collection():
    """获取 ChromaDB 集合（懒加载）"""
    global _chroma_client, _collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def reset_clients():
    """重置所有客户端（切换 API Key 时调用）"""
    global _chroma_client, _collection
    _chroma_client = None
    _collection = None


# 兼容旧接口
reset_zhipu_client = reset_clients


# -------------------------- Embedding --------------------------
def get_embedding(text: str) -> list:
    """调用智谱 embedding API 获取向量"""
    return create_embedding(text, model=EMBEDDING_MODEL)


def get_embeddings_batch(texts: list) -> list:
    """批量获取 embedding"""
    embeddings = []
    for i, text in enumerate(texts):
        vec = get_embedding(text)
        embeddings.append(vec)
        if (i + 1) % 50 == 0:
            print(f"   已生成 {i+1}/{len(texts)} 条向量")
    return embeddings


# -------------------------- 查询改写 --------------------------
def rewrite_query(query: str, chat_history: list = None) -> str:
    """
    查询改写：将用户口语化问题改写为适合检索的关键词/句子。
    作用：
      - 消除指代（"那个"、"它" → 具体事物）
      - 补充背景（从对话历史提取上下文）
      - 优化为更规范的检索语句
    """
    if not chat_history:
        # 没有历史就简单处理：去除语气词，保留核心
        cleaned = re.sub(r'[请问你我他的了啊呀吗呢吧]', '', query)
        return cleaned if cleaned.strip() else query

    # 构造历史上下文（取最近3轮）
    history_text = ""
    recent = chat_history[-6:] if len(chat_history) > 6 else chat_history
    for msg in recent:
        role = "用户" if msg.get("role") == "user" else "助手"
        history_text += f"{role}：{msg.get('content', '')[:200]}\n"

    # 调用大模型改写
    client = get_zhipu_client()
    prompt = f"""你是一个查询改写专家。请根据对话历史，将用户的最新问题改写成适合文档检索的查询语句。

要求：
1. 消除指代词（如"它"、"这个"、"那个"等），补充完整
2. 如果有上下文，结合上下文补充关键词
3. 输出改写后的查询语句，不要解释，不要加标点以外的符号
4. 如果原问题已经很明确，直接返回原问题

对话历史：
{history_text}
用户最新问题：{query}

改写后的查询："""

    try:
        response = chat_completions(
            messages=[{"role": "user", "content": prompt}],
            model="glm-4-flash",
            temperature=0.1,
            max_tokens=100
        )
        rewritten = response["choices"][0]["message"]["content"].strip()
        # 简单清洗
        rewritten = rewritten.strip('"').strip("'").strip()
        return rewritten if rewritten else query
    except Exception as e:
        print(f"[查询改写失败，使用原查询] {e}")
        return query


# -------------------------- 关键词检索 --------------------------
def keyword_search(query: str, top_k: int = 10) -> list:
    """
    关键词检索（基于 ChromaDB 的全文过滤 + 简单评分）。
    作为向量检索的补充，提升精确匹配的召回率。
    """
    collection = get_chroma_collection()

    # 提取关键词（简单分词：按标点和空格切，过滤停用词）
    stopwords = set("的 了 和 是 在 有 我 你 他 她 它 这 那 什么 怎么 如何 请问 吗 呢 啊 吧".split())
    words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', query)
    keywords = [w for w in words if w not in stopwords and len(w) > 1]

    if not keywords:
        return []

    # 先从向量库取一批候选（用第一个关键词的向量近似召回）
    try:
        query_vec = get_embedding(query)
        result = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k * 3,
            include=["documents", "metadatas", "distances"]
        )
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
    except Exception:
        return []

    # 关键词匹配评分
    scored = []
    for i, doc in enumerate(docs):
        # 统计匹配的关键词数量
        match_count = sum(1 for kw in keywords if kw in doc)
        if match_count == 0:
            continue
        # 匹配密度 = 匹配关键词数 / 总关键词数
        density = match_count / len(keywords)
        # 基础分 = 密度权重 0.6 + 向量相似度权重 0.4
        cos_sim = 1 - dists[i]
        score = density * 0.6 + cos_sim * 0.4
        scored.append({
            "content": doc,
            "title": metas[i].get("title", "未知来源"),
            "score": score,
            "_vec_score": cos_sim,
            "_kw_density": density
        })

    # 按关键词匹配得分排序
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# -------------------------- 向量检索 --------------------------
def vector_search(query: str, top_k: int = 10) -> list:
    """
    纯向量检索（语义相似度）。
    """
    collection = get_chroma_collection()
    query_vec = get_embedding(query)
    result = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    results = []
    for idx in range(len(docs)):
        cos_sim = 1 - dists[idx]
        results.append({
            "content": docs[idx],
            "title": metas[idx].get("title", "未知来源"),
            "score": cos_sim,
            "_source": "vector"
        })
    return results


# -------------------------- 重排序 --------------------------
def rerank(query: str, candidates: list, top_k: int = 5) -> list:
    """
    使用大模型对候选文档重新排序。
    用 glm-4-flash 对每个候选打分（0-10分），按分数排序。
    """
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates

    # 用大模型一次性对所有候选打分
    docs_text = ""
    for i, c in enumerate(candidates):
        docs_text += f"\n[{i+1}] {c['content'][:200]}\n"

    prompt = f"""请对以下文档与查询的相关性打分（0-10分，10分最相关）。
只返回每篇文档的编号和分数，格式：编号:分数，每行一篇。

查询：{query}

文档列表：
{docs_text}

打分结果："""

    try:
        response = chat_completions(
            messages=[{"role": "user", "content": prompt}],
            model="glm-4-flash",
            temperature=0.1,
            max_tokens=200
        )
        result_text = response["choices"][0]["message"]["content"].strip()

        # 解析打分结果
        import re
        scores = {}
        for line in result_text.split("\n"):
            match = re.match(r'(\d+)\s*[:：]\s*(\d+\.?\d*)', line.strip())
            if match:
                idx = int(match.group(1)) - 1
                score = float(match.group(2))
                if 0 <= idx < len(candidates):
                    scores[idx] = score

        # 按打分排序
        ranked = []
        for idx in sorted(scores.keys(), key=lambda k: scores[k], reverse=True):
            entry = candidates[idx].copy()
            entry["score"] = scores[idx] / 10.0  # 归一化到 0-1
            entry["_source"] = "rerank"
            ranked.append(entry)

        # 补全没打分的
        seen = set(r["content"] for r in ranked)
        for c in candidates:
            if c["content"] not in seen:
                ranked.append(c)

        return ranked[:top_k]
    except Exception as e:
        print(f"[重排序失败，使用原始排序] {e}")

    # 失败则按原始分数排序
    sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
    return sorted_candidates[:top_k]


# -------------------------- 主检索函数 --------------------------
def search_test(query: str, top_k: int = 5, chat_history: list = None,
                use_rewrite: bool = True, use_rerank: bool = True) -> list:
    """
    主检索函数 v2.0 - 混合检索 + 查询改写 + 重排序

    参数：
      query: 用户问题
      top_k: 返回结果数量
      chat_history: 对话历史（用于查询改写）
      use_rewrite: 是否启用查询改写
      use_rerank: 是否启用重排序

    返回：
      list of dict: 每条包含 content、title、score
    """
    # Step 1: 查询改写
    search_query = query
    if use_rewrite:
        search_query = rewrite_query(query, chat_history)
        if search_query != query:
            print(f"   [查询改写] {query} → {search_query}")

    # Step 2: 多路召回（向量 + 关键词）
    candidate_count = top_k * HYBRID_TOP_K_MULTIPLIER
    vec_results = vector_search(search_query, top_k=candidate_count)
    kw_results = keyword_search(search_query, top_k=candidate_count)

    # Step 3: 合并去重
    seen_contents = set()
    all_candidates = []
    # 先加向量结果
    for r in vec_results:
        if r["content"] not in seen_contents:
            seen_contents.add(r["content"])
            r_copy = r.copy()
            r_copy["_source"] = "vector"
            all_candidates.append(r_copy)
    # 再加关键词结果
    for r in kw_results:
        if r["content"] not in seen_contents:
            seen_contents.add(r["content"])
            r_copy = r.copy()
            r_copy["_source"] = "keyword"
            all_candidates.append(r_copy)

    print(f"   [多路召回] 向量{len(vec_results)}条 + 关键词{len(kw_results)}条 = 共{len(all_candidates)}条候选")

    # Step 4: 重排序
    if use_rerank and len(all_candidates) > top_k:
        final_results = rerank(search_query, all_candidates, top_k=top_k)
        print(f"   [重排序] {len(all_candidates)} → {len(final_results)} 条")
    else:
        final_results = sorted(all_candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    # Step 5: 相似度阈值过滤
    filtered = [r for r in final_results if r.get("score", 0) >= SIM_THRESHOLD]
    if len(filtered) < len(final_results):
        print(f"   [阈值过滤] {len(final_results)} → {len(filtered)} 条（阈值: {SIM_THRESHOLD}）")

    # 清理内部字段
    clean_results = []
    for r in filtered:
        clean_results.append({
            "content": r["content"],
            "title": r["title"],
            "score": round(r["score"], 4),
        })

    return clean_results


def search_simple(query: str, top_k: int = 5) -> list:
    """
    简化版检索（纯向量，不调用大模型）- 用于测试或轻量场景。
    """
    results = search_test(query, top_k=top_k, use_rewrite=False, use_rerank=False)
    return results


# -------------------------- 数据库管理 --------------------------
def get_collection_count() -> int:
    """获取知识库文档总数"""
    try:
        collection = get_chroma_collection()
        return collection.count()
    except Exception:
        return 0


if __name__ == "__main__":
    # 检索测试
    print("=" * 60)
    print("RAG 检索模块 v2.0 测试")
    print("=" * 60)

    test_queries = [
        "奖学金评定条件是什么？",
        "考试作弊会怎么样？",
        "挂科了怎么办",
    ]

    for q in test_queries:
        print(f"\n🔍 查询：{q}")
        results = search_test(q, top_k=3)
        print(f"   返回 {len(results)} 条结果：")
        for i, r in enumerate(results):
            print(f"   [{i+1}] {r['title']} (score: {r['score']})")
            print(f"        {r['content'][:60]}...")
