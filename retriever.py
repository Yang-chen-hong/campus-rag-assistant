import jsonlines
import chromadb
import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()

# -------------------------- 全局配置 --------------------------
INPUT_VECTOR_FILE = "rag_output/chunk_with_vector.jsonl"
CHROMA_DB_PATH = "./db"
COLLECTION_NAME = "hunnu_school_knowledge"
SIM_THRESHOLD = 0.0  # 低于该相似度直接过滤
EMBEDDING_MODEL = "embedding-2"  # 智谱 embedding 模型，1024 维


def _get_api_key() -> str:
    """获取 API Key：优先环境变量，其次 Streamlit Secrets"""
    # 1. 从环境变量获取（本地开发）
    api_key = os.getenv("ZHIPU_API_KEY")
    if api_key:
        return api_key
    # 2. 从 Streamlit Secrets 获取（云端部署）
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "ZHIPU_API_KEY" in st.secrets:
            return st.secrets["ZHIPU_API_KEY"]
    except Exception:
        pass
    raise ValueError("未找到 ZHIPU_API_KEY，请在环境变量或 Streamlit Secrets 中配置")


# 懒加载智谱客户端
_zhipu_client = None


def get_zhipu_client() -> ZhipuAI:
    """获取智谱客户端（懒加载）"""
    global _zhipu_client
    if _zhipu_client is None:
        _zhipu_client = ZhipuAI(api_key=_get_api_key())
    return _zhipu_client

# 初始化向量库客户端，指定余弦距离空间
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
# 创建/获取集合，使用余弦相似度
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


def get_embedding(text: str) -> list:
    """调用智谱 embedding API 获取向量"""
    client = get_zhipu_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list) -> list:
    """批量获取 embedding（逐个调用，避免超限）"""
    embeddings = []
    for i, text in enumerate(texts):
        vec = get_embedding(text)
        embeddings.append(vec)
        if (i + 1) % 50 == 0:
            print(f"   已生成 {i+1}/{len(texts)} 条向量")
    return embeddings


# 批量入库函数：从 jsonl 文件读取文本，用智谱 API 生成向量，写入 ChromaDB
def load_all_chunk_to_chroma(batch_size: int = 50):
    """
    从原始分片数据重建向量库
    batch_size: 每批处理的条数（智谱 embedding API 单条调用）
    """
    batch_ids = []
    batch_embeddings = []
    batch_docs = []
    batch_meta = []
    total_count = 0

    print("===== 开始读取分片文件并重建向量库 =====")
    print(f"使用模型：{EMBEDDING_MODEL}（智谱在线 API）")

    # 先清空旧集合
    try:
        client.delete_collection(COLLECTION_NAME)
        print("已清空旧向量库")
    except Exception:
        pass
    global collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    with jsonlines.open(INPUT_VECTOR_FILE, "r") as reader:
        all_items = list(reader)
        total_items = len(all_items)
        print(f"共 {total_items} 条分片待处理")

        for item in all_items:
            total_count += 1
            current_id = f"id_{total_count}"

            # 用智谱 API 生成向量
            vec = get_embedding(item["chunk_text"])

            batch_ids.append(current_id)
            batch_embeddings.append(vec)
            batch_docs.append(item["chunk_text"])
            meta_info = {
                "title": item["source_title"],
                "source_url": item.get("source_url", "")
            }
            batch_meta.append(meta_info)

            # 满 batch_size 条执行一次入库
            if len(batch_ids) >= batch_size:
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_docs,
                    metadatas=batch_meta
                )
                print(f"已完成入库 {total_count}/{total_items} 条分片")
                batch_ids.clear()
                batch_embeddings.clear()
                batch_docs.clear()
                batch_meta.clear()

        # 写入最后不足 batch_size 条的剩余数据
        if len(batch_ids) > 0:
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_docs,
                metadatas=batch_meta
            )

    final_total = collection.count()
    print(f"\n✅ 入库操作结束！库内现有总数据量：{final_total} 条")


# 检索函数：用智谱 API 编码查询，从 ChromaDB 检索
def search_test(query: str, top_k: int = 5):
    print(f"🔍 检索：{query}")
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
        if cos_sim < SIM_THRESHOLD:
            continue
        results.append({
            "content": docs[idx],
            "title": metas[idx]["title"],
            "score": cos_sim
        })

    print(f"   → 找到 {len(results)} 条高匹配结果")
    return results


if __name__ == "__main__":
    # 首次部署前执行：重建向量库
    # load_all_chunk_to_chroma()

    # 检索测试
    search_test("奖学金评定条件")
