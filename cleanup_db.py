"""清理数据库中其他学院的结构化文档，重新导入新的"""
import chromadb
import init_db
from zhipu_api import create_embedding

client = chromadb.PersistentClient(path="./db")
col = client.get_or_create_collection(
    name="hunnu_school_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# 删除旧的结构化文档
result = col.get(where={"source": "structured_policy"}, include=["metadatas"])
old_ids = result.get("ids", [])
print(f"删除旧结构化文档: {len(old_ids)} 条")
if old_ids:
    col.delete(ids=old_ids)

# 重新添加新的结构化文档
added = 0
for i, doc in enumerate(init_db.ALL_DOCS):
    doc_id = f"policy_struct_{i}"
    try:
        embedding = create_embedding(doc["content"])
        col.add(
            ids=[doc_id],
            documents=[doc["content"]],
            embeddings=[embedding],
            metadatas=[{"title": doc["title"], "source": "structured_policy"}]
        )
        added += 1
    except Exception as e:
        print(f"  失败: {doc['title']} - {e}")

print(f"新增结构化文档: {added} 条")
print(f"数据库总数: {col.count()}")
