"""
导出数据库所有文档为纯文本JSON，便于在Streamlit Cloud重建数据库
"""
import chromadb
import json
import os

DB_PATH = "./db"
COLLECTION_NAME = "hunnu_school_knowledge"
OUTPUT_FILE = "./db_export/docs_export.json"


def export_all():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    total = col.count()
    print(f"总条数: {total}")

    # 分批获取全部数据
    batch_size = 1000
    all_docs = []

    for offset in range(0, total, batch_size):
        limit = min(batch_size, total - offset)
        result = col.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"]
        )
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])

        for i in range(len(ids)):
            all_docs.append({
                "id": ids[i],
                "content": docs[i] if i < len(docs) else "",
                "title": metas[i].get("title", "") if i < len(metas) else "",
                "source": metas[i].get("source", "unknown") if i < len(metas) else "unknown",
            })

        print(f"  已导出 {offset + len(ids)} / {total}")

    # 写入JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False)

    print(f"\n导出完成: {len(all_docs)} 条 -> {OUTPUT_FILE}")
    size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    print(f"文件大小: {size_mb:.2f} MB")

    # 按来源统计
    from collections import Counter
    sources = Counter(d["source"] for d in all_docs)
    print("\n按来源统计:")
    for k, v in sources.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    export_all()
