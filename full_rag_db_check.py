import chromadb
from sentence_transformers import SentenceTransformer

# 和7.py配置完全对齐
LOCAL_BGE_PATH = "./models/bge-m3"
CHROMA_DB_PATH = "./db"
COLLECTION_NAME = "hunnu_school_knowledge"
COS_THRESHOLD = 0.6

# 加载本地离线嵌入模型
print("===== 加载本地BGE-M3向量模型 =====")
bge_model = SentenceTransformer(LOCAL_BGE_PATH)
# 连接余弦空间向量库
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
coll = client.get_collection(COLLECTION_NAME)

def base_database_check():
    """基础完整性校验：数量、向量维度、抽样文本"""
    print("\n===== 一、向量库基础完整性校验 =====")
    total = coll.count()
    print(f"向量库存储总分片：{total} 条")
    if total == 3521:
        print("✅ 数据数量匹配，无丢失")
    else:
        print(f"❌ 数据异常！预期3521条，实际{total}条")

    # 随机抽取3条校验向量维度与文本
    sample = coll.get(limit=3, include=["documents", "metadatas", "embeddings"])
    vec_dim = len(sample["embeddings"][0])
    print(f"向量标准维度：{vec_dim}")
    if vec_dim == 1024:
        print("✅ 向量维度统一为1024，无损坏")
    else:
        print("❌ 向量维度异常")

    print("\n随机3条样本预览：")
    for i in range(3):
        print(f"\n【样本{i+1}】标题：{sample['metadatas'][i]['title']}")
        print(f"文本片段：{sample['documents'][i][:300]}...")
    print("-" * 70)

def multi_scene_retrieval_test():
    """6大类校园场景批量检索测试，统计匹配质量"""
    test_questions = [
        "统计学科建设座谈会专家建议",
        "学校创新创业挑战杯竞赛获奖名单",
        "数学与统计学院学术讲座安排",
        "学院师德师风专题培训活动",
        "研究生招生复试工作通知",
        "省部级科研项目申报培训会"
    ]
    total_result_count = 0
    high_match_count = 0

    print("===== 二、多场景检索效果测试（6类校园问题） =====")
    for idx, query in enumerate(test_questions):
        print(f"\n【场景{idx+1}提问】{query}")
        query_vec = bge_model.encode(query, normalize_embeddings=True).tolist()
        res = coll.query(
            query_embeddings=[query_vec],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        for i in range(len(docs)):
            dist = dists[i]
            cos_sim = 1 - dist
            total_result_count += 1
            if cos_sim >= COS_THRESHOLD:
                high_match_count += 1
                tag = "✅高匹配"
                print(f"  {tag} 相似度{cos_sim:.4f} | 标题：{metas[i]['title']}")
            else:
                tag = "❌低匹配"
                print(f"  {tag} 相似度{cos_sim:.4f} | 标题：{metas[i]['title']}")
        print("-" * 60)

    # 整体质量汇总
    print("===== 三、知识库整体效果汇总报告 =====")
    print(f"本次测试总召回片段：{total_result_count} 条")
    print(f"高匹配有效片段(余弦≥0.6)：{high_match_count} 条")
    high_rate = high_match_count / total_result_count if total_result_count > 0 else 0
    print(f"整体高匹配召回率：{high_rate:.2%}")

    # 质量判定标准
    if high_rate >= 0.85:
        print("🌟 结论：向量库质量优秀，分片、向量化、检索全部达标，可作为最终交付成果")
    elif high_rate >= 0.65:
        print("⚠️ 结论：向量库基本可用，少量场景召回精度一般，可按需优化分片规则")
    else:
        print("❌ 结论：向量库检索效果较差，需要回头调整文本分片与过滤逻辑")

if __name__ == "__main__":
    base_database_check()
    multi_scene_retrieval_test()