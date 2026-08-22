"""
重建向量库脚本
用智谱 embedding-2 替换 BGE-M3 向量
运行: python rebuild_db.py
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from retriever import load_all_chunk_to_chroma, collection

if __name__ == "__main__":
    start_time = time.time()
    print("=" * 50)
    print("开始重建向量库（智谱 embedding-2）")
    print("=" * 50)

    try:
        load_all_chunk_to_chroma(batch_size=50)
        elapsed = time.time() - start_time
        print(f"\n⏱️  总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
        print(f"📊 最终文档数: {collection.count()} 条")
        print("✅ 向量库重建完成！")
    except Exception as e:
        print(f"❌ 出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
