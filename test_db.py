import chromadb

client = chromadb.PersistentClient(path='./db')

# 查看所有 collection 名字
print("所有 collection:", client.list_collections())

# 尝试连接指定的 collection
try:
    collection = client.get_collection('hunnu_school_knowledge')
    print("总条数:", collection.count())
    results = collection.peek(3)
    print("样本内容:", results['documents'])
    print("样本标题:", results['metadatas'])
except Exception as e:
    print("出错:", e)