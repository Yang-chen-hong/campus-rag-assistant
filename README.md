# 🎓 湖南师范大学校园智能问答助手

基于 **LangChain + LangGraph + ChromaDB + Streamlit** 构建的校园 RAG 智能问答系统，支持多轮对话记忆、引用溯源和业务 Skills 工具调用。

## ✨ 功能特性

- 📚 **RAG 检索增强** — 基于 3521 条校园文档分片，通过智谱 embedding 向量检索实现精准召回
- 🧠 **Agent 自主决策** — 基于 LangGraph 构建，支持意图识别、分支路由、多轮对话
- 🔗 **引用溯源** — 每个回答都标注来源，支持「来源：编号」和「来源：Skill」双重溯源
- 🛠️ **Skills 工具调用** — 集成奖学金查询、处分查询、政策时效查询等业务工具
- 💬 **多轮对话记忆** — 支持上下文理解，能够记住历史对话并正确理解代词指代
- 🌐 **Web 交互界面** — 基于 Streamlit 构建，支持思考过程可视化

## 🛠️ 技术栈

| 组件 | 技术 |
| :--- | :--- |
| 大模型 | 智谱 GLM-4-Flash |
| Agent 框架 | LangChain + LangGraph |
| 向量数据库 | ChromaDB |
| 嵌入模型 | 智谱 embedding-2（1024 维） |
| Web 界面 | Streamlit |
| 编程语言 | Python 3.10+ |

## 🚀 本地运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```properties
ZHIPU_API_KEY="你的智谱API_Key"
```

### 3. 准备向量库

向量库已预置在 `db/` 目录中。如需重建：

```bash
python rebuild_db.py
```

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可使用。

## ☁️ 部署到 Streamlit Community Cloud

1. 将代码推送到 GitHub 仓库
2. 访问 [share.streamlit.io](https://share.streamlit.io)
3. 点击 "New app"，选择你的 GitHub 仓库
4. Main file path 选择 `app.py`
5. 在 Advanced settings → Secrets 中添加：
   ```
   ZHIPU_API_KEY = "你的智谱API_Key"
   ```
6. 点击 "Deploy" 等待部署完成

## 📁 项目结构

```
campus_rag/
├── app.py                  # Streamlit Web 界面
├── agent_graph.py          # Agent 核心逻辑（LangGraph）
├── retriever.py            # 检索模块（智谱 embedding + ChromaDB）
├── skills_tools.py         # Skills 工具集
├── rebuild_db.py           # 向量库重建脚本
├── db/                     # Chroma 向量数据库
├── requirements.txt        # Python 依赖
├── .env                    # 环境变量（API Key）
└── README.md               # 项目说明
```

## 📊 数据统计

| 指标 | 数量 |
| :--- | :--- |
| 文档分片总数 | 3521 条 |
| 向量维度 | 1024 |
| 相似度空间 | Cosine（余弦） |
| 检索 Top-K | 5 |

## 📄 License

MIT License
