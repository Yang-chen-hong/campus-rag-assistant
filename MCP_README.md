# 校园智能助手 MCP 服务使用说明

> **队员E（MCP编写 + 主讲）负责模块**

## 什么是 MCP？

MCP（Model Context Protocol）是一种开放协议，让 AI 模型可以安全地与外部工具和数据源交互。

把校园助手封装成 MCP 服务后，可以在**任何支持 MCP 的客户端**中调用，比如：
- Claude Desktop
- Cursor IDE
- Windsurf
- Cline
- 以及所有支持 MCP 的 AI 工具

## 📦 提供的 MCP 工具

| 工具名称 | 类型 | 说明 |
|---------|------|------|
| `campus_rag_query` | 核心工具 | 完整的 RAG 问答，带思考过程和引用来源 |
| `campus_knowledge_search` | 检索工具 | 纯向量检索，只返回相关文档片段 |
| `query_scholarship` | 业务工具 | 奖学金评定资格查询 |
| `query_discipline` | 业务工具 | 学校处分规定查询 |
| `get_policy_time` | 业务工具 | 政策文件修订时间查询 |

## 📚 提供的 MCP 资源

| 资源 URI | 说明 |
|---------|------|
| `knowledge://overview` | 知识库概览（数据量、覆盖范围、使用说明） |

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install mcp
```

### 2. 配置 API Key

设置环境变量 `ZHIPU_API_KEY` 为你的智谱 API Key，或在 MCP 配置的 `env` 中指定。

### 3. 配置到 Claude Desktop

在 Claude Desktop 的配置文件中添加：

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "hunnu-campus-assistant": {
      "command": "python",
      "args": [
        "你的项目路径\\mcp_server.py"
      ],
      "env": {
        "ZHIPU_API_KEY": "你的智谱API_KEY"
      }
    }
  }
}
```

> 注意：把 `你的项目路径` 替换为实际的项目绝对路径，
> 把 `你的智谱API_KEY` 替换为你的真实 API Key。

### 4. 重启 Claude Desktop

重启后，在对话中问关于湖南师大的问题，Claude 会自动调用校园助手工具。

## 🔧 配置到其他客户端

### Cursor IDE

1. 打开 Cursor Settings → Features → MCP
2. 点击 "Add MCP Server"
3. 选择 "Stdio" 模式
4. 填入命令和参数：
   - Command: `python`
   - Args: `你的项目路径/mcp_server.py`
5. 添加环境变量 `ZHIPU_API_KEY`

### Windsurf

1. 打开 Settings → MCP
2. 添加新的 MCP Server
3. 配置方式同上

## 📖 使用示例

### 示例1：政策查询

**用户问**："奖学金怎么评？"

Claude 会调用 `campus_rag_query` 工具，传入问题，获得：
- 意图分类结果（policy）
- 检索到的相关文档
- 完整的回答
- 思考过程

### 示例2：业务工具直接调用

**用户问**："我绩点3.6，大二，没处分，能评一等奖学金吗？"

Claude 可以直接调用 `query_scholarship` 工具，快速得到评定结果。

### 示例3：知识库检索

**用户问**："帮我找关于考试作弊的所有规定"

Claude 可以调用 `campus_knowledge_search` 工具，获取所有相关文档片段。

## 🏗️ 架构说明

```
┌─────────────────────────────────────────────────────┐
│              MCP 客户端（Claude/Cursor 等）           │
└────────────────────────┬────────────────────────────┘
                         │ MCP 协议（stdio）
                         ▼
┌─────────────────────────────────────────────────────┐
│           MCP 服务器（mcp_server.py）                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  RAG 问答   │  │ 向量检索    │  │ 业务工具    │ │
│  │ (Agent)     │  │ (Retriever) │  │ (Skills)    │ │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘ │
└─────────┼────────────────┼──────────────────────────┘
          │                │
          ▼                ▼
   ┌─────────────┐  ┌─────────────┐
   │  智谱大模型  │  │ ChromaDB    │
   │  (glm-4)    │  │ 向量库      │
   └─────────────┘  └─────────────┘
```

## 🎯 队员E的工作内容

作为队员E（MCP编写 + 主讲），你负责的工作包括：

### 已完成 ✅
1. **MCP 服务器封装** - 将整个 Agent 系统封装为标准 MCP 服务
2. **5 个 MCP 工具** - 涵盖核心问答、检索、业务工具
3. **1 个 MCP 资源** - 知识库概览信息
4. **配置文件示例** - 方便快速接入各种客户端
5. **使用文档** - 详细的接入和使用说明

### 演示要点（主讲用）
1. 介绍 MCP 是什么，为什么要用 MCP
2. 演示在 Claude Desktop 中配置 MCP 服务
3. 演示校园助手的各种工具调用效果
4. 讲解 MCP 的架构和实现原理
5. 展示如何扩展更多 MCP 工具

## 🔍 调试方法

### 测试 MCP 服务器是否能启动

```bash
python mcp_server.py
```

如果启动成功，会看到 MCP 的 JSON-RPC 通信输出。

### 查看 MCP 工具列表

在支持 MCP Inspector 的工具中，可以看到所有可用的工具和资源。

## 📝 扩展开发

### 添加新的 MCP 工具

在 `mcp_server.py` 中添加：

```python
@mcp.tool()
def your_new_tool(param1: str, param2: int) -> str:
    """
    工具描述（会被 AI 看到，用来决定何时调用）。

    参数：
      param1: 参数1说明
      param2: 参数2说明

    返回：
      返回值说明
    """
    # 你的工具逻辑
    return "结果"
```

### 添加新的 MCP 资源

```python
@mcp.resource("your://resource/uri")
def your_resource() -> str:
    """资源描述"""
    return "资源内容"
```

## 📄 相关文件

- `mcp_server.py` - MCP 服务器主程序
- `mcp-config.example.json` - MCP 配置文件示例
- `MCP_README.md` - 本文档
