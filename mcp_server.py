"""
湖南师范大学校园智能助手 - MCP 服务器
==========================================
队员E（MCP编写 + 主讲）负责模块

将校园 RAG Agent 封装为标准 MCP（Model Context Protocol）服务，
支持在任何兼容 MCP 的客户端中调用（Claude Desktop、Cursor、Windsurf 等）。

MCP 工具列表：
  1. campus_rag_query        - 完整的 RAG 问答（带思考过程）
  2. campus_knowledge_search - 纯向量检索（只返回相关文档）
  3. query_scholarship       - 查询奖学金评定资格
  4. query_discipline        - 查询学校处分规定
  5. get_policy_time         - 查询政策文件修订时间

MCP 资源列表：
  1. knowledge://overview    - 知识库概览

使用方式：
  python mcp_server.py
  （通过 stdio 模式与 MCP 客户端通信）
"""

import os
import sys
from typing import Any

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(__file__))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    TextResourceContents,
    CallToolResult,
    CallToolRequest,
    ListToolsResult,
    ListToolsRequest,
    Resource,
    ReadResourceResult,
    ReadResourceRequest,
    ListResourcesResult,
    ListResourcesRequest,
)


# ========== 初始化 MCP 服务器 ==========
server = Server(
    name="hunnu-campus-assistant",
    version="1.0.0",
)


# ========== 工具定义 ==========
TOOLS = [
    {
        "name": "campus_rag_query",
        "description": "【核心工具】湖南师范大学校园智能助手 - 完整问答。基于RAG（检索增强生成）技术，结合学校知识库回答问题。支持：政策制度、专业介绍、新闻通知、日常咨询等。",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "用户的问题，用中文描述"
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "campus_knowledge_search",
        "description": "【检索工具】从校园知识库中搜索相关文档片段。只返回检索结果，不调用大模型生成回答。适合需要自己处理检索结果的场景。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或问题"
                },
                "top_k": {
                    "type": "number",
                    "description": "返回结果数量，默认5条",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "query_scholarship",
        "description": "查询奖学金评定资格。根据年级、绩点和处分次数，判断学生符合哪类奖学金申请条件。",
        "input_schema": {
            "type": "object",
            "properties": {
                "grade": {
                    "type": "string",
                    "description": "年级，如大一、大二、大三、大四"
                },
                "gpa": {
                    "type": "number",
                    "description": "绩点，如3.5（满分一般为4.0）"
                },
                "punishment_count": {
                    "type": "number",
                    "description": "处分次数，如0（无处分）"
                }
            },
            "required": ["grade", "gpa", "punishment_count"]
        }
    },
    {
        "name": "query_discipline",
        "description": "查询学校处分相关规定。支持查询：挂科、作弊、旷课、考试、处分 等相关规定。",
        "input_schema": {
            "type": "object",
            "properties": {
                "rule_type": {
                    "type": "string",
                    "description": "规则类型，如挂科、作弊、旷课、考试、处分"
                }
            },
            "required": ["rule_type"]
        }
    },
    {
        "name": "get_policy_time",
        "description": "查询某个政策文件的最新修订日期或有效性。",
        "input_schema": {
            "type": "object",
            "properties": {
                "policy_name": {
                    "type": "string",
                    "description": "政策名称，如奖学金、学生手册、处分、学籍"
                }
            },
            "required": ["policy_name"]
        }
    },
]


# ========== 资源定义 ==========
RESOURCES = [
    {
        "uri": "knowledge://overview",
        "name": "校园知识库概览",
        "description": "湖南师范大学校园知识库概览信息，包含数据量、覆盖范围、使用说明等。",
        "mimeType": "text/markdown",
    }
]


# ========== 工具列表处理器 ==========
async def handle_list_tools(params: ListToolsRequest, session) -> ListToolsResult:
    """返回可用工具列表"""
    tools = [
        Tool(
            name=t["name"],
            description=t["description"],
            inputSchema=t["input_schema"]
        )
        for t in TOOLS
    ]
    return ListToolsResult(tools=tools)


# ========== 工具调用处理器 ==========
async def handle_call_tool(params: CallToolRequest, session) -> CallToolResult:
    """处理工具调用"""
    name = params.name
    arguments = params.arguments if params.arguments else {}

    try:
        if name == "campus_rag_query":
            return await _tool_campus_rag_query(arguments)
        elif name == "campus_knowledge_search":
            return await _tool_campus_knowledge_search(arguments)
        elif name == "query_scholarship":
            return await _tool_query_scholarship(arguments)
        elif name == "query_discipline":
            return await _tool_query_discipline(arguments)
        elif name == "get_policy_time":
            return await _tool_get_policy_time(arguments)
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"未知工具：{name}")],
                isError=True
            )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"工具调用出错：{str(e)}")],
            isError=True
        )


# ========== 具体工具实现 ==========
async def _tool_campus_rag_query(args: dict) -> CallToolResult:
    """完整 RAG 问答"""
    question = args.get("question", "")
    if not question:
        return CallToolResult(
            content=[TextContent(type="text", text="请提供问题")],
            isError=True
        )

    from agent_graph import agent
    result = agent.invoke({
        "question": question,
        "chat_history": [],
        "thinking_steps": []
    })

    answer = result.get("answer", "抱歉，我没有找到答案。")
    thinking_steps = result.get("thinking_steps", [])
    retrieved_docs = result.get("retrieved_docs", [])
    intent = result.get("intent", "chat")

    # 构造结构化输出
    output_parts = [
        f"## 回答\n\n{answer}",
        f"\n\n## 问题分类\n\n{intent}",
    ]

    if thinking_steps:
        steps_text = "\n".join(f"- {step}" for step in thinking_steps)
        output_parts.append(f"\n\n## 思考过程\n\n{steps_text}")

    if retrieved_docs:
        docs_text = "\n\n".join(
            f"### 文档 {i+1}\n{doc}" for i, doc in enumerate(retrieved_docs)
        )
        output_parts.append(f"\n\n## 参考资料\n\n{docs_text}")

    full_output = "".join(output_parts)

    return CallToolResult(
        content=[TextContent(type="text", text=full_output)]
    )


async def _tool_campus_knowledge_search(args: dict) -> CallToolResult:
    """纯向量检索"""
    query = args.get("query", "")
    top_k = int(args.get("top_k", 5))

    if not query:
        return CallToolResult(
            content=[TextContent(type="text", text="请提供搜索关键词")],
            isError=True
        )

    from retriever import search_test
    results = search_test(query, top_k=top_k)

    if not results:
        return CallToolResult(
            content=[TextContent(type="text", text="未找到相关文档")]
        )

    output_parts = [f"找到 {len(results)} 条相关文档：\n\n"]
    for i, r in enumerate(results):
        output_parts.append(f"### [{i+1}] {r['title']}（相似度：{r['score']:.3f}）\n\n")
        output_parts.append(f"{r['content']}\n\n")

    full_output = "".join(output_parts)

    return CallToolResult(
        content=[TextContent(type="text", text=full_output)]
    )


async def _tool_query_scholarship(args: dict) -> CallToolResult:
    """奖学金查询"""
    grade = args.get("grade", "")
    gpa = float(args.get("gpa", 0))
    punishment_count = int(args.get("punishment_count", 0))

    if punishment_count > 0:
        result = "❌ 有处分记录，不得参评一等奖学金。根据学校规定，处分期间取消所有评奖评优资格。"
    elif gpa >= 3.5:
        result = "✅ 符合一等奖学金申请条件（要求：无挂科、无处分、绩点≥3.5）"
    elif gpa >= 3.0:
        result = "✅ 符合二等奖学金申请条件（要求：挂科≤1门、无处分、绩点≥3.0）"
    elif gpa >= 2.5:
        result = "⚠️ 符合三等奖学金申请条件（要求：挂科≤2门、无处分、绩点≥2.5）"
    else:
        result = "⚠️ 建议申请单项奖学金或进步奖。当前绩点较低，请努力提高学习成绩。"

    detail = f"\n\n**输入信息**：\n- 年级：{grade}\n- 绩点：{gpa}\n- 处分次数：{punishment_count}"

    return CallToolResult(
        content=[TextContent(type="text", text=result + detail)]
    )


async def _tool_query_discipline(args: dict) -> CallToolResult:
    """处分规定查询"""
    rule_type = args.get("rule_type", "")

    rules = {
        "挂科": "📚 挂科规定：学生挂科后可以申请补考，补考通过后成绩按60分计入。补考仍不通过者需重修该课程。",
        "作弊": "🚫 考试作弊规定：考试作弊将给予记过及以上处分，取消学士学位授予资格，并记入学生诚信档案。",
        "旷课": "📋 旷课规定：旷课累计超过20学时的，给予警告处分；累计超过40学时的，给予严重警告处分；累计超过60学时的，给予记过处分。",
        "考试": "📝 考试规定：学生应按时参加考试，无故缺考按旷考处理。考试作弊按《学生违纪处分条例》处理。",
        "处分": "⚠️ 处分规定：处分分为警告、严重警告、记过、留校察看、开除学籍五级。处分期间取消评奖评优资格。",
    }

    for key, value in rules.items():
        if key in rule_type or rule_type in key:
            return CallToolResult(
                content=[TextContent(type="text", text=value)]
            )

    return CallToolResult(
        content=[TextContent(type="text", text=f"❓ 未找到关于「{rule_type}」的相关规定。请尝试：挂科、作弊、旷课、考试、处分")]
    )


async def _tool_get_policy_time(args: dict) -> CallToolResult:
    """政策时间查询"""
    policy_name = args.get("policy_name", "")

    policies = {
        "奖学金": "《奖学金评定办法》最新版本为2024年9月修订，当前仍然有效。",
        "学生手册": "《学生手册》最新版本为2024年8月修订，当前仍然有效。",
        "处分": "《学生违纪处分条例》最新版本为2024年6月修订，当前仍然有效。",
        "学籍": "《学籍管理规定》最新版本为2024年9月修订，当前仍然有效。",
    }

    for key, value in policies.items():
        if key in policy_name:
            return CallToolResult(
                content=[TextContent(type="text", text=value)]
            )

    return CallToolResult(
        content=[TextContent(type="text", text=f"📄 《{policy_name}》当前版本为2024年修订版，仍在有效期内。如需确认具体日期，建议咨询教务处。")]
    )


# ========== 资源列表处理器 ==========
async def handle_list_resources(params: ListResourcesRequest, session) -> ListResourcesResult:
    """返回可用资源列表"""
    resources = [
        Resource(
            uri=r["uri"],
            name=r["name"],
            description=r["description"],
            mimeType=r.get("mimeType", "text/plain"),
        )
        for r in RESOURCES
    ]
    return ListResourcesResult(resources=resources)


# ========== 资源读取处理器 ==========
async def handle_read_resource(params: ReadResourceRequest, session) -> ReadResourceResult:
    """读取资源内容"""
    uri = params.uri

    if uri == "knowledge://overview":
        from retriever import collection
        try:
            doc_count = collection.count()
        except Exception:
            doc_count = "未知"

        content = f"""# 湖南师范大学校园知识库概览

## 📊 基本信息
- 文档数量：{doc_count} 条
- 向量模型：智谱 embedding-2（1024 维）
- 向量数据库：ChromaDB（本地持久化）
- 大语言模型：智谱 glm-4-flash

## 📚 覆盖范围
1. **政策制度** - 奖学金、处分、学籍管理等学校规章制度
2. **专业介绍** - 各学院专业设置、培养方案等
3. **新闻通知** - 学校最新公告、通知等

## 🔧 可用工具
1. `campus_rag_query` - 完整问答（推荐，带思考过程）
2. `campus_knowledge_search` - 纯检索，返回相关文档
3. `query_scholarship` - 奖学金评定资格查询
4. `query_discipline` - 处分规定查询
5. `get_policy_time` - 政策修订时间查询

## 💡 使用建议
- 一般问题直接用 `campus_rag_query`
- 需要自己处理检索结果时用 `campus_knowledge_search`
- 明确的业务查询（奖学金、处分等）用对应工具更快
"""
        return ReadResourceResult(
            contents=[TextResourceContents(uri=uri, mimeType="text/markdown", text=content)]
        )
    else:
        return ReadResourceResult(
            contents=[TextResourceContents(uri=uri, mimeType="text/plain", text=f"未知资源：{uri}")],
        )


# ========== 注册处理器 ==========
server.add_request_handler("tools/list", ListToolsRequest, handle_list_tools)
server.add_request_handler("tools/call", CallToolRequest, handle_call_tool)
server.add_request_handler("resources/list", ListResourcesRequest, handle_list_resources)
server.add_request_handler("resources/read", ReadResourceRequest, handle_read_resource)


# ========== 启动服务器 ==========
async def main():
    """主入口函数"""
    print("🎓 湖南师范大学校园智能助手 MCP 服务启动中...", file=sys.stderr)
    print("📡 运行模式：stdio", file=sys.stderr)
    print(f"🔧 提供 {len(TOOLS)} 个工具，{len(RESOURCES)} 个资源", file=sys.stderr)
    print("💡 在 MCP 客户端中配置此脚本即可使用", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
