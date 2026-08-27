"""
湖南师范大学校园智能助手 - MCP 服务器 v2.0
==========================================
队员E（MCP编写 + 主讲）负责模块

将校园 RAG Agent 封装为标准 MCP（Model Context Protocol）服务，
支持在任何兼容 MCP 的客户端中调用（Claude Desktop、Cursor、Windsurf 等）。

v2.0 改进：从 skills_tools.py 统一加载 Skill，一处定义三处复用
提供 10 个 MCP 工具 + 1 个资源

使用方式：
 python mcp_server.py
 （通过 stdio 模式与 MCP 客户端通信）
"""

import os
import sys
from typing import Any

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(__file__))

from skills_tools import get_mcp_tools, execute_skill, registry
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
    version="2.0.0",
)


# ========== 工具定义（从 skills_tools.py 统一加载） + campus_rag_query ==========
# campus_rag_query 是 MCP 专属工具，调用完整 Agent
EXTRA_TOOLS = [
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
]

TOOLS = get_mcp_tools() + EXTRA_TOOLS


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


# ========== 工具调用处理器（统一委托给 skills_tools.py） ==========
async def handle_call_tool(params: CallToolRequest, session) -> CallToolResult:
    """处理工具调用"""
    name = params.name
    arguments = params.arguments if params.arguments else {}

    try:
        if name == "campus_rag_query":
            return await _tool_campus_rag_query(arguments)
        else:
            # 所有其他工具统一交给 skills_tools.py 执行
            result = execute_skill(name, arguments)
            return CallToolResult(
                content=[TextContent(type="text", text=result)]
            )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"工具调用出错：{str(e)}")],
            isError=True
        )


# ========== 具体工具实现 ==========
async def _tool_campus_rag_query(args: dict) -> CallToolResult:
    """完整 RAG 问答（v4.0 ReAct Agent）"""
    question = args.get("question", "")
    if not question:
        return CallToolResult(
            content=[TextContent(type="text", text="请提供问题")],
            isError=True
        )

    from agent_graph import agent_invoke
    result = agent_invoke(question, chat_history=[])

    answer = result.get("answer", "抱歉，我没有找到答案。")
    thinking_steps = result.get("thinking_steps", [])
    tool_calls = result.get("tool_calls", [])
    intent = result.get("intent", "chat")

    output_parts = [
        f"## 回答\n\n{answer}",
        f"\n\n## 问题分类\n\n{intent}",
    ]

    if tool_calls:
        tools_text = "\n".join(
            f"- **{tc['tool']}**：{tc.get('args', {})}"
            for tc in tool_calls
        )
        output_parts.append(f"\n\n## 调用的工具\n\n{tools_text}")

    if thinking_steps:
        steps_text = "\n".join(f"- {step}" for step in thinking_steps)
        output_parts.append(f"\n\n## 思考过程\n\n{steps_text}")

    full_output = "".join(output_parts)

    return CallToolResult(
        content=[TextContent(type="text", text=full_output)]
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
3. **校园生活** - 学费、宿舍、食堂、交通、社团、军训等
4. **学生服务** - 医保、档案户口、防诈骗、心理咨询等

## 🔧 可用工具（{len(TOOLS)} 个）
1. `campus_rag_query` - 完整问答（推荐，带思考过程）
2. `search_knowledge_base` - 知识库检索
3. `campus_faq_match` - 校园高频问题匹配
4. `check_scholarship_eligibility` - 奖学金资格判断
5. `query_discipline_rules` - 处分规定查询
6. `get_college_info` - 学院专业信息
7. `calculate_gpa` - GPA计算器
8. `check_graduation_requirements` - 毕业条件检查
9. `get_campus_contacts` - 校园常用电话
10. `check_tuition_fees` - 学费查询
11. `get_dormitory_info` - 宿舍信息

## 💡 使用建议
- 一般问题直接用 `campus_rag_query`
- 精确检索用 `search_knowledge_base`
- 明确的业务查询用对应工具更快
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
    print(f"🔧 提供 {len(TOOLS)} 个工具，{len(RESOURCES)} 个资源（v2.0 统一 Skill 系统）", file=sys.stderr)
    print(f"📋 Skill 列表：{', '.join(registry.names())}", file=sys.stderr)
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
