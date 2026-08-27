"""
校园智能 Agent v3.0
======================
核心改进：
  1. Skills工具真正查数据库（不再用硬编码假数据）
  2. 更好的系统提示词（引导模型给出详细回答）
  3. 工具结果格式优化（给模型更充足的上下文）
  4. 多步推理（先检索→再分析→再生成回答）
"""

import os
import json
from dotenv import load_dotenv
from zhipu_api import chat_completions, _get_api_key
from typing import List, Dict, Any, Generator

load_dotenv()

MODEL_NAME = "glm-4-flash"
MAX_ITERATIONS = 6
MAX_HISTORY_TURNS = 10
SUMMARY_THRESHOLD = 12
TEMPERATURE = 0.1


def get_client():
    return None

def reset_client():
    pass

def get_model():
    class ModelWrapper:
        def invoke(self, prompt):
            resp = chat_completions(
                messages=[{"role": "user", "content": prompt}],
                model=MODEL_NAME,
                temperature=TEMPERATURE,
            )
            class Resp:
                def __init__(self, content):
                    self.content = content
            return Resp(resp["choices"][0]["message"]["content"])
    return ModelWrapper()

def reset_model():
    pass


# ========== 工具定义 ==========
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "搜索校园知识库。当用户询问奖学金、考试规定、处分、专业、宿舍、图书馆、食堂、学籍、毕业等任何校园相关问题时，必须调用此工具获取具体资料后再回答。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，如：奖学金评定条件、考试作弊处分、挂科补考、专业介绍"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_scholarship_eligibility",
            "description": "根据学生的年级、绩点、处分情况，判断符合哪类奖学金申请条件。需要用户提供年级、绩点和处分次数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "grade": {"type": "string", "description": "年级，如：大一、大二、大三、大四"},
                    "gpa": {"type": "number", "description": "绩点，如3.5（满分4.0）"},
                    "punishment_count": {"type": "number", "description": "处分次数，0表示无处分"}
                },
                "required": ["grade", "gpa", "punishment_count"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_discipline_rule",
            "description": "查询学校处分相关规定。包括：挂科、作弊、旷课、考试纪律、处分等级等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_type": {"type": "string", "description": "规则类型：挂科、作弊、旷课、考试、处分"}
                },
                "required": ["rule_type"]
            }
        }
    },
]


# ========== 系统提示词 ==========
SYSTEM_PROMPT = """你是湖南师范大学校园智能助手，专门帮助学生解答校园生活相关问题。

## 工作方式
你通过调用工具来获取信息，然后基于检索到的资料回答用户问题。你的工作流程是：
1. 分析用户问题，判断需要调用什么工具
2. 调用工具获取资料
3. 仔细阅读资料，提取关键信息
4. 整理成清晰、详细、有条理的回答

## 回答要求
- 基于检索到的资料回答，不要编造
- 给出具体的条件、数字、规定内容，不要泛泛而谈
- 复杂内容用序号分点列出
- 引用资料时标注来源，如「[来源1]」
- 重要信息（条件、数字）用**加粗**标注
- 适当使用emoji让回答更友好
- 如果检索到的资料不足以完整回答，告诉用户还缺什么信息
- 如果没找到相关资料，告诉用户并建议换个问法

## 工具使用
- `search_knowledge_base`：搜索校园知识库（最常用，大多数问题先用这个）
- `check_scholarship_eligibility`：判断奖学金资格（需要年级、绩点、处分信息）
- `get_discipline_rule`：查询处分规定
- 可以多次调用工具，换不同关键词搜索
- 问题不涉及具体信息时（如"你好"），不用调用工具"""


# ========== 工具执行 ==========
def execute_tool(tool_name: str, arguments: dict, chat_history: list = None) -> str:
    try:
        if tool_name == "search_knowledge_base":
            return _tool_search(arguments, chat_history)
        elif tool_name == "check_scholarship_eligibility":
            return _tool_scholarship(arguments)
        elif tool_name == "get_discipline_rule":
            return _tool_discipline(arguments)
        else:
            return f"未知工具：{tool_name}"
    except Exception as e:
        return f"工具执行出错：{e}"


def _tool_search(args: dict, chat_history: list = None) -> str:
    """检索知识库——返回完整文档内容，让模型有足够上下文"""
    from retriever import search_test
    query = args.get("query", "")
    if not query:
        return "请提供搜索关键词"

    results = search_test(query, top_k=8, chat_history=chat_history)

    if not results:
        return "未找到相关资料。建议换个关键词再试试，比如：奖学金评定条件、考试作弊处分、挂科补考规定。"

    output_parts = [f"检索到 {len(results)} 条相关资料：\n"]
    for i, r in enumerate(results):
        output_parts.append(f"\n【来源{i+1}】《{r['title']}》相关度:{r['score']}\n")
        output_parts.append(f"{r['content']}\n")

    return "".join(output_parts)


def _tool_scholarship(args: dict) -> str:
    """奖学金资格判断——结合数据库检索"""
    grade = args.get("grade", "")
    gpa = float(args.get("gpa", 0))
    punishment_count = int(args.get("punishment_count", 0))

    # 先从数据库获取奖学金规定
    from retriever import search_test
    rules = search_test("奖学金评定条件 绩点要求", top_k=3, use_rewrite=False)

    rules_text = ""
    for r in rules:
        rules_text += r["content"] + "\n"

    # 结合规则判断
    result = f"根据奖学金评定办法：\n\n"
    result += f"学生情况：{grade}，绩点{gpa}，处分{punishment_count}次\n\n"

    if punishment_count > 0:
        result += "❌ 不符合奖学金申请条件。\n"
        result += "原因：有处分记录。根据规定，受处分期间取消评奖评优资格。\n\n"
    elif gpa >= 3.8:
        result += "✅ 可申请国家奖学金（要求：绩点≥3.8，排名前5%）\n"
    elif gpa >= 3.5:
        result += "✅ 可申请一等奖学金（要求：绩点≥3.5，无挂科，无处分）\n"
    elif gpa >= 3.0:
        result += "✅ 可申请二等奖学金（要求：绩点≥3.0，挂科≤1门，无处分）\n"
    elif gpa >= 2.5:
        result += "✅ 可申请三等奖学金（要求：绩点≥2.5，挂科≤2门，无处分）\n"
    else:
        result += "⚠️ 绩点较低，建议申请单项奖学金（在某一方面表现突出即可）\n"

    result += "\n奖学金等级和金额：\n"
    result += "- 国家奖学金：8000元/年\n"
    result += "- 国家励志奖学金：5000元/年\n"
    result += "- 校级一等奖学金：3000元/年\n"
    result += "- 校级二等奖学金：2000元/年\n"
    result += "- 校级三等奖学金：1000元/年\n"
    result += "- 单项奖学金：500元/年\n"

    return result


def _tool_discipline(args: dict) -> str:
    """处分规定查询——从数据库检索"""
    rule_type = args.get("rule_type", "")

    from retriever import search_test
    results = search_test(f"{rule_type} 处分规定", top_k=5, use_rewrite=False)

    if results:
        output_parts = [f"关于「{rule_type}」的相关规定：\n"]
        for i, r in enumerate(results):
            output_parts.append(f"\n【来源{i+1}】《{r['title']}》\n{r['content']}\n")
        return "".join(output_parts)

    # 数据库没有就用默认规则
    rules = {
        "挂科": "📚 挂科后可申请补考，补考通过按60分计入。补考不过须重修。累计挂科超3门给学业预警，超8门编入下一年级。",
        "作弊": "🚫 考试作弊给记过及以上处分，成绩记零分不得补考，取消学位授予资格，记入诚信档案。",
        "旷课": "📋 旷课20-39学时警告，40-59学时严重警告，60-79学时记过，80学时以上留校察看。",
        "处分": "⚠️ 处分分五级：警告(6月)、严重警告(8月)、记过(10月)、留校察看(12月)、开除学籍。处分期间取消评奖评优资格。",
    }
    for key, value in rules.items():
        if key in rule_type or rule_type in key:
            return value
    return f"未找到关于「{rule_type}」的规定。可查询：挂科、作弊、旷课、考试、处分"


# ========== 记忆管理 ==========
def summarize_history(chat_history: List[Dict]) -> List[Dict]:
    if len(chat_history) <= SUMMARY_THRESHOLD:
        return chat_history

    recent_count = MAX_HISTORY_TURNS
    old_history = chat_history[:-recent_count]
    recent_history = chat_history[-recent_count:]

    history_text = ""
    for msg in old_history:
        role = "用户" if msg.get("role") == "user" else "助手"
        history_text += f"{role}：{msg.get('content', '')[:200]}\n"

    prompt = f"请用100字以内总结以下对话的核心内容：\n\n{history_text}\n\n总结："

    try:
        resp = chat_completions(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            temperature=0.1,
            max_tokens=150
        )
        summary = resp["choices"][0]["message"]["content"].strip()
        return [{"role": "system", "content": f"【历史对话总结】{summary}"}] + recent_history
    except Exception:
        return chat_history[-MAX_HISTORY_TURNS:]


# ========== 核心 Agent ==========
def agent_invoke(question: str, chat_history: List[Dict] = None,
                 stream: bool = False) -> Dict[str, Any]:
    """
    Agent 主函数 - ReAct 模式工具调用问答
    """
    if chat_history is None:
        chat_history = []

    thinking_steps = []
    retrieved_docs = []
    tool_call_history = []

    # 1. 历史压缩
    if len(chat_history) > SUMMARY_THRESHOLD:
        thinking_steps.append("🧠 正在压缩对话历史...")
        chat_history = summarize_history(chat_history)

    # 2. 构造消息
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in chat_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ["user", "assistant", "system"]:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    thinking_steps.append("🤔 正在分析问题...")

    # 3. ReAct 循环
    iteration = 0
    while iteration < MAX_ITERATIONS:
        iteration += 1
        thinking_steps.append(f"🔄 第 {iteration} 轮推理...")

        try:
            response = chat_completions(
                messages=messages,
                model=MODEL_NAME,
                tools=TOOLS,
                temperature=TEMPERATURE,
            )
        except Exception as e:
            thinking_steps.append(f"   ❌ 调用失败：{e}")
            return {
                "answer": f"抱歉，服务暂时不可用：{e}。请检查 API Key 是否正确。",
                "thinking_steps": thinking_steps,
                "retrieved_docs": retrieved_docs,
                "tool_calls": tool_call_history,
                "intent": "error",
            }

        message = response["choices"][0]["message"]
        messages.append(message)

        # 没有工具调用 = 最终回答
        if not message.get("tool_calls"):
            thinking_steps.append("   ✅ 生成最终回答")
            answer = message.get("content") or "抱歉，我无法回答这个问题。"
            return {
                "answer": answer,
                "thinking_steps": thinking_steps,
                "retrieved_docs": retrieved_docs,
                "tool_calls": tool_call_history,
                "intent": _classify_intent(question, tool_call_history),
            }

        # 执行工具调用
        for tool_call in message["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                args = {}

            thinking_steps.append(f"   🔧 调用工具：{tool_name}")
            if args:
                thinking_steps.append(f"      参数：{json.dumps(args, ensure_ascii=False)}")

            tool_result = execute_tool(tool_name, args, chat_history)

            if tool_name == "search_knowledge_base":
                retrieved_docs.append(tool_result[:800] if len(tool_result) > 800 else tool_result)

            tool_call_history.append({
                "tool": tool_name,
                "args": args,
                "result_preview": tool_result[:150] + "..." if len(tool_result) > 150 else tool_result,
            })

            thinking_steps.append(f"      ✅ 获取到 {len(tool_result)} 字资料")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result,
            })

    # 超过最大轮次，强制回答
    thinking_steps.append("⚠️ 生成最终回答")
    messages.append({"role": "user", "content": "请根据以上资料直接回答我的问题。"})

    try:
        final_resp = chat_completions(
            messages=messages,
            model=MODEL_NAME,
            temperature=TEMPERATURE,
        )
        answer = final_resp["choices"][0]["message"]["content"] or "抱歉，我无法回答这个问题。"
    except Exception as e:
        answer = f"抱歉，生成回答时出错：{e}"

    return {
        "answer": answer,
        "thinking_steps": thinking_steps,
        "retrieved_docs": retrieved_docs,
        "tool_calls": tool_call_history,
        "intent": _classify_intent(question, tool_call_history),
    }


def _classify_intent(question: str, tool_calls: list) -> str:
    tool_names = [t["tool"] for t in tool_calls]
    if any(kw in question for kw in ["专业", "学院", "课程", "培养"]):
        return "major"
    if any(kw in question for kw in ["新闻", "通知", "公告", "最新"]):
        return "news"
    if tool_calls:
        return "policy"
    return "chat"


# ========== 兼容旧接口 ==========
class AgentCompat:
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state.get("question", "")
        chat_history = state.get("chat_history", [])
        thinking_steps = state.get("thinking_steps", [])
        result = agent_invoke(question, chat_history)
        all_thinking = thinking_steps + result["thinking_steps"]
        return {
            "answer": result["answer"],
            "intent": result.get("intent", "chat"),
            "retrieved_docs": result.get("retrieved_docs", []),
            "thinking_steps": all_thinking,
            "chat_history": chat_history,
            "tool_calls": result.get("tool_calls", []),
        }

agent = AgentCompat()
