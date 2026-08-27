"""
校园智能 Agent v4.0
======================
核心改进：
  1. 统一 Skill 系统：从 skills_tools.py 加载，一处定义三处复用
  2. 10个业务 Skill 全部查数据库（不再硬编码）
  3. 更好的系统提示词（引导模型给出详细回答）
  4. 工具结果格式优化（给模型更充足的上下文）
  5. 多步推理（先检索→再分析→再生成回答）
"""

import os
import json
from dotenv import load_dotenv
from zhipu_api import chat_completions, _get_api_key
from skills_tools import get_openai_tools, execute_skill
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


# ========== 工具定义（从 skills_tools.py 统一加载，一处定义三处复用） ==========
TOOLS = get_openai_tools()


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
- `campus_faq_match`：校园高频问题匹配（常见问题快速匹配）
- `check_scholarship_eligibility`：判断奖学金资格（需要年级、绩点、处分信息）
- `query_discipline_rules`：查询处分规定（挂科、作弊、旷课等）
- `get_college_info`：查询学院和专业信息
- `calculate_gpa`：GPA计算器（输入各科成绩和学分）
- `check_graduation_requirements`：检查毕业条件
- `get_campus_contacts`：查校园常用电话和办公地点
- `check_tuition_fees`：查询学费标准
- `get_dormitory_info`：查询宿舍信息
- 可以多次调用工具，换不同关键词搜索
- 问题不涉及具体信息时（如"你好"），不用调用工具"""


# ========== 工具执行（统一委托给 skills_tools.py） ==========
def execute_tool(tool_name: str, arguments: dict, chat_history: list = None) -> str:
    return execute_skill(tool_name, arguments, chat_history)


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
