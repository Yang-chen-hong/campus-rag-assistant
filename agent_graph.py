"""
校园智能 Agent v2.0
======================
队员A（核心Agent设计）负责模块 - 升级版

升级内容：
  1. ReAct Agent - 真正的工具调用（Tool Calling），大模型自主决定用什么工具
  2. 多轮对话记忆管理 - 自动总结压缩历史，避免 token 爆炸
  3. 引用来源标注 - 回答中标注引用了哪篇文档
  4. 最大迭代限制 - 防止死循环
  5. 错误降级 - 工具调用失败有优雅的 fallback
  6. 流式生成支持 - 支持逐 token 输出回答
"""

import os
import json
from dotenv import load_dotenv
from zhipu_api import chat_completions, _get_api_key
from typing import List, Dict, Any, Optional, Generator

load_dotenv()

# ========== 配置 ==========
MODEL_NAME = "glm-4-flash"
MAX_ITERATIONS = 5              # 最大工具调用轮次（防止死循环）
MAX_HISTORY_TURNS = 10          # 保留最近多少轮对话
SUMMARY_THRESHOLD = 12          # 超过多少轮触发总结压缩
TEMPERATURE = 0.1


# ========== 客户端管理 ==========
def get_client():
    """兼容旧接口 - 返回 None（已改用 zhipu_api 模块）"""
    return None


def reset_client():
    """重置客户端（切换 API Key 时调用）"""
    pass


# 兼容旧接口
def get_model():
    """兼容旧版调用（返回客户端的 invoke 包装）"""
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
            "description": "搜索校园知识库，获取学校政策制度、专业介绍、新闻通知、奖学金评定办法、处分规定等具体文档内容。凡是询问'XX是什么''XX有哪些''XX规定'等需要依据的问题，必须先调用此工具获取具体内容再回答。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题，应该是具体的、便于检索的短语"
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
            "description": "查询奖学金评定资格。根据年级、绩点和处分次数，自动判断符合哪类奖学金申请条件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "grade": {
                        "type": "string",
                        "description": "年级，如：大一、大二、大三、大四"
                    },
                    "gpa": {
                        "type": "number",
                        "description": "绩点，如 3.5（满分一般为4.0）"
                    },
                    "punishment_count": {
                        "type": "number",
                        "description": "处分次数，0表示无处分"
                    }
                },
                "required": ["grade", "gpa", "punishment_count"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_discipline_rule",
            "description": "查询学校处分相关规定，包括挂科、作弊、旷课、考试纪律等具体规定。",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_type": {
                        "type": "string",
                        "description": "规则类型，如：挂科、作弊、旷课、考试、处分"
                    }
                },
                "required": ["rule_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_policy_revision_time",
            "description": "查询某个政策文件的最新修订日期或有效性说明。",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_name": {
                        "type": "string",
                        "description": "政策名称，如：奖学金、学生手册、处分条例、学籍管理"
                    }
                },
                "required": ["policy_name"]
            }
        }
    },
]


# ========== 系统提示词 ==========
SYSTEM_PROMPT = """你是「湖南师范大学校园智能助手」，一个专业、友好的校园咨询 AI。

## 你的身份
- 你是湖南师范大学的官方智能助手
- 你只回答与湖南师范大学校园相关的问题
- 对于非校园相关的闲聊，可以友好回应但不要展开

## ⚠️ 核心规则（必须严格遵守）
1. **凡事先检索**：只要用户问的是校园政策、专业、奖学金、处分、新闻、规章制度等任何需要依据的问题，**必须先调用 search_knowledge_base 检索资料**，再根据检索结果回答。绝对不能凭印象回答。

2. **不确定就检索**：哪怕你觉得自己知道答案，只要涉及具体规定、数字、条件等，也要先检索确认。

3. **标注来源**：回答中引用了检索资料的内容，必须标注来源编号，格式如「[来源1]」。

4. **可以多次检索**：一次检索信息不够时，换关键词再检索，或者调用不同工具补充。

5. **信息不足就追问**：需要用户提供更多信息才能回答时（比如查奖学金需要绩点），主动询问。

6. **绝不编造**：没有找到相关信息时，明确告诉用户"未找到相关资料"，不要凭空编造。

## 你的工具
- `search_knowledge_base` - 搜索校园知识库（政策、专业、新闻等），**最常用**
- `check_scholarship_eligibility` - 奖学金资格评定（需要年级、绩点、处分次数）
- `get_discipline_rule` - 查询处分规定（挂科、作弊、旷课等）
- `get_policy_revision_time` - 查询政策修订时间

## 调用示例（参考）
- 用户问："奖学金评定条件是什么？" → 调用 search_knowledge_base 检索具体规定
- 用户问："我绩点3.5能评奖学金吗？" → 先追问年级和处分情况，再调用 check_scholarship_eligibility
- 用户问："考试作弊会怎么样？" → 调用 search_knowledge_base + get_discipline_rule
- 用户问："你好" → 直接回应，不需要调用工具

## 回答要求
- **充分利用检索结果**：仔细阅读每一篇检索到的资料，将相关信息整合到回答中，不要只罗列文档标题
- **详细具体**：给出具体的条件、数字、规定内容，而不是泛泛而谈
- **分点说明**：复杂内容用 1、2、3 分点列出，清晰易读
- **标注来源**：引用资料内容时标注来源编号，如「[来源1]」「[来源2]」
- **专业准确**：用词严谨，符合学校官方表述
- **适当使用 emoji**：增加可读性，但不要过多
- **重要信息加粗**：关键条件、数字等用 **加粗** 强调
"""


# ========== 工具执行函数 ==========
def execute_tool(tool_name: str, arguments: dict, chat_history: list = None) -> str:
    """
    执行工具调用，返回工具结果字符串。
    """
    try:
        if tool_name == "search_knowledge_base":
            return _tool_search(arguments, chat_history)
        elif tool_name == "check_scholarship_eligibility":
            return _tool_scholarship(arguments)
        elif tool_name == "get_discipline_rule":
            return _tool_discipline(arguments)
        elif tool_name == "get_policy_revision_time":
            return _tool_policy_time(arguments)
        else:
            return f"错误：未知工具「{tool_name}」"
    except Exception as e:
        return f"工具调用出错：{str(e)}"


def _tool_search(args: dict, chat_history: list = None) -> str:
    """检索知识库"""
    from retriever import search_test
    query = args.get("query", "")
    if not query:
        return "请提供搜索关键词"

    results = search_test(query, top_k=5, chat_history=chat_history)

    if not results:
        return "未找到相关资料。建议换个关键词再试试。"

    output_parts = [f"检索到 {len(results)} 条相关资料：\n"]
    for i, r in enumerate(results):
        output_parts.append(f"\n【来源{i+1}】《{r['title']}》(相关度: {r['score']})\n")
        output_parts.append(f"{r['content']}\n")

    return "".join(output_parts)


def _tool_scholarship(args: dict) -> str:
    """奖学金资格查询"""
    grade = args.get("grade", "")
    gpa = float(args.get("gpa", 0))
    punishment_count = int(args.get("punishment_count", 0))

    if punishment_count > 0:
        return "❌ 有处分记录，不得参评一等奖学金。根据学校规定，处分期间取消所有评奖评优资格。"
    elif gpa >= 3.5:
        return "✅ 符合一等奖学金申请条件（要求：无挂科、无处分、绩点≥3.5）"
    elif gpa >= 3.0:
        return "✅ 符合二等奖学金申请条件（要求：挂科≤1门、无处分、绩点≥3.0）"
    elif gpa >= 2.5:
        return "⚠️ 符合三等奖学金申请条件（要求：挂科≤2门、无处分、绩点≥2.5）"
    else:
        return "⚠️ 建议申请单项奖学金或进步奖。当前绩点较低，请努力提高学习成绩。"


def _tool_discipline(args: dict) -> str:
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
            return value

    return f"❓ 未找到关于「{rule_type}」的相关规定。可查询的类别：挂科、作弊、旷课、考试、处分"


def _tool_policy_time(args: dict) -> str:
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
            return value

    return f"📄 《{policy_name}》当前版本为2024年修订版，仍在有效期内。如需确认具体日期，建议咨询教务处。"


# ========== 记忆管理 ==========
def summarize_history(chat_history: List[Dict]) -> List[Dict]:
    """
    对话历史总结压缩。
    当历史超过阈值时，将较早的对话总结为一条 system 消息。
    """
    if len(chat_history) <= SUMMARY_THRESHOLD:
        return chat_history

    # 保留最近 N 轮，前面的总结
    recent_count = MAX_HISTORY_TURNS
    old_history = chat_history[:-recent_count]
    recent_history = chat_history[-recent_count:]

    # 构造总结请求
    history_text = ""
    for msg in old_history:
        role = "用户" if msg.get("role") == "user" else "助手"
        history_text += f"{role}：{msg.get('content', '')[:200]}\n"

    prompt = f"""请用简洁的语言总结以下对话历史的核心内容，保留关键信息点。
重点保留：用户问了什么、助手回答了什么、有哪些关键信息被确认。

对话历史：
{history_text}

总结（100字以内）："""

    try:
        resp = chat_completions(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            temperature=0.1,
            max_tokens=150
        )
        summary = resp["choices"][0]["message"]["content"].strip()
        # 用一条 system 消息承载总结
        summarized = [{"role": "system", "content": f"【历史对话总结】{summary}"}]
        return summarized + recent_history
    except Exception as e:
        print(f"[历史总结失败，保留原历史] {e}")
        return chat_history[-MAX_HISTORY_TURNS:]


# ========== 核心 Agent（ReAct 模式） ==========
def agent_invoke(question: str, chat_history: List[Dict] = None,
                 stream: bool = False) -> Dict[str, Any]:
    """
    Agent 主函数 - ReAct 模式的工具调用问答。

    参数：
      question: 用户问题
      chat_history: 对话历史列表，每条包含 role 和 content
      stream: 是否流式输出（暂未完全实现流式工具调用，最终回答支持流式）

    返回：
      dict: 包含 answer、thinking_steps、retrieved_docs、tool_calls
    """
    if chat_history is None:
        chat_history = []

    thinking_steps = []
    retrieved_docs = []
    tool_call_history = []  # 记录工具调用过程

    # 1. 历史记忆管理（总结压缩）
    if len(chat_history) > SUMMARY_THRESHOLD:
        thinking_steps.append("🧠 正在压缩对话历史...")
        chat_history = summarize_history(chat_history)
        thinking_steps.append(f"   → 已压缩为 {len(chat_history)} 条消息")

    # 2. 构造消息列表
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 加入历史（注意：历史里可能已经有总结好的 system 消息）
    for msg in chat_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ["user", "assistant", "system"]:
            messages.append({"role": role, "content": content})

    # 加入当前问题
    messages.append({"role": "user", "content": question})

    thinking_steps.append("🤔 正在分析问题...")

    # 3. ReAct 循环
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        thinking_steps.append(f"🔄 第 {iteration} 轮思考...")

        try:
            response = chat_completions(
                messages=messages,
                model=MODEL_NAME,
                tools=TOOLS,
                temperature=TEMPERATURE,
            )
        except Exception as e:
            thinking_steps.append(f"   ❌ 模型调用失败：{e}")
            return {
                "answer": f"抱歉，服务暂时不可用（{e}）。请稍后重试或检查 API Key。",
                "thinking_steps": thinking_steps,
                "retrieved_docs": retrieved_docs,
                "tool_calls": tool_call_history,
                "intent": "error",
            }

        message = response["choices"][0]["message"]
        messages.append(message)

        # 检查是否有工具调用
        if not message.get("tool_calls"):
            # 没有工具调用 = 生成最终回答
            thinking_steps.append("   ✅ 生成最终回答")
            answer = message.get("content") or "抱歉，我无法回答这个问题。"
            return {
                "answer": answer,
                "thinking_steps": thinking_steps,
                "retrieved_docs": retrieved_docs,
                "tool_calls": tool_call_history,
                "intent": _classify_intent_from_history(question, tool_call_history),
            }

        # 有工具调用，逐个执行
        for tool_call in message["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                args = {}

            thinking_steps.append(f"   🔧 调用工具：{tool_name}")
            if args:
                args_display = {k: v for k, v in args.items() if k != "query" or len(str(v)) < 50}
                thinking_steps.append(f"      参数：{json.dumps(args_display, ensure_ascii=False)}")

            # 执行工具
            tool_result = execute_tool(tool_name, args, chat_history)

            # 记录检索到的文档（用于前端展示）
            if tool_name == "search_knowledge_base":
                # 从结果中提取文档
                lines = tool_result.split("\n")
                for line in lines:
                    if line.startswith("【来源"):
                        # 提取标题
                        pass
                # 简单把完整检索结果存起来
                retrieved_docs.append(tool_result[:500] + "..." if len(tool_result) > 500 else tool_result)

            tool_call_history.append({
                "tool": tool_name,
                "args": args,
                "result_preview": tool_result[:100] + "..." if len(tool_result) > 100 else tool_result,
            })

            thinking_steps.append(f"      ✅ 工具执行完成（结果 {len(tool_result)} 字）")

            # 将工具结果加入消息
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result,
            })

    # 超过最大迭代次数，强制生成回答
    thinking_steps.append(f"⚠️ 已达到最大迭代次数（{MAX_ITERATIONS}），生成最终回答")
    messages.append({"role": "user", "content": "请根据已有的信息直接回答我的问题，不要再调用工具了。"})

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
        "intent": _classify_intent_from_history(question, tool_call_history),
    }


def _classify_intent_from_history(question: str, tool_calls: list) -> str:
    """根据调用过的工具粗略分类意图（用于兼容旧接口）"""
    tool_names = [t["tool"] for t in tool_calls]
    if "check_scholarship_eligibility" in tool_names or "get_discipline_rule" in tool_names or "get_policy_revision_time" in tool_names:
        return "policy"
    if "search_knowledge_base" in tool_names:
        # 粗略判断：根据问题关键词
        if any(kw in question for kw in ["专业", "学院", "课程", "培养"]):
            return "major"
        if any(kw in question for kw in ["新闻", "通知", "公告", "最新"]):
            return "news"
        return "policy"
    return "chat"


# ========== 流式生成（仅最终回答） ==========
def agent_stream(question: str, chat_history: List[Dict] = None) -> Generator[str, None, Dict]:
    """
    流式输出版本 - 先做 ReAct 推理（非流式），再流式输出最终回答。

    返回：
      生成器，逐步产出回答文本。最后通过 StopIteration 返回完整结果 dict。
    """
    # 先执行完整推理（获取思考过程和工具调用结果）
    result = agent_invoke(question, chat_history, stream=True)

    # 流式输出最终回答
    client = get_client()

    # 重新构造一个只输出最终回答的请求（更轻量）
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if chat_history:
        for msg in chat_history[-6:]:
            if msg.get("role") in ["user", "assistant"]:
                messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": result["answer"]})

    # 直接把已生成的答案流式返回（模拟流式效果，实际已经生成好了）
    # 真正的流式推理比较复杂，这里做折中方案：
    # 思考过程一次性出，最终回答逐字"蹦"出来
    answer = result["answer"]
    chunk_size = 2  # 每次返回几个字

    for i in range(0, len(answer), chunk_size):
        yield answer[i:i + chunk_size]

    # 返回完整结果
    return result


# ========== 兼容旧接口（agent_graph.py 的旧调用方式） ==========
class AgentCompat:
    """兼容旧版 agent.invoke() 接口"""

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        兼容旧接口的 invoke 方法。
        旧接口接收 state dict，返回 state dict。
        """
        question = state.get("question", "")
        chat_history = state.get("chat_history", [])
        thinking_steps = state.get("thinking_steps", [])

        result = agent_invoke(question, chat_history)

        # 合并 thinking_steps
        all_thinking = thinking_steps + result["thinking_steps"]

        return {
            "answer": result["answer"],
            "intent": result.get("intent", "chat"),
            "retrieved_docs": result.get("retrieved_docs", []),
            "thinking_steps": all_thinking,
            "chat_history": chat_history,
            "tool_calls": result.get("tool_calls", []),
        }


# 旧接口兼容
agent = AgentCompat()


# ========== 终端测试 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("校园智能 Agent v2.0 - 终端测试")
    print("=" * 60)
    print()

    chat_history = []

    test_questions = [
        "奖学金评定条件是什么？",
        "我绩点3.6，大二，没处分，能评一等奖吗？",
        "考试作弊有什么后果？",
    ]

    for q in test_questions:
        print(f"\n{'='*50}")
        print(f"👤 你：{q}")
        print(f"{'='*50}")

        result = agent_invoke(q, chat_history)

        print("\n🧠 思考过程：")
        for step in result["thinking_steps"]:
            print(f"   {step}")

        print(f"\n🤖 助手：{result['answer']}")

        chat_history.append({"role": "user", "content": q})
        chat_history.append({"role": "assistant", "content": result["answer"]})

    print("\n\n✅ 测试完成！")
