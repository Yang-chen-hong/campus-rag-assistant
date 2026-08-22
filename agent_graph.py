import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# ========== 导入队友的真实检索函数 ==========
# 注意：需要把 7-.py 重命名为 retriever.py
# 然后在终端执行：rename 7-.py retriever.py
from retriever import search_test
from skills_tools import query_scholarship, query_discipline, get_policy_time

load_dotenv()


def _get_api_key() -> str:
    """获取 API Key：优先环境变量，其次 Streamlit Secrets"""
    api_key = os.getenv("ZHIPU_API_KEY")
    if api_key:
        return api_key
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "ZHIPU_API_KEY" in st.secrets:
            return st.secrets["ZHIPU_API_KEY"]
    except Exception:
        pass
    raise ValueError("未找到 ZHIPU_API_KEY，请在环境变量或 Streamlit Secrets 中配置")


# 懒加载模型
_model = None


def get_model() -> ChatZhipuAI:
    """获取智谱大模型客户端（懒加载）"""
    global _model
    if _model is None:
        _model = ChatZhipuAI(
            model="glm-4-flash",
            api_key=_get_api_key(),
            temperature=0.1
        )
    return _model

class AgentState(TypedDict):
    question: str
    intent: str
    retrieved_docs: List[str]
    answer: str
    retry_count: int
    chat_history: List[dict]
    thinking_steps: List[str]

# ========== 已删除 mock_retrieve，改用真实检索 ==========

# ---------- 节点 ----------
def classify_intent(state: AgentState):
    history = state.get("chat_history", [])
    thinking = state.get("thinking_steps", [])
    thinking.append("🔍 正在分析问题意图...")
    print(f"[DEBUG] classify 收到的历史条数: {len(history)}")
    
    history_text = ""
    if history:
        for msg in history[-4:]:
            history_text += f"{msg['role']}: {msg['content']}\n"
        history_text = "对话历史：\n" + history_text + "\n"
    
    prompt = f"""
    {history_text}
    用户最新问题：{state["question"]}
    判断属于哪个类别，只输出一个词：policy / major / news / chat
    """
    response = get_model().invoke(prompt)
    intent = response.content.strip().lower()
    thinking.append(f"   → 判断为：{intent}")
    
    return {
        "intent": intent,
        "chat_history": state.get("chat_history", []),
        "thinking_steps": thinking
    }

def retrieve_policy(state: AgentState):
    thinking = state.get("thinking_steps", [])
    thinking.append("📚 正在检索政策制度库...")
    # ===== 调用真实的检索函数 =====
    results = search_test(state["question"], top_k=5)
    docs = [r["content"] for r in results]
    thinking.append(f"   → 找到 {len(docs)} 条相关内容")
    return {
        "retrieved_docs": docs,
        "chat_history": state.get("chat_history", []),
        "thinking_steps": thinking
    }

def retrieve_major(state: AgentState):
    thinking = state.get("thinking_steps", [])
    thinking.append("📚 正在检索专业介绍库...")
    # ===== 调用真实的检索函数 =====
    results = search_test(state["question"], top_k=5)
    docs = [r["content"] for r in results]
    thinking.append(f"   → 找到 {len(docs)} 条相关内容")
    return {
        "retrieved_docs": docs,
        "chat_history": state.get("chat_history", []),
        "thinking_steps": thinking
    }

def retrieve_news(state: AgentState):
    thinking = state.get("thinking_steps", [])
    thinking.append("📚 正在检索新闻通知库...")
    # ===== 调用真实的检索函数 =====
    results = search_test(state["question"], top_k=5)
    docs = [r["content"] for r in results]
    thinking.append(f"   → 找到 {len(docs)} 条相关内容")
    return {
        "retrieved_docs": docs,
        "chat_history": state.get("chat_history", []),
        "thinking_steps": thinking
    }

def direct_answer(state: AgentState):
    thinking = state.get("thinking_steps", [])
    thinking.append("💬 直接回答（无需检索）")
    response = get_model().invoke(f"请友好地回答用户：{state['question']}")
    thinking.append("   → 回答已生成")
    return {
        "answer": response.content,
        "chat_history": state.get("chat_history", []),
        "thinking_steps": thinking
    }

def generate_answer(state: AgentState):
    thinking = state.get("thinking_steps", [])
    thinking.append("✍️ 正在生成回答...")
    
    docs = state.get("retrieved_docs", [])
    history = state.get("chat_history", [])
    intent = state.get("intent", "")  # ← 获取意图
    
    # ========== 新增：根据意图调用对应的 Skill ==========
    skill_result = ""
    if intent == "policy":
        # 如果是政策类，尝试调用奖学金和处分查询
        # 从问题中提取关键词
        question = state["question"]
        if "奖学金" in question or "奖学" in question:
            # 简单示例：假设绩点3.2，无处分
            # 实际可以从对话历史或用户信息中提取
            skill_result += query_scholarship.invoke({"grade": "大二", "gpa": 3.2, "punishment_count": 0})
            skill_result += "\n"
        if "处分" in question or "违纪" in question or "作弊" in question:
            skill_result += query_discipline.invoke({"rule_type": question})
            skill_result += "\n"
        if "时间" in question or "有效" in question or "修订" in question:
            skill_result += get_policy_time.invoke({"policy_name": question})
            skill_result += "\n"
    
    if skill_result:
        thinking.append(f"   → 调用了 Skills，得到额外信息")
    # =====================================================
    
    history_text = ""
    if history:
        for msg in history[-4:]:
            history_text += f"{msg['role']}: {msg['content']}\n"
        history_text = "对话历史：\n" + history_text + "\n"
    
    context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(docs)]) if docs else "没有找到相关资料。"
    
    # ========== 修改 Prompt，加入 Skill 结果 ==========
    # ========== 修改 Prompt，加入 Skill 结果 ==========
    # 先构造 skill 部分（避免 f-string 内包含反斜杠）
    skill_section = ""
    if skill_result:
        skill_section = f"\n【Skills工具返回的补充信息】\n{skill_result}\n"
    
    prompt = f"""
    {history_text}
    参考资料：
    {context}
    {skill_section}
    用户最新问题：{state["question"]}
    
    规则（必须遵守）：
    1. 优先使用【参考资料】中的内容回答，标注【来源：编号】
    2. 如果有【Skills工具返回的补充信息】，也可以参考，标注【来源：Skill】
    3. 如果用户的问题中有指代词（如"求推荐"、"那个"、"它"、"这个"），必须结合对话历史来理解
    4. 如果仍然不确定用户具体要什么，主动追问确认
    5. 不要在没有理解指代的情况下，做出泛泛的回答
    """    # ===================================================
    
    response = get_model().invoke(prompt)
    thinking.append("   → 回答已生成")
    
    return {
        "answer": response.content,
        "retrieved_docs": docs,
        "chat_history": state.get("chat_history", []),
        "thinking_steps": thinking
    }
    
# ---------- 路由 ----------
def route_by_intent(state: AgentState):
    intent = state.get("intent", "chat")
    if intent == "policy":
        return "retrieve_policy"
    elif intent == "major":
        return "retrieve_major"
    elif intent == "news":
        return "retrieve_news"
    else:
        return "direct_answer"

# ---------- 构建图 ----------
builder = StateGraph(AgentState)
builder.add_node("classify", classify_intent)
builder.add_node("retrieve_policy", retrieve_policy)
builder.add_node("retrieve_major", retrieve_major)
builder.add_node("retrieve_news", retrieve_news)
builder.add_node("direct_answer", direct_answer)
builder.add_node("generate_answer", generate_answer)

builder.set_entry_point("classify")

builder.add_conditional_edges("classify", route_by_intent, {
    "retrieve_policy": "retrieve_policy",
    "retrieve_major": "retrieve_major",
    "retrieve_news": "retrieve_news",
    "direct_answer": "direct_answer"
})

builder.add_edge("retrieve_policy", "generate_answer")
builder.add_edge("retrieve_major", "generate_answer")
builder.add_edge("retrieve_news", "generate_answer")
builder.add_edge("generate_answer", END)
builder.add_edge("direct_answer", END)

agent = builder.compile()

if __name__ == "__main__":
    print("终端测试模式")
    chat_history = []
    while True:
        q = input("你: ")
        if q.lower() in ["exit","退出"]: break
        result = agent.invoke({
            "question": q,
            "chat_history": chat_history,
            "thinking_steps": []
        })
        print("\n思考过程:")
        for s in result.get("thinking_steps", []):
            print("  ", s)
        print("\n助手:", result["answer"])
        chat_history.append({"role": "user", "content": q})
        chat_history.append({"role": "assistant", "content": result["answer"]})