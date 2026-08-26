"""
湖南师范大学校园智能助手 - Streamlit 前端 v2.0
==================================================
升级内容：
  1. 流式输出（打字机效果）
  2. 工具调用过程可视化（思考面板）
  3. 引用来源卡片（可展开查看原文）
  4. 快捷提问按钮
  5. 更专业的 UI 设计
  6. 错误提示优化
"""

import streamlit as st
import sys
import os
import time

sys.path.append(os.path.dirname(__file__))

from agent_graph import agent_invoke, reset_client
from retriever import get_collection_count, reset_clients

# ========== 页面配置 ==========
st.set_page_config(
    page_title="湖南师范大学校园智能助手",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义 CSS
st.markdown("""
<style>
    /* 主色调 */
    :root {
        --primary: #1e88e5;
        --primary-light: #e3f2fd;
    }

    /* 隐藏 Streamlit 默认菜单和页脚 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* 标题样式 */
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e88e5, #42a5f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    /* 聊天气泡 */
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        line-height: 1.6;
    }
    .user-message {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 4px solid #1e88e5;
    }
    .assistant-message {
        background: #fafafa;
        border-left: 4px solid #66bb6a;
    }

    /* 工具调用指示器 */
    .tool-call-badge {
        display: inline-block;
        background: #fff3e0;
        color: #e65100;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 6px;
    }

    /* 快捷按钮 */
    .quick-btn {
        background: white !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 20px !important;
        padding: 6px 14px !important;
        font-size: 0.85rem !important;
        transition: all 0.2s !important;
    }
    .quick-btn:hover {
        border-color: #1e88e5 !important;
        color: #1e88e5 !important;
        background: #e3f2fd !important;
    }

    /* 引用来源卡片 */
    .source-card {
        background: #f5f5f5;
        border-radius: 8px;
        padding: 8px 12px;
        margin-top: 8px;
        font-size: 0.85rem;
    }

    /* 思考过程折叠面板 */
    .thinking-step {
        padding: 4px 0;
        font-size: 0.85rem;
        color: #666;
    }

    /* 状态指示器 */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    .status-active {
        background: #4caf50;
    }
    .status-idle {
        background: #9e9e9e;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* 欢迎卡片 */
    .welcome-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }

    /* 引用角标 */
    .ref-badge {
        display: inline-block;
        background: #e8f5e9;
        color: #2e7d32;
        font-size: 0.7rem;
        padding: 1px 6px;
        border-radius: 4px;
        font-weight: 600;
        margin: 0 2px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ========== 侧边栏 ==========
with st.sidebar:
    # Logo 和标题
    st.markdown("## 🎓 校园助手")
    st.caption("湖南师范大学 · 智能问答")

    st.divider()

    # API Key 设置
    st.markdown("### 🔑 API Key")
    api_key_input = st.text_input(
        "智谱 API Key",
        type="password",
        placeholder="请输入你的 API Key",
        help="在 https://open.bigmodel.cn/ 注册获取",
        key="api_key_input",
        label_visibility="collapsed"
    )

    if api_key_input:
        st.session_state.user_api_key = api_key_input.strip()
        reset_clients()
        reset_client()
        st.success("✅ 已设置")
    else:
        st.warning("⚠️ 请先输入 API Key")
        if "user_api_key" in st.session_state:
            del st.session_state.user_api_key

    st.divider()

    # 系统状态
    st.markdown("### 📊 系统状态")
    has_key = "user_api_key" in st.session_state and st.session_state.user_api_key
    if has_key:
        st.markdown('<span class="status-dot status-active"></span> Agent 在线', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot status-idle"></span> 等待输入 Key', unsafe_allow_html=True)

    try:
        doc_count = get_collection_count()
        st.markdown(f"📚 知识库：**{doc_count}** 条文档")
    except Exception:
        st.markdown("📚 知识库：连接中...")

    st.divider()

    # 控制面板
    st.markdown("### ⚙️ 设置")
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

    show_thinking = st.toggle("显示思考过程", value=True)

    st.divider()

    # 版本信息
    st.caption("v2.0 · ReAct Agent · 混合检索")


# ========== 主界面 ==========
st.markdown('<div class="main-title">🎓 湖南师范大学校园智能助手</div>', unsafe_allow_html=True)
st.caption("基于 ReAct Agent + 混合检索 + Rerank 的校园 RAG 问答系统")

# 初始化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# 欢迎卡片（没有消息时显示）
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <h3>👋 你好！我是师大校园助手</h3>
        <p style="color:#555; margin: 0.5rem 0;">
            我可以帮你查询学校政策、专业介绍、奖学金评定、处分规定等各类校园信息。
        </p>
        <p style="color:#777; font-size: 0.9rem; margin-top: 1rem;">
            💡 试试问我：
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 快捷提问按钮
    col1, col2, col3, col4 = st.columns(4)
    quick_questions = [
        "奖学金评定条件是什么？",
        "考试作弊有什么后果？",
        "挂科了怎么办？",
        "学校有哪些专业？",
    ]
    cols = [col1, col2, col3, col4]
    for i, (col, q) in enumerate(zip(cols, quick_questions)):
        with col:
            if st.button(q, key=f"quick_{i}", use_container_width=True):
                st.session_state.quick_question = q
                st.rerun()

# 处理快捷提问
if "quick_question" in st.session_state and st.session_state.quick_question:
    q = st.session_state.quick_question
    st.session_state.quick_question = None
    st.session_state.messages.append({"role": "user", "content": q})
    # 下面会自动处理


# ========== 显示历史消息 ==========
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        # 如果有助手消息带引用来源
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown("---")
            st.markdown("**📄 参考来源：**")
            for i, src in enumerate(msg["sources"]):
                with st.expander(f"来源 {i+1}：{src.get('title', '未知')}"):
                    st.markdown(src.get("content", ""))


# ========== 输入框 ==========
has_api_key = "user_api_key" in st.session_state and st.session_state.user_api_key

if prompt := st.chat_input("问我关于校园的问题...", disabled=not has_api_key):
    if not has_api_key:
        st.info("请先在左侧输入你的智谱 API Key")
        st.stop()

    # 保存用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # 生成回复
    with st.chat_message("assistant", avatar="🤖"):
        # 思考过程容器
        if show_thinking:
            thinking_expander = st.expander("🧠 思考过程", expanded=True)
        else:
            thinking_expander = None

        # 回答容器
        answer_placeholder = st.empty()

        try:
            # 显示思考中状态
            if thinking_expander:
                with thinking_expander:
                    thinking_placeholder = st.empty()
                    thinking_placeholder.markdown("⏳ 正在思考...")

            # 调用 Agent
            result = agent_invoke(
                question=prompt,
                chat_history=st.session_state.messages[:-1],  # 排除刚加的用户消息
            )

            # 显示思考过程
            if thinking_expander:
                with thinking_expander:
                    steps_html = ""
                    for step in result.get("thinking_steps", []):
                        steps_html += f'<div class="thinking-step">{step}</div>'
                    thinking_placeholder.markdown(steps_html, unsafe_allow_html=True)

                    # 显示工具调用
                    tool_calls = result.get("tool_calls", [])
                    if tool_calls:
                        st.markdown("---")
                        st.markdown("**🔧 调用的工具：**")
                        for tc in tool_calls:
                            tool_name = tc["tool"]
                            tool_args = tc.get("args", {})
                            args_str = ", ".join(f"{k}={v}" for k, v in tool_args.items())
                            st.markdown(f'- <span class="tool-call-badge">{tool_name}</span> {args_str}', unsafe_allow_html=True)

            # 流式输出回答（模拟打字机效果）
            answer = result.get("answer", "抱歉，我没有找到答案。")
            displayed = ""
            words = list(answer)
            for i in range(len(words)):
                displayed += words[i]
                if i % 3 == 0 or i == len(words) - 1:  # 每3个字刷新一次
                    answer_placeholder.markdown(displayed + "▌")
                    time.sleep(0.01)

            # 最终显示（去掉光标）
            answer_placeholder.markdown(answer)

            # 提取引用来源（从检索结果中解析）
            sources = []
            retrieved_docs = result.get("retrieved_docs", [])
            for doc_text in retrieved_docs:
                # 简单解析来源标题和内容
                lines = doc_text.split("\n")
                title = "未知来源"
                content = doc_text
                for line in lines:
                    if line.startswith("【来源"):
                        # 提取标题
                        import re
                        match = re.search(r'《(.+?)》', line)
                        if match:
                            title = match.group(1)
                        break
                sources.append({"title": title, "content": content})

            # 显示引用来源
            if sources:
                st.markdown("---")
                st.markdown("**📄 参考来源：**")
                for i, src in enumerate(sources[:5]):  # 最多显示5个
                    with st.expander(f"来源 {i+1}：{src.get('title', '未知')}"):
                        st.markdown(src.get("content", "")[:800])

            # 保存到历史
            msg_data = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "tool_calls": result.get("tool_calls", []),
            }
            st.session_state.messages.append(msg_data)
            st.session_state.last_result = result

        except Exception as e:
            answer_placeholder.error(f"出错了：{e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"抱歉，出错了：{e}",
            })

    st.rerun()
