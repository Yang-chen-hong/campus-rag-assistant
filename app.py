"""
湖南师范大学信科学院智能助手 - Streamlit 前端 v3.0
==================================================
核心设计：
  1. 深色/明亮主题切换（运行时动态切换）
  2. 左侧数据面板（数据库统计、文档分类、Skill列表）
  3. 流式输出 + 思考过程可视化
  4. 引用来源卡片 + 快捷提问
"""

import streamlit as st
import sys
import os
import time
import json

sys.path.append(os.path.dirname(__file__))

from agent_graph import agent_invoke, reset_client
from retriever import get_collection_count, reset_clients
from init_db import init_database, ALL_DOCS
import init_db
from skills_tools import registry

# ========== 页面配置 ==========
st.set_page_config(
    page_title="信科学院智能助手",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== 数据库初始化（延迟到用户输入API Key后） ==========

# ========== 主题系统 ==========
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

DARK_CSS = """
<style>
/* ========== 深色主题 ========== */
:root {
    --bg-primary: #0f1117;
    --bg-secondary: #1a1d29;
    --bg-card: #1e2233;
    --bg-hover: #252a3d;
    --text-primary: #e4e6eb;
    --text-secondary: #9ca3af;
    --text-muted: #6b7280;
    --accent: #5b9fff;
    --accent-light: rgba(91, 159, 255, 0.15);
    --accent-glow: rgba(91, 159, 255, 0.4);
    --success: #4ade80;
    --warning: #fbbf24;
    --danger: #f87171;
    --border: #2d3142;
    --border-light: #3d4154;
    --shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.stApp {
    background: var(--bg-primary);
    color: var(--text-primary);
}

#MainMenu, footer {visibility: hidden;}

/* 侧边栏 */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .stMarkdown, 
section[data-testid="stSidebar"] .stText {
    color: var(--text-primary) !important;
}

/* 主标题 */
.main-title {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #5b9fff, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

/* 数据卡片 */
.data-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    transition: all 0.2s;
}
.data-card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 12px var(--accent-glow);
}

/* 统计数字 */
.stat-number {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
}
.stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 4px;
}

/* Skill 标签 */
.skill-tag {
    display: inline-block;
    background: var(--accent-light);
    color: var(--accent);
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px;
    border: 1px solid rgba(91, 159, 255, 0.3);
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
.status-active { background: var(--success); box-shadow: 0 0 6px var(--success); }
.status-idle { background: var(--text-muted); }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* 欢迎卡片 */
.welcome-card {
    background: linear-gradient(135deg, rgba(91, 159, 255, 0.1), rgba(167, 139, 250, 0.1));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

/* 思考步骤 */
.thinking-step {
    padding: 3px 0;
    font-size: 0.82rem;
    color: var(--text-secondary);
}

/* 工具标签 */
.tool-call-badge {
    display: inline-block;
    background: rgba(251, 191, 36, 0.15);
    color: var(--warning);
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.72rem;
    font-weight: 500;
    margin-right: 6px;
}

/* 主题切换按钮 */
.theme-toggle {
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 999;
}

/* 分类条形图 */
.category-bar {
    height: 6px;
    border-radius: 3px;
    background: var(--bg-hover);
    margin-top: 4px;
    overflow: hidden;
}
.category-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}

/* 输入框 */
.stChatInput textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 12px !important;
}

/* 聊天消息 */
.stChatMessage {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* expander */
details {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* 滚动条 */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
"""

LIGHT_CSS = """
<style>
/* ========== 明亮主题 v3 — 全面覆盖所有组件 ========== */
:root {
    --bg-primary: #edf0f6;
    --bg-secondary: #f5f7fa;
    --bg-card: #ffffff;
    --bg-hover: #e5e9f2;
    --bg-input: #ffffff;
    --text-primary: #1a202c;
    --text-secondary: #374151;
    --text-muted: #6b7280;
    --accent: #4f46e5;
    --accent-light: rgba(79, 70, 229, 0.08);
    --accent-glow: rgba(79, 70, 229, 0.25);
    --accent-dark: #3730a3;
    --success: #059669;
    --warning: #b45309;
    --danger: #b91c1c;
    --border: #cbd5e0;
    --border-light: #e2e8f0;
    --shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-hover: 0 4px 12px rgba(79, 70, 229, 0.15);
}

/* ===== 全局 ===== */
.stApp {
    background: var(--bg-primary);
    color: var(--text-primary) !important;
}
.stApp p,
.stApp span,
.stApp div,
.stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: var(--text-primary);
}
#MainMenu, footer {visibility: hidden;}

/* ===== 侧边栏全面修复 ===== */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 2px solid var(--border) !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.04);
}
/* 侧边栏所有文字元素 */
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] .stMarkdown div,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown h4,
section[data-testid="stSidebar"] .stText,
section[data-testid="stSidebar"] .stText p,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stCaption p,
section[data-testid="stSidebar"] .stCaption span,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] label {
    color: var(--text-primary) !important;
    opacity: 1 !important;
}
/* 侧边栏次要文字 */
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stCaption p,
section[data-testid="stSidebar"] small {
    color: var(--text-secondary) !important;
}

/* ===== 主标题 ===== */
.main-title {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

/* ===== 数据卡片 ===== */
.data-card {
    background: var(--bg-card);
    border: 1px solid var(--border-light);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: var(--shadow);
    transition: all 0.2s;
}
.data-card:hover {
    border-color: var(--accent);
    box-shadow: var(--shadow-hover);
    transform: translateY(-1px);
}

.stat-number {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
}
.stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 4px;
    font-weight: 500;
}

/* ===== Skill 标签 ===== */
.skill-tag {
    display: inline-block;
    background: var(--accent-light);
    color: var(--accent-dark) !important;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px;
    border: 1px solid rgba(79, 70, 229, 0.25);
    opacity: 1 !important;
}

/* ===== 状态指示器 ===== */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
.status-active { background: var(--success); box-shadow: 0 0 4px var(--success); }
.status-idle { background: var(--text-muted); }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ===== 欢迎卡片 ===== */
.welcome-card {
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.08), rgba(124, 58, 237, 0.08));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
}
.welcome-card p {
    color: var(--text-secondary) !important;
}

/* ===== 思考步骤 ===== */
.thinking-step {
    padding: 3px 0;
    font-size: 0.82rem;
    color: var(--text-secondary) !important;
    opacity: 1 !important;
}

/* ===== 工具标签 ===== */
.tool-call-badge {
    display: inline-block;
    background: rgba(180, 83, 9, 0.12);
    color: var(--warning) !important;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
    border: 1px solid rgba(180, 83, 9, 0.25);
    opacity: 1 !important;
}

/* ===== 聊天输入框（全面修复） ===== */
[data-testid="stBottom"] {
    background: var(--bg-primary) !important;
}
[data-testid="stBottom"] > div {
    background: var(--bg-primary) !important;
}
[data-testid="stChatInputContainer"] {
    background: var(--bg-primary) !important;
}
[data-testid="stChatInputContainer"] > div {
    background: var(--bg-primary) !important;
}
.stChatInput {
    background: var(--bg-primary) !important;
}
.stChatInput > div {
    background: var(--bg-input) !important;
    border: 2px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow) !important;
}
.stChatInput textarea {
    background: var(--bg-input) !important;
    color: var(--text-primary) !important;
    border: none !important;
    box-shadow: none !important;
    opacity: 1 !important;
}
.stChatInput textarea::placeholder {
    color: var(--text-muted) !important;
    opacity: 0.7;
}
.stChatInput textarea:focus {
    outline: none !important;
}
.stChatInput > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
.stChatInput button {
    color: var(--accent) !important;
    background: transparent !important;
}

/* ===== 聊天气泡（全面修复文字颜色） ===== */
.stChatMessage {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow) !important;
}
.stChatMessage .stMarkdown,
.stChatMessage .stMarkdown p,
.stChatMessage .stMarkdown span,
.stChatMessage .stMarkdown div,
.stChatMessage .stMarkdown li,
.stChatMessage .stMarkdown h1,
.stChatMessage .stMarkdown h2,
.stChatMessage .stMarkdown h3,
.stChatMessage .stMarkdown h4,
.stChatMessage .stMarkdown strong,
.stChatMessage .stMarkdown em,
.stChatMessage .stMarkdown code,
.stChatMessage .stMarkdown a,
.stChatMessage p,
.stChatMessage span,
.stChatMessage div {
    color: var(--text-primary) !important;
    opacity: 1 !important;
}
.stChatMessage strong {
    font-weight: 700 !important;
}
.stChatMessage code {
    background: var(--bg-hover) !important;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
}
.stChatMessage a {
    color: var(--accent) !important;
    text-decoration: underline;
}

/* ===== Expander ===== */
details {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    box-shadow: var(--shadow) !important;
}
details summary,
details summary p,
details summary span {
    color: var(--text-primary) !important;
    opacity: 1 !important;
}
details .stMarkdown p,
details .stMarkdown span,
details p,
details span,
details div {
    color: var(--text-primary) !important;
    opacity: 1 !important;
}

/* ===== 分类条形图 ===== */
.category-bar {
    height: 6px;
    border-radius: 3px;
    background: var(--bg-hover);
    margin-top: 4px;
    overflow: hidden;
}
.category-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}

/* ===== 按钮 ===== */
.stButton > button {
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    color: var(--text-secondary) !important;
    background: var(--bg-card) !important;
    transition: all 0.2s !important;
    opacity: 1 !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-light) !important;
}
.stButton > button:disabled {
    background: var(--accent) !important;
    color: white !important;
    border-color: var(--accent) !important;
    opacity: 0.7 !important;
}

/* ===== Toggle 开关 ===== */
.stToggle label,
.stToggle p,
.stToggle span {
    color: var(--text-primary) !important;
    opacity: 1 !important;
}

/* ===== 文本输入 ===== */
.stTextInput input {
    background: var(--bg-input) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    opacity: 1 !important;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
.stTextInput label,
.stTextInput p {
    color: var(--text-primary) !important;
    opacity: 1 !important;
}

/* ===== Caption 小字 ===== */
.stCaption,
.stCaption p,
.stCaption span,
[data-testid="stCaptionContainer"] {
    color: var(--text-secondary) !important;
    opacity: 1 !important;
}

/* ===== 分隔线 ===== */
hr {
    border-color: var(--border) !important;
    opacity: 1 !important;
}

/* ===== 链接 ===== */
a {
    color: var(--accent) !important;
}

/* ===== 滚动条 ===== */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ===== 主内容区文字兜底 ===== */
.main .block-container {
    color: var(--text-primary) !important;
}
.main .block-container p,
.main .block-container span,
.main .block-container div,
.main .block-container li {
    color: var(--text-primary) !important;
    opacity: 1 !important;
}
</style>
"""

# 根据主题选择CSS
if st.session_state.theme == "dark":
    st.markdown(DARK_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)


# ========== 主题切换按钮 ==========
col_title, col_theme = st.columns([8, 2])
with col_title:
    st.markdown('<div class="main-title">🎓 信科学院智能助手</div>', unsafe_allow_html=True)
    st.caption("湖南师范大学 · 信息科学与工程学院")
with col_theme:
    theme_col1, theme_col2 = st.columns(2)
    with theme_col1:
        if st.button("🌙 深色", use_container_width=True,
                     key="dark_btn",
                     disabled=(st.session_state.theme == "dark")):
            st.session_state.theme = "dark"
            st.rerun()
    with theme_col2:
        if st.button("☀️ 明亮", use_container_width=True,
                     key="light_btn",
                     disabled=(st.session_state.theme == "light")):
            st.session_state.theme = "light"
            st.rerun()


# ========== 左侧数据面板 ==========
with st.sidebar:
    st.markdown("## 🎓 信科助手")
    st.caption("信息科学与工程学院 · 智能问答")

    st.divider()

    # ---- 数据统计面板 ----
    st.markdown("### 📊 数据面板")

    # 文档总数
    try:
        doc_count = get_collection_count()
    except Exception:
        doc_count = 0

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"""
        <div class="data-card">
            <div class="stat-number">{doc_count}</div>
            <div class="stat-label">知识库文档</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""
        <div class="data-card">
            <div class="stat-number">{len(ALL_DOCS)}</div>
            <div class="stat-label">结构化文档</div>
        </div>
        """, unsafe_allow_html=True)

    # 文档分类统计
    st.markdown("#### 📁 文档分类")
    categories = {
        "学校概况": len(init_db.SCHOOL_INTRO_DOCS),
        "学院介绍": len(init_db.COLLEGE_DOCS),
        "教务指南": len(init_db.ACADEMIC_GUIDE_DOCS),
        "政策文档": len(init_db.POLICY_DOCS),
        "校园生活": len(init_db.CAMPUS_LIFE_DOCS),
        "学生FAQ": len(init_db.FAQ_DOCS),
        "更多服务": len(init_db.MORE_SERVICES_DOCS),
        "实用指南": len(init_db.STUDENT_GUIDE_DOCS),
    }
    max_count = max(categories.values())
    cat_colors = ["#5b9fff", "#a78bfa", "#4ade80", "#fbbf24",
                  "#f87171", "#22d3ee", "#fb923c", "#e879f9"]

    for i, (cat, count) in enumerate(categories.items()):
        pct = int(count / max_count * 100)
        color = cat_colors[i % len(cat_colors)]
        st.markdown(f"""
        <div style="margin-bottom: 6px;">
            <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
                <span>{cat}</span>
                <span style="color: var(--text-muted);">{count}篇</span>
            </div>
            <div class="category-bar">
                <div class="category-bar-fill" style="width:{pct}%; background:{color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ---- Skill 面板 ----
    st.markdown("### 🔧 Skill 工具箱")
    st.caption(f"共 {len(registry.all())} 个 Skill")

    skill_categories = {}
    for skill in registry.all():
        cat = skill.category
        if cat not in skill_categories:
            skill_categories[cat] = []
        skill_categories[cat].append(skill.name)

    for cat, skills in skill_categories.items():
        tags = "".join(f'<span class="skill-tag">{s}</span>' for s in skills)
        st.markdown(f"""
        <div class="data-card" style="padding:10px;">
            <div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:4px;">{cat}</div>
            {tags}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ---- 系统状态 ----
    st.markdown("### ⚡ 系统状态")

    has_key = "user_api_key" in st.session_state and st.session_state.user_api_key
    if has_key:
        if st.session_state.get("db_initialized", False):
            st.markdown('<span class="status-dot status-active"></span> Agent 在线 · 知识库就绪', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-dot status-idle"></span> 等待 API Key', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot status-idle"></span> 等待 API Key', unsafe_allow_html=True)

    # 模型信息
    st.markdown("""
    <div class="data-card" style="padding:10px;">
        <div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:4px;">模型信息</div>
        <div style="font-size:0.82rem;">
            🧠 GLM-4-Flash<br/>
            📐 embedding-2 (1024维)<br/>
            🗄️ ChromaDB 本地
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ---- API Key 设置 ----
    st.markdown("### 🔑 API Key")
    api_key_input = st.text_input(
        "智谱 API Key",
        type="password",
        placeholder="粘贴你的 API Key",
        help="在 https://open.bigmodel.cn/ 注册获取",
        key="api_key_input",
        label_visibility="collapsed"
    )

    if api_key_input:
        st.session_state.user_api_key = api_key_input.strip()
        reset_clients()
        reset_client()
        st.success("✅ Key 已设置")

        # 首次设置Key时触发数据库初始化
        if not st.session_state.get("db_initialized", False):
            with st.spinner("正在初始化知识库，首次需要1-3分钟..."):
                try:
                    init_database()
                    st.session_state.db_initialized = True
                    st.rerun()
                except Exception:
                    st.session_state.db_initialized = True
                    st.rerun()
    else:
        st.warning("⚠️ 请输入 API Key")
        if "user_api_key" in st.session_state:
            del st.session_state.user_api_key

    st.divider()

    # ---- 控制面板 ----
    st.markdown("### ⚙️ 设置")

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

    show_thinking = st.toggle("显示思考过程", value=True)
    show_sources = st.toggle("显示引用来源", value=True)

    st.divider()
    st.caption(f"v4.0 · 统一Skill · 10工具 · {len(ALL_DOCS)}文档")


# ========== 主界面 ==========

# 初始化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# 欢迎卡片（没有消息时显示）
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <h3>👋 你好！我是信科学院智能助手</h3>
        <p style="color: var(--text-secondary); margin: 0.5rem 0;">
            我可以帮你查询 <b>信科学院专业介绍、师资队伍、课程信息、奖学金评定、保研条件、就业方向</b> 等。<br/>
            也支持 <b>GPA计算、毕业检查、学费查询</b> 等实用功能。
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 快捷提问
    col1, col2, col3, col4 = st.columns(4)
    quick_questions = [
        "信科学院有哪些专业？",
        "保研需要什么条件？",
        "计算机科学与技术专业课程有哪些？",
        "学院师资力量怎么样？",
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

# ========== 显示历史消息 ==========
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

        # 助手消息附带信息
        if msg["role"] == "assistant":
            # 工具调用
            if msg.get("tool_calls"):
                with st.expander(f"🔧 调用了 {len(msg['tool_calls'])} 个工具"):
                    for tc in msg["tool_calls"]:
                        tool_name = tc["tool"]
                        tool_args = tc.get("args", {})
                        args_str = ", ".join(f"{k}={v}" for k, v in tool_args.items())
                        st.markdown(f'- <span class="tool-call-badge">{tool_name}</span> {args_str}',
                                   unsafe_allow_html=True)

            # 引用来源
            if msg.get("sources") and show_sources:
                st.markdown("---")
                st.markdown(f"**📄 参考来源（{len(msg['sources'])}条）**")
                for i, src in enumerate(msg["sources"]):
                    with st.expander(f"来源 {i+1}：{src.get('title', '未知')}"):
                        st.markdown(src.get("content", "")[:800])


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

        answer_placeholder = st.empty()

        try:
            # 思考中
            if thinking_expander:
                with thinking_expander:
                    thinking_placeholder = st.empty()
                    thinking_placeholder.markdown("⏳ 正在分析问题...")

            # 调用 Agent
            result = agent_invoke(
                question=prompt,
                chat_history=st.session_state.messages[:-1],
            )

            # 显示思考过程
            if thinking_expander:
                with thinking_expander:
                    steps_html = ""
                    for step in result.get("thinking_steps", []):
                        steps_html += f'<div class="thinking-step">{step}</div>'
                    thinking_placeholder.markdown(steps_html, unsafe_allow_html=True)

                    tool_calls = result.get("tool_calls", [])
                    if tool_calls:
                        st.markdown("---")
                        st.markdown(f"**🔧 调用了 {len(tool_calls)} 个工具：**")
                        for tc in tool_calls:
                            tool_name = tc["tool"]
                            tool_args = tc.get("args", {})
                            args_str = ", ".join(f"{k}={v}" for k, v in tool_args.items())
                            st.markdown(f'- <span class="tool-call-badge">{tool_name}</span> {args_str}',
                                       unsafe_allow_html=True)

            # 流式输出
            answer = result.get("answer", "抱歉，我没有找到答案。")
            displayed = ""
            words = list(answer)
            for i in range(len(words)):
                displayed += words[i]
                if i % 3 == 0 or i == len(words) - 1:
                    answer_placeholder.markdown(displayed + "▌")
                    time.sleep(0.01)
            answer_placeholder.markdown(answer)

            # 解析引用来源
            import re
            sources = []
            retrieved_docs = result.get("retrieved_docs", [])
            for doc_text in retrieved_docs:
                lines = doc_text.split("\n")
                title = "未知来源"
                for line in lines:
                    if line.startswith("【来源"):
                        match = re.search(r'《(.+?)》', line)
                        if match:
                            title = match.group(1)
                        break
                sources.append({"title": title, "content": doc_text})

            # 显示引用
            if sources and show_sources:
                st.markdown("---")
                st.markdown(f"**📄 参考来源（{len(sources)}条）**")
                for i, src in enumerate(sources[:5]):
                    with st.expander(f"来源 {i+1}：{src.get('title', '未知')}"):
                        st.markdown(src.get("content", "")[:800])

            # 保存历史
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
