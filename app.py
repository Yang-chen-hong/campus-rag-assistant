import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(__file__))

from agent_graph import agent
# 导入向量库连接对象，用于获取真实文档条数
from retriever import collection

# ========== 页面配置 ==========
st.set_page_config(
    page_title="湖南师范大学校园智能助手",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 湖南师范大学校园智能助手")
st.caption("基于 LangGraph 的校园 RAG 问答系统（带多轮记忆 + 思考过程展示）")

# ========== 侧边栏 ==========
with st.sidebar:
    st.header("⚙️ 控制面板")
    if st.button("🗑️ 清空聊天记录"):
        st.session_state.messages = []
        st.session_state.last_docs = []
        st.rerun()
    
    st.divider()
    
    st.header("📊 系统状态")
    st.success("✅ Agent 已就绪")
    # 显示真实知识库条数
    try:
        doc_count = collection.count()
        st.info(f"📚 知识库：{doc_count} 条真实文档")
    except:
        st.info("📚 知识库：已连接（待加载）")
    
    st.divider()
    
    st.header("📖 检索到的原始资料")
    if "last_docs" in st.session_state and st.session_state.last_docs:
        for i, doc in enumerate(st.session_state.last_docs):
            with st.expander(f"📄 资料 {i+1}"):
                st.write(doc)
    else:
        st.caption("暂无检索记录")

# ========== 初始化聊天历史 ==========
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []

# ========== 显示历史消息 ==========
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ========== 输入框 ==========
if prompt := st.chat_input("问我关于校园的问题..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        # 用容器显示思考步骤
        thinking_container = st.container()
        # 用容器显示最终答案
        answer_container = st.container()
        
        try:
            result = agent.invoke({
                "question": prompt,
                "chat_history": st.session_state.messages,
                "thinking_steps": []
            })
            
            # 显示思考步骤
            with thinking_container:
                st.markdown("**🧠 思考过程：**")
                for step in result.get("thinking_steps", []):
                    st.text(step)
            
            # 显示答案
            answer = result.get("answer", "抱歉，我没有找到答案。")
            docs = result.get("retrieved_docs", [])
            if docs:
                st.session_state.last_docs = docs
            
            with answer_container:
                st.markdown(answer)
            
            # 保存到历史
            msg_data = {"role": "assistant", "content": answer}
            if docs:
                msg_data["doc_count"] = len(docs)
            st.session_state.messages.append(msg_data)
            
            st.rerun()
            
        except Exception as e:
            st.error(f"出错了：{e}")