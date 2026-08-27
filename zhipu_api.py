"""
智谱 AI API 直接调用模块（不依赖 zhipuai SDK）
=============================================
使用 requests 直接调用智谱 API，兼容所有 Python 版本。
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


def _get_api_key() -> str:
    """获取 API Key（优先级从高到低）：
    1. Streamlit session_state（用户在页面输入的 Key）
    2. 环境变量（本地开发）
    3. Streamlit Secrets（云端部署默认 Key）
    """
    try:
        import streamlit as st
        if hasattr(st, "session_state") and "user_api_key" in st.session_state:
            key = st.session_state.user_api_key
            if key and isinstance(key, str) and key.strip():
                return key.strip()
    except Exception:
        pass

    api_key = os.getenv("ZHIPU_API_KEY")
    if api_key:
        return api_key

    try:
        import streamlit as st
        if hasattr(st, "secrets") and "ZHIPU_API_KEY" in st.secrets:
            return st.secrets["ZHIPU_API_KEY"]
    except Exception:
        pass

    raise ValueError("未找到 ZHIPU_API_KEY，请在侧边栏输入你的 API Key，或在环境变量/Streamlit Secrets 中配置")


def chat_completions(
    messages: list,
    model: str = "glm-4-flash",
    temperature: float = 0.1,
    tools: list = None,
    tool_choice: dict = None,
    max_tokens: int = None,
    stream: bool = False,
) -> dict:
    """
    调用智谱聊天 API，返回解析后的响应 dict。

    返回格式（兼容 SDK 的结构）：
    {
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "...",
                "tool_calls": [...] or None
            },
            "finish_reason": "stop"
        }],
        "usage": {...}
    }
    """
    api_key = _get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if stream:
        payload["stream"] = True

    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def create_embedding(text: str, model: str = "embedding-2") -> list:
    """
    调用智谱 embedding API，返回向量列表。
    """
    api_key = _get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": text,
    }

    resp = requests.post(
        f"{BASE_URL}/embeddings",
        headers=headers,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]
