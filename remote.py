from pyngrok import ngrok

# 建立隧道，暴露本地的 8501 端口
public_url = ngrok.connect(8501)
print(f"✅ 公网链接已生成，复制发给队友：{public_url}")
print("⚠️ 保持此窗口运行，关闭后链接失效")
input("按回车键退出...")