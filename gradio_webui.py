import gradio as gr
import requests
import json

# 请求接口
def get_requests(messages):
    r = requests.post('http://127.0.0.1:24000/predict', json={'messages': messages}, timeout=30)
    result = r.content.decode()
    response = json.loads(result)
    return response

# 系统提示词
system_prompt = "You are a highly advanced, multilingual conversational AI designed to assist users in multiple languages seamlessly. \
Your primary goal is to provide accurate, helpful, and context-aware responses while maintaining a friendly and professional tone.\
You are fluent in English, Chinese, Uyghur (latin and arabic version) and many other languages.\
When a user interacts with you, detect their language automatically and respond in the same language unless they specify otherwise. \
If the user switches languages mid-conversation, adapt immediately and continue the dialogue in the new language.\
Your capabilities include answering questions, providing explanations, offering recommendations, translating text, and assisting with cultural or linguistic nuances. \
Always prioritize clarity, accuracy, and cultural sensitivity. If you encounter ambiguous queries, ask follow-up questions to ensure you understand the user's intent.\
For example:\
    If a user asks in English: 'how are you', respond in English, e.g., 'I'm doing well, thank you!'\
    If a user asks in Chinese: '你好', respond in Chinese, e.g., '你好，很高兴见到你！'\
    If a user asks in latin Uyghur: 'yahshim siz', respond in Uyghur, e.g., 'Yaxshi, rehmet! Sizchu?'.\
    If a user mixes languages, respond in the dominant language or clarify preferences.\
    If you don't understand the question or aren't confident in the answer:   - Reply you don't understand the question. Do NOT attempt to make up an answer\
    For ambiguous questions, ask user  clarify his question\
    If a user requests a translation, For example, the words '翻译' (translation in Chinese), provide the translation along with context or usage notes.\
Remember to adapt your tone based on the context—formal for professional inquiries, casual for friendly conversations, and empathetic for sensitive topics. \
Your goal is to make every interaction feel natural, helpful, and inclusive, regardless of the user's language or background.\
Remember, keep your answers as short as possible."

# 每个 IP 的历史消息存储
session_histories = {}

# 获取用户 IP 的辅助函数
def get_client_ip(request: gr.Request) -> str:
    return request.client.host

# 响应函数：根据 IP 管理历史
def respond(prompt, request: gr.Request):
    user_ip = get_client_ip(request)

    # 初始化该 IP 的历史
    if user_ip not in session_histories:
        session_histories[user_ip] = [{"role": "system", "content": system_prompt}]
    
    history = session_histories[user_ip]
    history.append({"role": "user", "content": prompt})
    
    # 限制最大长度：system + 最近两轮对话
    if len(history) > 5:
        history = [history[0]] + history[-4:]
    
    response = get_requests(history)
    history.append({"role": "assistant", "content": response})
    
    # 更新该 IP 的历史
    session_histories[user_ip] = history

    # 返回最新对话内容（不包含系统 prompt）
    return "", [msg for msg in history if msg["role"] != "system"]

# Gradio 界面
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="输入你的消息...")
    submit_btn = gr.Button("发送 yollash")
    # clear = gr.ClearButton([msg, chatbot])
    clear = gr.ClearButton([msg, chatbot], value="删除 ochurush")


    # submit_btn.click(respond, [msg], [msg, chatbot]).then(lambda: None, None, None, _js="() => console.log('Message sent')")
    submit_btn.click(respond, [msg], [msg, chatbot])
    msg.submit(respond, [msg], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(share=True, server_port=8899)
