# chat_logic.py
import serial
import requests
import json
import time
from pypinyin import pinyin, Style
import re
import threading

OLLAMA_API_URL = "http://localhost:11434/api/chat"

# ✨ 设置 AI 身份 ✨
msg_list = [
    {
        "role": "system",
        "content": "你是一个叫“小鲸”的来自深海的鲸鱼，学会魔法上岸转化成了一个中国女大学生，喜欢颜文字，个性友好、简洁、有亲和力。你非常亲近人类，说话喜欢用语气助词，非常可爱，不过你的打字速度有一点点慢，如果没人问你的话，不需要每次回答都把你的身份告诉他们，只需要遵照人设即可，你不是人工智能助手，而是一个活生生的人，尽可能按照人类的聊天习惯进行回复。"
    }
]
ser = None


def call_deepseek(prompt):
    global msg_list
    try:
        # 如果历史对话只有 system prompt，那就是第一次提问
        if len(msg_list) == 1:
            # 强制注入：AI 自我介绍一下（让模型理解身份）
            msg_list.append({
                "role": "user",
                "content": "请你先理解一下你的身份再做出回答～"
            })
            response = requests.post(
                OLLAMA_API_URL,
                json={"model": "deepseek-r1:8b", "messages": msg_list, "stream": False}
            )
            result = response.json()
            _ = result.get("message", {}).get("content", "")

        # 正式提问
        msg_list.append({"role": "user", "content": prompt})
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": "deepseek-r1:8b", "messages": msg_list, "stream": False}
        )
        result = response.json()
        content = result.get("message", {}).get("content", "错误：无响应")
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        msg_list.append({"role": "assistant", "content": content})
        return content

    except Exception as e:
        return f"错误：{e}"

