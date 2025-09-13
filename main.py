import serial
import requests
import json
import time
from pypinyin import pinyin, Style
import re
import threading

# Ollama DeepSeek R1 API
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# 串口配置

# 保存对话记录
msg_list = []

def call_deepseek(prompt):
    """调用 Ollama DeepSeek R1"""
    global msg_list
    try:
        msg_list.append({"role": "user", "content": prompt})
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": "deepseek-r1:8b",
                "messages": msg_list,
                "stream": False
            }
        )
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("message", {})
            content = generated_text.get("content", "错误：无响应")
            # 移除 <think> 和 </think> 标签
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = content.strip()
            msg_list.append({"role": "assistant", "content": content})
            return content
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return "错误：调用失败"
    except Exception as e:
        print(f"调用 DeepSeek 失败: {e}")
        return "错误：调用失败"



def main():
    # 尝试初始化串口
    while True:
        try:
            # 用户输入
            prompt = input("输入对话（或按 Ctrl+C 退出）: ")
            if not prompt:
                continue

            # 调用 DeepSeek R1
            response = call_deepseek(prompt)
            print(f"DeepSeek 响应: {response}")

        except KeyboardInterrupt:
            print("\n退出程序")
            break
        except Exception as e:
            print(f"错误: {e}")
            time.sleep(0.1)


if __name__ == "__main__":
    main()