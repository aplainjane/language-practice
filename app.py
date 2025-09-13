# app.py
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_cors import CORS
from chat_logic import call_deepseek
import os
import base64
import json
import wave
import numpy as np
from vosk import Model, KaldiRecognizer
import time
from pydub import AudioSegment
import torch
import torchaudio
import io

# 修改为支持静态文件
app = Flask(__name__, static_folder='static')
CORS(app)

# 确保静态文件目录存在
STATIC_JS_DIR = "static/js"
if not os.path.exists(STATIC_JS_DIR):
    os.makedirs(STATIC_JS_DIR)

# Vosk模型路径
MODEL_PATH = "models/vosk-model-en-us-0.22"

# 确保临时目录和录音保存目录存在
TEMP_VOICE_DIR = "tempvoice"

if not os.path.exists(TEMP_VOICE_DIR):
    os.makedirs(TEMP_VOICE_DIR)


# 加载Vosk模型
if not os.path.exists(MODEL_PATH):
    print(f"请下载Vosk模型并解压到: {MODEL_PATH}")
    print("下载地址: https://alphacephei.com/vosk/models")
    model = None
else:
    model = Model(MODEL_PATH)

def convert_to_vosk_compatible(input_path, output_path, target_sr=16000):
    """
    将输入音频转换为 Vosk 支持的格式：单声道，16 位 PCM，16kHz 采样率。
    
    Args:
        input_path (str): 输入音频文件路径
        output_path (str): 输出转换后的音频文件路径
        target_sr (int): 目标采样率，默认为 16000 Hz
    """
    try:
        # 加载音频
        waveform, sample_rate = torchaudio.load(input_path)
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 重新采样到目标采样率
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # 确保信号在 [-1, 1] 范围内，并转换为 16 位 PCM
        waveform = waveform.clamp(-1, 1) * 32767  # 映射到 16 位整数范围 [-32768, 32767]
        waveform = waveform.to(torch.int16)  # 转换为 16 位整数
        
        # 保存为 16 位 PCM WAV 文件
        torchaudio.save(
            output_path,
            waveform,
            target_sr,
            bits_per_sample=16,
            encoding="PCM_S"
        )
        
        # 验证转换后的文件
        with wave.open(output_path, "rb") as wf:
            if (wf.getnchannels() != 1 or
                wf.getsampwidth() != 2 or
                wf.getcomptype() != "NONE" or
                wf.getframerate() != target_sr):
                raise ValueError(
                    f"转换后的文件不符合 Vosk 要求："
                    f"声道数={wf.getnchannels()}, "
                    f"采样宽度={wf.getsampwidth()}, "
                    f"压缩类型={wf.getcomptype()}, "
                    f"采样率={wf.getframerate()}"
                )
        print(f"成功转换 {input_path} 为 Vosk 兼容格式：单声道，16 位 PCM，{target_sr}Hz")
        
    except Exception as e:
        print(f"音频转换失败: {e}")
        raise

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    response = call_deepseek(user_input)
    return jsonify({
        "reply": response,
    })

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    try:
        if model is None:
            return jsonify({"error": "语音识别模型未加载，请先下载模型"}), 400
            
        # 获取音频数据
        audio_data = request.json.get('audio')
        if not audio_data:
            return jsonify({"error": "没有接收到音频数据"}), 400
        
        # 解码base64音频数据
        try:
            # 处理不同格式的base64字符串
            if ',' in audio_data:
                mime_type = audio_data.split(',')[0]
                print(f"音频MIME类型: {mime_type}")
                audio_bytes = base64.b64decode(audio_data.split(',')[1])
            else:
                audio_bytes = base64.b64decode(audio_data)
            
            # 打印前几个字节，查看文件头
            print(f"音频文件头: {audio_bytes[:16]}")
        except Exception as e:
            return jsonify({"error": f"解码音频数据失败: {str(e)}"}), 400
        
        # 生成带时间戳的文件名，确保唯一性
        timestamp = int(time.time())
        temp_file = os.path.join(TEMP_VOICE_DIR, f"temp_audio.wav")
        
        # 保存原始音频数据（用于调试）
            
        # 尝试修复音频格式并保存为标准WAV
        try:
            # 检查是否已经是WAV格式
            is_wav = False
            if len(audio_bytes) > 12:
                if audio_bytes[0:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE':
                    is_wav = True
                    print("检测到标准WAV格式")
                else:
                    print(f"非WAV格式，前4字节: {audio_bytes[0:4]}，8-12字节: {audio_bytes[8:12]}")
            else:
                print(f"音频数据太短: {len(audio_bytes)} 字节")
            
            if is_wav:
                # 如果已经是WAV格式，直接保存
                with open(temp_file, "wb") as f:
                    f.write(audio_bytes)

            else:
                # 如果不是WAV格式，使用pydub转换
                print(f"音频不是标准WAV格式，使用pydub转换...")
                
                try:
                    # 尝试使用pydub加载音频
                    # 先保存为临时文件
                    temp_webm = os.path.join(TEMP_VOICE_DIR, f"temp_{timestamp}.webm")
                    with open(temp_webm, "wb") as f:
                        f.write(audio_bytes)
                    
                    # 使用pydub加载并转换
                    audio = AudioSegment.from_file(temp_webm)
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    
                    # 导出为WAV
                    audio.export(temp_file, format="wav")
                    
                    # 删除临时webm文件
                    os.remove(temp_webm)
                    
                    print("pydub转换成功")
                except Exception as e:
                    print(f"pydub转换失败: {e}")
                    # 如果pydub失败，尝试使用wave库
                    create_wav_file(audio_bytes, temp_file)
                
        except Exception as e:
            return jsonify({"error": f"保存音频文件失败: {str(e)}"}), 400
        
        try:
            # 使用wave库读取音频文件
            convert_to_vosk_compatible(temp_file, temp_file)
            with wave.open((temp_file), 'rb') as wf:
                # 获取音频参数
                sample_rate = wf.getframerate()
                print(f"音频采样率: {sample_rate}Hz, 通道数: {wf.getnchannels()}")
                
                # 读取所有音频数据
                audio_data = wf.readframes(wf.getnframes())
                
                # 创建识别器
                rec = KaldiRecognizer(model, sample_rate)
                rec.SetWords(True)
                
                # 识别
                rec.AcceptWaveform(audio_data)
                result = json.loads(rec.FinalResult())

        except Exception as e:
            # 如果wave库无法打开，记录错误但不删除文件（用于调试）
            error_msg = f"音频格式不正确: {str(e)}"
            print(error_msg)
            return jsonify({
                "error": error_msg,
                "saved_file": temp_file  # 返回保存的文件路径
            }), 400
        
        # 处理识别结果
        if "text" in result and result["text"]:
            return jsonify({
                "text": result["text"],
                "saved_file": temp_file  # 返回保存的文件路径
            })
        else:
            return jsonify({
                "error": "未能识别语音内容",
                "saved_file": temp_file  # 返回保存的文件路径
            }), 400
            
    except Exception as e:
        error_msg = f"处理语音时出错: {str(e)}"
        print(error_msg)
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

