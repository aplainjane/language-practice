import wave
import sys
import json
import torch
import torchaudio
import os
from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(0)
MODEL_PATH = "models/vosk-model-en-us-0.22"
TEMP_OUTPUT = "tempvoice/converted_test.wav"

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

# 确保临时输出目录存在
os.makedirs(os.path.dirname(TEMP_OUTPUT), exist_ok=True)

# 检查输入参数
if len(sys.argv) < 2:
    print("请提供输入 WAV 文件路径！")
    sys.exit(1)

# 转换输入文件到 Vosk 兼容格式
try:
    convert_to_vosk_compatible(sys.argv[1], TEMP_OUTPUT)
except Exception as e:
    print(f"转换失败: {e}")
    sys.exit(1)

# 打开转换后的文件
try:
    wf = wave.open(TEMP_OUTPUT, "rb")
except wave.Error as e:
    print(f"无法打开转换后的文件: {e}")
    if os.path.exists(TEMP_OUTPUT):
        os.remove(TEMP_OUTPUT)
    sys.exit(1)

# 初始化 Vosk 模型和识别器
try:
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
except Exception as e:
    print(f"初始化 Vosk 模型失败: {e}")
    wf.close()
    if os.path.exists(TEMP_OUTPUT):
        os.remove(TEMP_OUTPUT)
    sys.exit(1)

results = []

# 读取并识别音频
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        results.append(result.get("text", ""))

# 添加最终结果
final_result = json.loads(rec.FinalResult())
results.append(final_result.get("text", ""))

# 输出完整识别文本
full_text = " ".join([r for r in results if r])
print("\n✅ 识别完成：")
print(full_text)

# 清理
wf.close()
if os.path.exists(TEMP_OUTPUT):
    os.remove(TEMP_OUTPUT)
    print(f"已删除临时文件: {TEMP_OUTPUT}")