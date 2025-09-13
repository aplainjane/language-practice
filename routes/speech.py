from flask import Blueprint, request, jsonify
import os
import base64
import time
import json
import wave
from utils.audio_utils import convert_to_vosk_compatible
from utils.model_utils import get_vosk_model

speech_bp = Blueprint('speech', __name__)

TEMP_VOICE_DIR = "tempvoice"
if not os.path.exists(TEMP_VOICE_DIR):
    os.makedirs(TEMP_VOICE_DIR)

@speech_bp.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    model = get_vosk_model()
    if model is None:
        return jsonify({"error": "语音识别模型未加载，请先下载模型"}), 400

    audio_data = request.json.get('audio')
    if not audio_data:
        return jsonify({"error": "没有接收到音频数据"}), 400

    try:
        if ',' in audio_data:
            audio_bytes = base64.b64decode(audio_data.split(',')[1])
        else:
            audio_bytes = base64.b64decode(audio_data)
    except Exception as e:
        return jsonify({"error": f"解码音频数据失败: {str(e)}"}), 400

    timestamp = int(time.time())
    temp_file = os.path.join(TEMP_VOICE_DIR, f"temp_audio_{timestamp}.wav")

    with open(temp_file, "wb") as f:
        f.write(audio_bytes)

    try:
        convert_to_vosk_compatible(temp_file, temp_file)
        with wave.open(temp_file, 'rb') as wf:
            sample_rate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())
            from vosk import KaldiRecognizer
            rec = KaldiRecognizer(model, sample_rate)
            rec.SetWords(True)
            rec.AcceptWaveform(audio_data)
            result = json.loads(rec.FinalResult())
    except Exception as e:
        return jsonify({"error": f"音频格式不正确: {str(e)}", "saved_file": temp_file}), 400

    if "text" in result and result["text"]:
        return jsonify({"text": result["text"], "saved_file": temp_file})
    else:
        return jsonify({"error": "未能识别语音内容", "saved_file": temp_file}), 400