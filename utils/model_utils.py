import os
from vosk import Model

MODEL_PATH = "models/vosk-model-en-us-0.22"
_model_instance = None

def get_vosk_model():
    global _model_instance
    if _model_instance is None:
        if not os.path.exists(MODEL_PATH):
            print(f"请下载Vosk模型并解压到: {MODEL_PATH}")
            print("下载地址: https://alphacephei.com/vosk/models")
            return None
        _model_instance = Model(MODEL_PATH)
    return _model_instance