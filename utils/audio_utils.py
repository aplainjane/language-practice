import torchaudio
import torch
import wave

def convert_to_vosk_compatible(input_path, output_path, target_sr=16000):
    waveform, sample_rate = torchaudio.load(input_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    waveform = waveform.clamp(-1, 1) * 32767
    waveform = waveform.to(torch.int16)
    torchaudio.save(output_path, waveform, target_sr, bits_per_sample=16, encoding="PCM_S")
    with wave.open(output_path, "rb") as wf:
        if (wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE" or wf.getframerate() != target_sr):
            raise ValueError("转换后的文件不符合 Vosk 要求")