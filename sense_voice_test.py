import os
import numpy as np
import scipy.signal as signal
import pyaudio
import re
from funasr import AutoModel

# 1. 满血加载本地模型
print("🚀 正在加载本地 SenseVoice 生产级状态机引擎...")
model = AutoModel(
    model="models/SenseVoiceSmall",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    ncpu=4
)

TAG_CLEANER = re.compile(r'<\|.*?\|>')

CHUNK = 1024          # 规整化数据块
FORMAT = pyaudio.paInt16
CHANNELS = 2          # 完美兼容你的环形麦克风阵列
RATE = 48000          # 48k 采样

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("\n🎤 【小宝听着呢】双指针流式引擎已点火！请正常说话 (按 Ctrl+C 退出)...")

try:
    # 工业级状态机变量
    collected_frames = []
    is_speaking = False
    silence_counter = 0

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # 矩阵转换：取左声道并重采样到 16000Hz 归一化 float32
        numpy_data = np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS)
        mono_data = numpy_data[:, 0]
        num_samples = int(len(mono_data) * 16000 / RATE)
        resampled_data = signal.resample(mono_data, num_samples)
        float32_data = (resampled_data / 32768.0).astype(np.float32)

        # 实时快速扫描单切片的音量能量 (均方根能量 RMS)
        rms = np.sqrt(np.mean(float32_data**2))
        
        # 触发阈值：大于 0.015 说明此时麦克风捕捉到了明确的人声起伏
        if rms > 0.015:
            if not is_speaking:
                is_speaking = True
                print("🎙️  [捕捉到人声起点...]")
            collected_frames.append(float32_data)
            silence_counter = 0
        else:
            if is_speaking:
                collected_frames.append(float32_data)
                silence_counter += 1
                
                # 当连续大约 15-20 个块（约 400~500毫秒）都是静音时，判定说话结束
                if silence_counter > 18:
                    print("🛑 [判定说话结束，正在瞬间解码...]")
                    full_audio = np.concatenate(collected_frames)
                    
                    # 交付阿里大模型一次性吞噬全量特征
                    res = model.generate(
                        input=full_audio,
                        cache={},
                        language="zh",
                        use_itn=True
                    )
                    
                    if res and len(res) > 0 and 'text' in res[0]:
                        raw_text = res[0]['text'].strip()
                        clean_text = TAG_CLEANER.sub('', raw_text).strip()
                        # 规整无用标点
                        clean_text = clean_text.replace(" ", "").replace(".", "")
                        
                        if clean_text and len(clean_text) > 1:
                            print(f"\n🗣️  小宝听到: {clean_text}\n")
                    
                    # 瞬间重置状态机，释放硬件显存
                    collected_frames = []
                    is_speaking = False
                    silence_counter = 0

except KeyboardInterrupt:
    print("\n🛑 停止流式监听。")
finally:
    stream.stop_stream(); stream.close(); p.terminate()