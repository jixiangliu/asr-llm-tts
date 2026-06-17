import os
import numpy as np
import scipy.signal as signal
import pyaudio
import re
from funasr import AutoModel

print("🚀 正在加载本地 SenseVoice 生产级无损状态机引擎...")
model = AutoModel(
    model="models/SenseVoiceSmall",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    ncpu=4,
    disable_update=True # 强开离线，避免启动检查卡顿
)

TAG_CLEANER = re.compile(r'<\|.*?\|>')

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("\n🎤 【小宝听着呢】高敏无损预录引擎已点火！请正常说话 (按 Ctrl+C 退出)...")

try:
    collected_frames = []
    pre_roll_buffer = []  # 🌟 核心：预录环形缓冲区，挽救被提前或滞后触发的字头
    is_speaking = False
    silence_counter = 0

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # 矩阵转换与归一化
        numpy_data = np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS)
        mono_data = numpy_data[:, 0]
        num_samples = int(len(mono_data) * 16000 / RATE)
        resampled_data = signal.resample(mono_data, num_samples)
        float32_data = (resampled_data / 32768.0).astype(np.float32)

        # 🌟 只要程序在跑，内存里永远滚动保留最近 18 个块（约 400ms）的历史音频
        pre_roll_buffer.append(float32_data)
        if len(pre_roll_buffer) > 18: 
            pre_roll_buffer.pop(0)

        # 实时快速扫描单切片的音量能量
        rms = np.sqrt(np.mean(float32_data**2))
        
        # 🌟 调高起点阈值到 0.030，过滤普通的风扇和环境细微噪声
        if rms > 0.030:
            if not is_speaking:
                is_speaking = True
                print("🎙️  [捕捉到人声起点...]")
                # 🌟 关键：一旦起点亮起，不管是不是误触发，直接把此前历史 400ms 的声音全灌进去
                collected_frames.extend(pre_roll_buffer)
            
            collected_frames.append(float32_data)
            silence_counter = 0
        else:
            if is_speaking:
                collected_frames.append(float32_data)
                silence_counter += 1
                
                # 🌟 拉长防抖到 25（约 550ms），给你在“想词、换气、吞口水”时留足安全时间
                if silence_counter > 25:
                    print("🛑 [判定说话结束，正在离线瞬间解码...]")
                    full_audio = np.concatenate(collected_frames)
                    
                    res = model.generate(input=full_audio, cache={}, language="zh", use_itn=True)
                    
                    if res and len(res) > 0 and 'text' in res[0]:
                        raw_text = res[0]['text'].strip()
                        
                        # 清洗标签，精准提取纯汉字部分
                        clean_text = raw_text.split(">")[-1].strip() if ">" in raw_text else raw_text.strip()
                        clean_text = clean_text.replace(" ", "").replace(".", "").replace("?", "").replace("。", "")
                        
                        if clean_text and len(clean_text) > 1:
                            print(f"\n🗣️  小宝精准听到: {clean_text}\n")
                    
                    # 瞬间重置状态机
                    collected_frames = []
                    pre_roll_buffer = [] # 清空预录，重新滚动
                    is_speaking = False
                    silence_counter = 0

except KeyboardInterrupt:
    print("\n🛑 停止流式监听。")
finally:
    stream.stop_stream(); stream.close(); p.terminate()