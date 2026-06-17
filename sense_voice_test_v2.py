import os
import numpy as np
import pyaudio
import re
from funasr import AutoModel

print("🚀 [GPU 加速启用] 正在加载本地 SenseVoice 生产级硬加速引擎...")
model = AutoModel(
    model="models/SenseVoiceSmall",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    device="cuda",          # 🌟 强行挂载在 Orin 的英伟达 GPU 显存中
    ncpu=4,
    disable_update=True     # 闭环纯离线
)

TAG_CLEANER = re.compile(r'<\|.*?\|>')

# 🌟 像素级适配 Ubuntu Pulse 混音层参数
DEVICE_INDEX = 28      # 👈 锁定你刚才查出来的 default 设备索引
CHANNELS = 32          # 👈 必须填入系统底层声明的 32，强行破局 -9998 报错！
RATE = 16000           # 保持模型最需要的 16k 采样率
CHUNK = 1024           # 稳定的多通道硬缓冲区

p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX, # 强行锁定默认音频枢纽
    frames_per_buffer=CHUNK
)

print("\n🎤 【Pulse 通道对齐】小宝离线听觉完全体已点火！请正常说话 (按 Ctrl+C 退出)...")

try:
    collected_frames = []
    pre_roll_buffer = []  # 预录环形缓冲区，挽救吞字头
    is_speaking = False
    silence_counter = 0

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # 🌟 硬解密：将 32 通道的一维流，精准重塑为标准的二维矩阵
        numpy_data = np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS)
        
        # 🌟 提纯：Pulse 混音层默认会把讯飞模块的第一输入挂在第 0 声道，直接提取！
        mono_data = numpy_data[:, 0] 
        
        # 无损转化为 Float32 投喂给大模型
        float32_data = (mono_data / 32768.0).astype(np.float32)

        # 维护最近的前置预录缓存
        pre_roll_buffer.append(float32_data)
        if len(pre_roll_buffer) > 6: 
            pre_roll_buffer.pop(0)

        # 实时快速扫描音量能量
        rms = np.sqrt(np.mean(float32_data**2))
        
        if rms > 0.020:
            if not is_speaking:
                is_speaking = True
                print("🎙️  [捕捉到人声起点...]")
                collected_frames.extend(pre_roll_buffer) # 瞬间捞回错过的字头
            
            collected_frames.append(float32_data)
            silence_counter = 0
        else:
            if is_speaking:
                collected_frames.append(float32_data)
                silence_counter += 1
                
                # 连续大约 450ms 静音判定整句话说完
                if silence_counter > 14:
                    print("🛑 [判定说话结束，GPU 瞬间秒解...]")
                    full_audio = np.concatenate(collected_frames)
                    
                    res = model.generate(input=full_audio, cache={}, language="zh", use_itn=True)
                    
                    if res and len(res) > 0 and 'text' in res[0]:
                        raw_text = res[0]['text'].strip()
                        
                        # 清洗所有富文本标签，提取纯汉字
                        clean_text = raw_text.split(">")[-1].strip() if ">" in raw_text else raw_text.strip()
                        clean_text = clean_text.replace(" ", "").replace(".", "").replace("?", "").replace("。", "")
                        
                        if clean_text and len(clean_text) > 1:
                            print(f"\n🗣️  小宝精准听到: {clean_text}\n")
                    
                    # 瞬间重置状态机
                    collected_frames = []
                    is_speaking = False
                    silence_counter = 0

except KeyboardInterrupt:
    print("\n🛑 停止流式监听。")
finally:
    stream.stop_stream(); stream.close(); p.terminate()