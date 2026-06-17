import os
import numpy as np
import pyaudio
import re

print("🚀 [纯净保底模式] 正在以 CPU 绝对安全兼容模式加载本地 SenseVoice 引擎...")
from funasr import AutoModel

model = AutoModel(
    model="models/SenseVoiceSmall",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    device="cpu",          # 🌟 降维打击：强行走 CPU 运行，彻底无视显卡版本冲突！
    ncpu=4,
    disable_update=True     
)

TAG_CLEANER = re.compile(r'<\|.*?\|>')

# 🌟 全自动默认音频兼容参数
CHANNELS = 1           # 👈 绝大多数 Linux 默认输入通道，彻底碾碎 -9998 报错！
RATE = 16000           
CHUNK = 1024           

p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    # 🌟 不指定固定设备索引，让系统自动挑当前生效的默认麦克风
    frames_per_buffer=CHUNK
)

print("\n🎤 【安全保底通道】完全体已点火！请对着麦克风正常说话 (按 Ctrl+C 退出)...")

try:
    collected_frames = []
    pre_roll_buffer = []  
    is_speaking = False
    silence_counter = 0

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # 1声道标准处理
        mono_data = np.frombuffer(data, dtype=np.int16)
        float32_data = (mono_data / 32768.0).astype(np.float32)

        pre_roll_buffer.append(float32_data)
        if len(pre_roll_buffer) > 6: 
            pre_roll_buffer.pop(0)

        rms = np.sqrt(np.mean(float32_data**2))
        
        if rms > 0.020:
            if not is_speaking:
                is_speaking = True
                print("🎙️  [捕捉到人声起点...]")
                collected_frames.extend(pre_roll_buffer) 
            
            collected_frames.append(float32_data)
            silence_counter = 0
        else:
            if is_speaking:
                collected_frames.append(float32_data)
                silence_counter += 1
                
                if silence_counter > 14:
                    print("🛑 [判定说话结束，CPU 正在解码...]")
                    full_audio = np.concatenate(collected_frames)
                    
                    res = model.generate(input=full_audio, cache={}, language="zh", use_itn=True)
                    
                    if res and len(res) > 0 and 'text' in res[0]:
                        raw_text = res[0]['text'].strip()
                        clean_text = raw_text.split(">")[-1].strip() if ">" in raw_text else raw_text.strip()
                        clean_text = clean_text.replace(" ", "").replace(".", "").replace("?", "").replace("。", "")
                        
                        if clean_text and len(clean_text) > 1:
                            print(f"\n🗣️  小宝精准听到: {clean_text}\n")
                    
                    collected_frames = []
                    is_speaking = False
                    silence_counter = 0

except KeyboardInterrupt:
    print("\n🛑 停止流式监听。")
finally:
    stream.stop_stream(); stream.close(); p.terminate()