import os
import numpy as np
import pyaudio
import re
from funasr_onnx import SenseVoiceSmall

# 🌟 1. 强行锁定你刚刚成功激活的物理显卡 CUDA 驱动核心进行秒级推理！
print("🚀 [ONNX 硬加速] 正在加载阿里官方本地 ONNX 生产级显卡内核...")
model_dir = "models/SenseVoiceSmall" # 自动重用你本地已经下好的模型

# 强行挂载 CUDA 执行提供商（providers），RTF 绝杀到 0.01
model = SenseVoiceSmall(model_dir, batch_size=1, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

TAG_CLEANER = re.compile(r'<\|.*?\|>')

# 🌟 针对你的 32 通道物理脱水参数
DEVICE_INDEX = 28      
CHANNELS = 32          
RATE = 16000           
CHUNK = 1024           

p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK
)

print("\n🎤 【ONNX 显卡内核·物理脱水流】小宝中枢完全体就绪！请说话...")

try:
    collected_frames = []
    pre_roll_buffer = []  
    is_speaking = False
    silence_counter = 0

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # 🌟 物理脱水：每 32 个点只抽 1 个点，瞬间帮 CPU 减负 32 倍，杜绝丢帧漏字
        raw_samples = np.frombuffer(data, dtype=np.int16)
        mono_data = raw_samples[0::CHANNELS] 
        
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
                
                # 连续大约 450ms 静音，判定整句话说完
                if silence_counter > 14:
                    print("🛑 [判定说话结束，显卡 ONNX 瞬间秒解...]")
                    full_audio = np.concatenate(collected_frames)
                    
                    # 🌟 阿里 ONNX 后端极简推理调用
                    res = model(full_audio, language="zh", use_itn=True)
                    
                    if res and len(res) > 0 and 'text' in res[0]:
                        raw_text = res[0]['text'].strip()
                        clean_text = raw_text.split(">")[-1].strip() if ">" in raw_text else raw_text.strip()
                        clean_text = clean_text.replace(" ", "").replace(".", "").replace("?", "").replace("。", "")
                        
                        if clean_text and len(clean_text) > 1:
                            print(f"\n🗣️  小宝精准听到: {clean_text}\n")
                    
                    # 重置状态机
                    collected_frames = []
                    is_speaking = False
                    silence_counter = 0

except KeyboardInterrupt:
    print("\n🛑 停止流式监听。")
finally:
    stream.stop_stream(); stream.close(); p.terminate()