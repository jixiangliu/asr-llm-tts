import os
import numpy as np
import scipy.signal as signal
import pyaudio
import re
import requests
import subprocess
import asyncio  # 🛠️ 引入异步库处理流式音频
import edge_tts
from funasr import AutoModel

# =====================================================================
# ⚙️ 1. 全局配置中心
# =====================================================================
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"
SYSTEM_PROMPT = (
    "你是硅基公园的小宝，一个充满科技感、幽默且热心的人形智能机器人。"
    "你精通中文和英文。当用户用英文提问时，你必须用英文回答。"
    "由于是语音交互，你的回答必须口语化、简短，严格控制在 2 句话以内（30字左右），不输出特殊符号。"
)
TTS_VOICE = "zh-CN-YunxiNeural"

# =====================================================================
# 🧠 2. 核心组件重构 (流式边缘发声)
# =====================================================================

def ask_local_llm(user_text):
    """同步阻塞调用本地 Ollama Qwen2.5-7B 大脑"""
    print("🤖 小宝脑回路高速运转中...")
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ],
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['message']['content'].strip()
    except Exception as e:
        print(f"❌ LLM 链路故障: {e}")
    return "小宝刚刚走神了，没听清你说了什么。"


async def amain_stream_tts(text):
    """🛠️ 核心优化：利用 edge-tts 的 Communicate 异步流，通过管道直接喂给 mpv 播放"""
    # 启动 mpv 进程，配置为接收标准输入(stdin)的流式 MP3，并优化音频缓存以降低延迟
    mpv_proc = await asyncio.create_subprocess_exec(
        'mpv', '-', '--cache=no', '--no-video',
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    try:
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        # 边下载字节块，边往 mpv 的 stdin 里塞，实现“边下边播”
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mpv_proc.stdin.write(chunk["data"])
                await mpv_proc.stdin.drain()
    except Exception as e:
        print(f"❌ 流式 TTS 异常: {e}")
    finally:
        # 传输完毕，关闭管道并等待播放器结束
        if mpv_proc.stdin:
            mpv_proc.stdin.close()
        await mpv_proc.wait()


def play_tts(text):
    """同步包裹器：将大模型的文本丢进异步流式循环中，保持外层同步状态机不崩"""
    print(f"🔊 小宝正在流式发声: {text}")
    # 在同步函数里安全调度 asyncio 异步任务，阻塞直到小宝把话说完
    asyncio.run(amain_stream_tts(text))

# =====================================================================
# 🚀 3. ASR 状态机核心引擎
# =====================================================================
print("🚀 正在加载本地 SenseVoice 生产级状态机引擎...")
model = AutoModel(
    model="models/SenseVoiceSmall",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    ncpu=4
)

TAG_CLEANER = re.compile(r'<[^>]+>')

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("\n🎤 【小宝听着呢】极速流式闭环点火成功！请正常说话 (按 Ctrl+C 退出)...")

try:
    collected_frames = []
    is_speaking = False
    silence_counter = 0

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        numpy_data = np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS)
        mono_data = numpy_data[:, 0]
        num_samples = int(len(mono_data) * 16000 / RATE)
        resampled_data = signal.resample(mono_data, num_samples)
        float32_data = (resampled_data / 32768.0).astype(np.float32)

        rms = np.sqrt(np.mean(float32_data**2))
        
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
                
                if silence_counter > 18:
                    print("🛑 [判定说话结束，正在瞬间解码...]")
                    full_audio = np.concatenate(collected_frames)
                    
                    res = model.generate(input=full_audio, cache={}, language="zh", use_itn=True)
                    
                    if res and len(res) > 0 and 'text' in res[0]:
                        raw_text = res[0]['text'].strip()
                        clean_text = TAG_CLEANER.sub('', raw_text).strip()
                        clean_text = clean_text.replace(" ", "").replace(".", "")
                        
                        if clean_text and len(clean_text) > 1:
                            print(f"\n🗣️  小宝听到: {clean_text}\n")
                            
                            stream.stop_stream()  # 闭眼听，防止回音
                            
                            # 1. 投喂给本地大模型
                            llm_reply = ask_local_llm(clean_text)
                            print(f"🤖 大模型回复: {llm_reply}")
                            
                            # 2. ⚡ 极速流式发声
                            play_tts(llm_reply)
                            
                            print("\n🎤 【小宝听着呢】恢复监听...")
                            stream.start_stream()
                    
                    collected_frames = []
                    is_speaking = False
                    silence_counter = 0

except KeyboardInterrupt:
    print("\n🛑 停止流式监听。")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()