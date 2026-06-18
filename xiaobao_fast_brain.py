import os
import numpy as np
import scipy.signal as signal
import pyaudio
import re
import json
import requests
import subprocess
import asyncio
import edge_tts
from funasr import AutoModel

# =====================================================================
# ⚙️ 1. 全局配置中心
# =====================================================================
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"
SYSTEM_PROMPT = (
    "你是小宝，一个充满科技感、幽默且热心的人形智能机器人。"
    "你精通中文和英文。回答要口语化、生动。每次回答控制在2-3句话内，不要太长。"
)
TTS_VOICE = "zh-CN-YunxiNeural"

# =====================================================================
# 🧠 2. ⚡ 极速双流式联动组件 (LLM Stream + TTS Stream)
# =====================================================================

async def amain_stream_tts(text):
    """底层极速 TTS 播放流"""
    mpv_proc = await asyncio.create_subprocess_exec(
        'mpv', '-', '--cache=no', '--no-video',
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mpv_proc.stdin.write(chunk["data"])
                await mpv_proc.stdin.drain()
    except Exception as e:
        print(f"❌ 流式 TTS 异常: {e}")
    finally:
        if mpv_proc.stdin:
            mpv_proc.stdin.close()
        await mpv_proc.wait()


async def pipeline_llm_and_tts(user_text):
    """🛠️ 核心加速器：LLM一边吐字，TTS一边说话"""
    print("🤖 小宝脑回路高速运转中...")
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ],
        "stream": True  # ⚡ 开启大模型流式输出
    }
    
    # 句号、问号、感叹号、省略号、换行符，作为切句标志
    sentence_end_pattern = re.compile(r'([。？！；…?…\n])')
    
    try:
        # 使用 requests 的流式接收
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=30)
        if response.status_code != 200:
            print(f"❌ LLM 服务响应异常: {response.status_code}")
            return

        buffer = ""
        print("🤖 大模型回复: ", end="", flush=True)

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                token = chunk['message']['content']
                print(token, end="", flush=True) # 终端实时打字机
                
                buffer += token
                
                # 检查缓冲区里有没有一句完整的话了
                matches = list(sentence_end_pattern.finditer(buffer))
                if matches:
                    # 取出第一句完整的话（到最后一个标点符号为止）
                    end_pos = matches[-1].end()
                    sentence_to_speak = buffer[:end_pos].strip()
                    buffer = buffer[end_pos:] # 留下的残渣作为下一句的开头
                    
                    if len(sentence_to_speak) > 1:
                        # 扔给 TTS 播放，并阻塞等待这一句放完再切下一句，防止声音重叠
                        await amain_stream_tts(sentence_to_speak)
                        
        # 大模型全部吐完了，如果缓冲区还剩最后一点没标点符号的尾巴，补放出来
        if buffer.strip():
            await amain_stream_tts(buffer.strip())
        print() # 换行

    except Exception as e:
        print(f"\n❌ 双流式链路故障: {e}")


def play_combined_stream(text):
    """同步包裹器：对接原本的 ASR 同步主循环"""
    asyncio.run(pipeline_llm_and_tts(text))

# =====================================================================
# 🚀 3. ASR 状态机核心引擎
# =====================================================================
print("🚀 正在加载本地 SenseVoice 生产级状态机引擎...")
model = AutoModel(model="models/SenseVoiceSmall", vad_model="fsmn-vad", punc_model="ct-punc", ncpu=4)
TAG_CLEANER = re.compile(r'<[^>]+>')

CHUNK = 1024; FORMAT = pyaudio.paInt16; CHANNELS = 2; RATE = 48000
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("\n🎤 【小宝听着呢】双流式高并发闭环点火成功！请正常说话...")

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
                        clean_text = TAG_CLEANER.sub('', raw_text).strip().replace(" ", "").replace(".", "")
                        
                        if clean_text and len(clean_text) > 1:
                            print(f"\n🗣️  小宝听到: {clean_text}\n")
                            
                            stream.stop_stream()  # 闭眼
                            
                            # ⚡ 执行终极双流式联动
                            play_combined_stream(clean_text)
                            
                            print("\n🎤 【小宝听着呢】恢复监听...")
                            stream.start_stream() # 开眼
                    
                    collected_frames = []
                    is_speaking = False
                    silence_counter = 0

except KeyboardInterrupt:
    print("\n🛑 停止流式监听。")
finally:
    stream.stop_stream(); stream.close(); p.terminate()
