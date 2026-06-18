import os
import numpy as np
import scipy.signal as signal
import pyaudio
import re
import requests
import subprocess
from funasr import AutoModel

# =====================================================================
# ⚙️ 1. 全局配置中心 (针对 Jetson AGX Orin 32GB 优化)
# =====================================================================
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"  # 已成功拉取的 7B 满血版中英双语大模型

# 限制机器人的回答风格：高情商、短小精悍、中英双语地道切换
SYSTEM_PROMPT = (
    "你是小宝，一个充满科技感、幽默且热心的人形智能机器人。"
    "你精通中文和英文。当用户用英文提问时，你必须用英文回答。"
    "由于是语音交互，你的回答必须口语化、简短，严格控制在 2 句话以内（30字左右），不输出特殊符号。"
)

# TTS 发音人选择：
# zh-CN-YunxiNeural (地道中英双语男声，推荐) 
# zh-CN-XiaoxiaoNeural (甜美中英双语女声)
TTS_VOICE = "zh-CN-YunxiNeural" 

# =====================================================================
# 🧠 2. 核心联动组件 (LLM + TTS)
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
        # 🛠️ 首次加载模型需要时间，将 timeout 放大到 30 秒
        response = requests.post(OLLAMA_URL, json=payload, timeout=30) 
        if response.status_code == 200:
            return response.json()['message']['content'].strip()
    except Exception as e:
        print(f"❌ LLM 链路故障: {e}")
    return "小宝刚刚走神了，没听清你说了什么。"


def play_tts(text):
    """请求高质量中英双语语音，先生成mp3，转为wav后使用 aplay 阻塞播放"""
    print(f"🔊 小宝准备发声: {text}")
    mp3_file = "tts_output.mp3"
    wav_file = "tts_output.wav"
    
    try:
        # 1. 🛠️ 核心修正：把 --write-to-media 改为 --write-media
        cmd_tts = ["edge-tts", "--voice", TTS_VOICE, "--text", text, "--write-media", mp3_file]
        subprocess.run(cmd_tts, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 2. 转换为 wav
        cmd_convert = ["ffmpeg", "-y", "-i", mp3_file, wav_file]
        subprocess.run(cmd_convert, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 3. 硬件级阻塞播放
        os.system(f"aplay {wav_file} > /dev/null 2>&1")
        
    except Exception as e:
        print(f"❌ TTS 语音合成或播放失败: {e}")
    finally:
        # 4. 清理临时产生的音频碎片
        if os.path.exists(mp3_file):
            os.remove(mp3_file)
        if os.path.exists(wav_file):
            os.remove(wav_file)

# =====================================================================
# 🚀 3. ASR 状态机核心引擎 (基于你的工业级双指针架构)
# =====================================================================
print("🚀 正在加载本地 SenseVoice 生产级状态机引擎...")
model = AutoModel(
    model="models/SenseVoiceSmall",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    ncpu=4
)

TAG_CLEANER = re.compile(r'<[^>]+>')

CHUNK = 1024          # 规整化数据块
FORMAT = pyaudio.paInt16
CHANNELS = 2          # 完美兼容你的环形麦克风阵列
RATE = 48000          # 48k 采样

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("\n🎤 【小宝听着呢】全本地语音闭环点火成功！请正常说话 (按 Ctrl+C 退出)...")

try:
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
        
        # 触发阈值
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
                
                # 当连续大约 18 个块（约 400~500毫秒）都是静音时，判定说话结束
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
                        clean_text = clean_text.replace(" ", "").replace(".", "")
                        
                        if clean_text and len(clean_text) > 1:
                            print(f"\n🗣️  小宝听到: {clean_text}\n")
                            
                            # =========================================================
                            # 🛠️ 核心串联：闭环流控
                            # =========================================================
                            # 1. 暂停录音流（闭上耳朵，切断输入，防止喇叭里的声音自激啸叫）
                            stream.stop_stream()
                            
                            # 2. 投喂给本地 Ollama
                            llm_reply = ask_local_llm(clean_text)
                            print(f"🤖 大模型回复: {llm_reply}")
                            
                            # 3. 投喂给 TTS 语音合成并硬件播放 (同步阻塞)
                            play_tts(llm_reply)
                            
                            # 4. 彻底说完了，重新开启录音流，洗耳恭听下一次唤醒
                            print("\n🎤 【小宝听着呢】恢复监听...")
                            stream.start_stream()
                            # =========================================================
                    
                    # 瞬间重置状态机，释放硬件显存
                    collected_frames = []
                    is_speaking = False
                    silence_counter = 0

except KeyboardInterrupt:
    print("\n🛑 停止流式监听。")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
