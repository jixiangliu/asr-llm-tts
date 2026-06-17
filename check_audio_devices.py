import pyaudio

p = pyaudio.PyAudio()
print("\n🔍 正在扫描机器人物理音频输入设备...\n")

for i in range(p.get_device_count()):
    dev_info = p.get_device_info_by_index(i)
    if dev_info['maxInputChannels'] > 0:
        print(f"🌟 【设备索引 ID: {i}】")
        print(f"    设备名称: {dev_info['name']}")
        print(f"    硬件支持最大输入声道数: {dev_info['maxInputChannels']}")
        print(f"    默认采样率: {dev_info['defaultSampleRate']}Hz\n")
p.terminate()
