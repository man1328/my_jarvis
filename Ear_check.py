import pyaudio

p = pyaudio.PyAudio()
print("--- 🎤 AUDIO DEVICES DETECTED ---")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    print(f"ID: {i} | Name: {dev['name']} | Channels: {dev['maxInputChannels']}")
p.terminate()