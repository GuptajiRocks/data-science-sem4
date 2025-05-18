import pyaudio
pa = pyaudio.PyAudio()
print(pa.get_device_count())