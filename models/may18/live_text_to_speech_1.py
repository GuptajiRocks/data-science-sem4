import speech_recognition as sr

recognizer = sr.Recognizer()

# Replace `1` with the index of your desired microphone
mic_index = 2

with sr.Microphone(device_index=mic_index) as source:
    print("🎤 Using custom microphone. Speak something...")
    recognizer.adjust_for_ambient_noise(source)
    audio = recognizer.listen(source)

    try:
        print("🔍 Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"✅ You said: {text}")
    except sr.UnknownValueError:
        print("❌ Could not understand the audio.")
    except sr.RequestError as e:
        print(f"⚠️ Could not request results; {e}")
