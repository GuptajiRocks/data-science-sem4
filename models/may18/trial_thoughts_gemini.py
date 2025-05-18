import pyaudio
import wave
import os
import time
import threading
import queue
import speech_recognition as sr
from collections import deque
from google import genai
from google.genai import types

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
RECORD_SECONDS = 5
OVERLAP_SECONDS = 1
CHANNELS = 1
FORMAT = pyaudio.paInt16
PHRASE_TIMEOUT = 2.0
MAX_PHRASE_LENGTH = 30

audio_queue = queue.Queue()
stop_flag = threading.Event()
audio_buffer = []
MAX_BUFFER_SIZE = int(SAMPLE_RATE / CHUNK_SIZE * (RECORD_SECONDS + OVERLAP_SECONDS))

class PhraseBuffer:
    def __init__(self):
        self.transcript = []
        self.last_update = time.time()
        self.lock = threading.Lock()
        self.new_words = deque(maxlen=MAX_PHRASE_LENGTH)
        self.full_transcript = []
        self.thoughts_history = []
        
    def append(self, text):
        with self.lock:
            self.transcript.append(text)
            self.full_transcript.append(text)
            self.new_words.extend(text.split())
            self.last_update = time.time()
    
    def add_thought(self, thought):
        with self.lock:
            self.thoughts_history.append(thought)
    
    def get_last_thought(self):
        with self.lock:
            if self.thoughts_history:
                return self.thoughts_history[-1]
            return None
            
    def get_phrase(self):
        with self.lock:
            if self.time_since_update > PHRASE_TIMEOUT and self.new_words:
                phrase = ' '.join(self.new_words)
                self.new_words.clear()
                return phrase
            return None
            
    def get_full_transcript(self):
        with self.lock:
            return ' '.join(self.full_transcript)
            
    @property
    def time_since_update(self):
        return time.time() - self.last_update

def generate_thought(phrase, phrase_buffer):
    try:
        client = genai.Client(api_key="")
        last_thought = phrase_buffer.get_last_thought()
        if last_thought:
            prompt = f"Based on my previous thought '{last_thought}', generate a brief, insightful thought in (1 sentence) about this new phrase: '{phrase}'"
        else:
            prompt = f"Generate a brief, insightful thought (1 sentence) about this phrase: '{phrase}'"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=100
            )
        )
        thought = response.text.strip()
        phrase_buffer.add_thought(thought)
        return thought
    except Exception as e:
        print(f"\033[91mError generating thought: {e}\033[0m")
        return f"Thinking about: {phrase}"

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)
    
    print("* Recording started. Speak into the microphone.")
    
    global audio_buffer
    chunks_per_record = int(SAMPLE_RATE / CHUNK_SIZE * RECORD_SECONDS)
    overlap_chunks = int(SAMPLE_RATE / CHUNK_SIZE * OVERLAP_SECONDS)
    
    while not stop_flag.is_set():
        try:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_buffer.append(data)
            if len(audio_buffer) > MAX_BUFFER_SIZE:
                audio_buffer.pop(0)
            if len(audio_buffer) >= chunks_per_record:
                temp_file = f"temp_chunk_{time.time()}.wav"
                with wave.open(temp_file, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(b''.join(audio_buffer[-chunks_per_record:]))
                audio_queue.put(temp_file)
                if len(audio_buffer) > overlap_chunks:
                    audio_buffer = audio_buffer[-overlap_chunks:]
        except Exception as e:
            print(f"Error recording: {e}")
            continue
    print("* Recording stopped")
    stream.stop_stream()
    stream.close()
    p.terminate()

def transcribe_audio(phrase_buffer):
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    print("* Speech recognition initialized. Ready to transcribe.")
    while not stop_flag.is_set() or not audio_queue.empty():
        try:
            audio_file = audio_queue.get(timeout=1)
            if os.path.exists(audio_file) and os.path.getsize(audio_file) > 1000:
                with sr.AudioFile(audio_file) as source:
                    audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    if text:
                        phrase_buffer.append(text)
                        print(f"\033[94mTranscribed: {text}\033[0m")
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
            try:
                os.remove(audio_file)
            except Exception as e:
                print(f"Error removing temporary file: {e}")
        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in transcription: {e}")

def thought_generator(phrase_buffer):
    print("* Thought generator initialized. Waiting for complete phrases...")
    while not stop_flag.is_set():
        phrase = phrase_buffer.get_phrase()
        if phrase:
            thought = generate_thought(phrase, phrase_buffer)
            print(f"\033[92mThought: {thought}\033[0m")
        time.sleep(0.1)

def main():
    phrase_buffer = PhraseBuffer()
    record_thread = threading.Thread(target=record_audio)
    record_thread.start()
    transcribe_thread = threading.Thread(target=transcribe_audio, args=(phrase_buffer,))
    transcribe_thread.start()
    thought_thread = threading.Thread(target=thought_generator, args=(phrase_buffer,))
    thought_thread.start()
    try:
        print("* Live transcription and thought generation started. Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n* Stopping all processes...")
        stop_flag.set()
        record_thread.join()
        transcribe_thread.join()
        thought_thread.join()
        full_transcript = phrase_buffer.get_full_transcript()
        if full_transcript:
            with open("transcript.txt", "w", encoding="utf-8") as f:
                f.write(full_transcript)
            print("* Transcript saved to 'transcript.txt'")
        thoughts = phrase_buffer.thoughts_history
        if thoughts:
            with open("thoughts.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(thoughts))
            print("* Thoughts saved to 'thoughts.txt'")
        print("* Done!")

if __name__ == "__main__":
    main()
