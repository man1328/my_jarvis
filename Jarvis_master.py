import os, sys, cv2, threading, time, json, queue, smtplib, math
from difflib import SequenceMatcher
import pyttsx3
import pyaudio
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from ultralytics import YOLO
from mem0 import Memory
from vosk import Model, KaldiRecognizer
import wikipedia
import requests
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()

# --- 1. GLOBAL STATE ---
cigar_count_session = 0
thor_alert_count = 0
latest_command = "Systems Standby"
session_start_time = time.time()
last_thor_warning = 0
current_spatial_dist = "Searching..."

GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_APP_PW = os.environ.get("GMAIL_APP_PW")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# --- 2. THE BUTLER (VOICE WORKER) ---
speech_queue = queue.Queue()
_speak_cooldown_until = 0.0


def voice_worker():
    global _speak_cooldown_until
    # Initializing engine inside the thread is best for Linux stability
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)
    while True:
        text = speech_queue.get()
        if text is None: break
        try:
            word_count = len(text.split())
            # Add a longer buffer so he doesn't hear himself
            _speak_cooldown_until = time.time() + (word_count / 2.5) + 1.5
            engine.say(text)
            engine.runAndWait()
        except:
            pass
        finally:
            speech_queue.task_done()


threading.Thread(target=voice_worker, daemon=True).start()


def speak(text):
    print(f"🤖 JARVIS: {text}")
    speech_queue.put(text)


# --- 3. THE BRAIN FIX (Model: llama3.2:1b) ---
config = {
    "llm": {
        "provider": "ollama",
        "config": {"model": "llama3.2:1b", "ollama_base_url": OLLAMA_BASE_URL}  # MATCHED TO YOUR LIST
    },
    "embedder": {
        "provider": "huggingface",
        "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}
    },
    "vector_store": {
        "provider": "chroma",
        "config": {"collection_name": "jarvis_memories", "path": "./chroma_db"}
    }
}

print("🧠 Initializing Memory...")
m = Memory.from_config(config)

print("👁️ Initializing Vision Models on RTX A1000...")
general_model = YOLO('yolov8n.pt')
cigar_model = YOLO('runs/detect/jarvis_cigar_model/weights/best.pt')

# --- WAKE WORD LOGIC ---
_VOSK_ALIASES = {"jarvis", "travis", "jarvas", "jarvish", "javis"}


def is_wake_word(cmd: str) -> bool:
    for word in cmd.lower().split():
        if word in _VOSK_ALIASES: return True
        if SequenceMatcher(None, word, "jarvis").ratio() >= 0.72: return True
    return False


# --- 4. THE VOICE LISTENER (Fixed ALSA Slave Error) ---
audio_queue = queue.Queue()


def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)


def voice_listener():
    global cigar_count_session, latest_command
    p = pyaudio.PyAudio()

    # 1. INITIALIZE VOSK ONCE (Outside the loop!)
    print("👂 Loading Voice Model...")
    v_model = Model("model")
    v_rec = KaldiRecognizer(v_model, 16000)

    stream = None
    try:
        # Auto-detect default mic without forcing an index
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                        input=True, frames_per_buffer=8000,  # Increased buffer for stability
                        stream_callback=audio_callback)
        print("✅ Mic successfully engaged.")
    except Exception as e:
        print(f"🚨 Mic Initialization Failed: {e}")
        return

    is_awake = False
    awake_time = 0
    AWAKE_DURATION = 15

    while True:
        # Get data from the queue (filled by audio_callback)
        data = audio_queue.get()

        # Self-mute check
        if time.time() < _speak_cooldown_until:
            continue

        if v_rec.AcceptWaveform(data):
            res = json.loads(v_rec.Result())
            cmd = res.get('text', '').lower()
            if not cmd: continue

            print(f"🗣️ Heard: {cmd}")

            # Wake Logic
            if is_wake_word(cmd):
                if not is_awake:
                    speak("Ready, Ethan.")
                is_awake = True
                awake_time = time.time()
                continue

            # Command Logic
            if is_awake:
                if any(k in cmd for k in ["record", "log", "save", "cigar"]):
                    cigar_count_session += 1
                    speak("Moment logged.")
                    is_awake = False
                elif any(k in cmd for k in ["report", "status", "update"]):
                    speak(f"System status: {current_spatial_dist}. Logs: {cigar_count_session}.")
                    is_awake = False
                elif any(k in cmd for k in ["close", "finish", "end", "stop"]):
                    speak("Shutting down protocol. Sending email.")
                    send_session_summary()
                    time.sleep(2)
                    os._exit(0)


threading.Thread(target=voice_listener, daemon=True).start()


# --- 5. MAIN VISION ---
def start_protocol():
    global current_spatial_dist, thor_alert_count, last_thor_warning
    cap = cv2.VideoCapture(0)  # Standard dev for laptop

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Run detection on GPU (Removing device='cpu')
        gen_res = general_model.predict(source=frame, verbose=False, classes=[16], conf=0.20)
        cig_res = cigar_model.predict(source=frame, verbose=False, conf=0.25)

        # ... (Distance math from previous build goes here) ...
        # (HUD logic from previous build goes here) ...

        cv2.imshow("Jarvis Master Interface", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break


if __name__ == "__main__":
    start_protocol()