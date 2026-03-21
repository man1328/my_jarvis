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
import re

# Load environment variables
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

# --- MODEL ROSTER ---
# All model names are env-var overridable — change in .env without touching code.
# Pull all with: ollama pull <model>
MODELS = {
    # Always-warm router/quick-reply (tiny, stays in VRAM)
    "router":   os.environ.get("MODEL_ROUTER",   "llama3.2:1b"),
    # General assistant — research, recipes, conversation
    "general":  os.environ.get("MODEL_GENERAL",  "llama3.1:8b"),
    # Uncensored reasoning — complex Q&A, no refusals
    "chat":     os.environ.get("MODEL_CHAT",     "dolphin-mistral:7b"),
    # Code: fast path — simple scripts, snippets
    "code_fast": os.environ.get("MODEL_CODE_FAST", "qwen2.5-coder:7b"),
    # Code: deep path — full apps, complex architecture, multi-file
    "code_deep": os.environ.get("MODEL_CODE_DEEP", "deepseek-coder:6.7b"),
}

# Keywords that signal a COMPLEX coding job → route to the deeper code model
_DEEP_CODE_SIGNALS = [
    "full app", "full application", "entire", "complete", "architecture",
    "multi", "database", "api", "flask", "django", "fastapi", "website",
    "frontend", "backend", "game", "system", "framework", "complex",
]


def route_model(task_type: str) -> str:
    """Return the best model name for a given task type."""
    model = MODELS.get(task_type, MODELS["general"])
    print(f"🧭 Routing task '{task_type}' → {model}")
    return model


def route_code_model(query: str) -> str:
    """Return qwen (fast) for simple scripts, deepseek (deep) for complex jobs."""
    q = query.lower()
    if any(sig in q for sig in _DEEP_CODE_SIGNALS):
        model = MODELS["code_deep"]
        print(f"🧭 Complex code detected → {model}")
    else:
        model = MODELS["code_fast"]
        print(f"🧭 Quick code detected → {model}")
    return model


# --- SHARED STATE (For Streamlit Dashboard) ---
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_state.json")
CMD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_trigger.json")
command_log = []


def write_state():
    global cigar_count_session, thor_alert_count, latest_command, current_spatial_dist, command_log
    state = {
        "status": "online",
        "uptime_mins": round((time.time() - session_start_time) / 60, 1),
        "latest_command": latest_command,
        "cigar_count": cigar_count_session,
        "thor_alerts": thor_alert_count,
        "spatial_dist": current_spatial_dist,
        "command_log": command_log[-20:],
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S")
    }
    try:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        print(f"State write error: {e}")


# --- SMARTER WAKE WORD ---
_VOSK_ALIASES = {"jarvis", "travis", "jarvas", "jarvish"}
_WAKE_SIMILARITY_THRESHOLD = 0.72


def is_wake_word(cmd: str) -> bool:
    for word in cmd.lower().split():
        if word in _VOSK_ALIASES:
            return True
        if SequenceMatcher(None, word, "jarvis").ratio() >= _WAKE_SIMILARITY_THRESHOLD:
            return True
    return False


# --- 2. VOICE WORKER (The Butler Thread) ---
speech_queue = queue.Queue()
_speak_cooldown_until = 0.0


def voice_worker():
    global _speak_cooldown_until
    engine = pyttsx3.init()
    engine.setProperty('rate', 165)
    while True:
        text = speech_queue.get()
        if text is None: break
        try:
            word_count = len(text.split())
            estimated_secs = max(3.0, (word_count / 2.75) + 1.0)
            _speak_cooldown_until = time.time() + estimated_secs
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


# --- 3. AI BRAIN & RTX VISION ---
config = {
    "llm": {
        "provider": "ollama",
        "config": {"model": "llama3.2:1b", "ollama_base_url": OLLAMA_BASE_URL}
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
general_model = YOLO('yolov8n.pt').to('cuda')
cigar_model = YOLO('runs/detect/jarvis_cigar_model/weights/best.pt').to('cuda')

# --- WARM UP OLLAMA ---
print("🔌 Checking AI brain connectivity...")
try:
    _ping = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=5)
    if _ping.status_code == 200:
        print("✅ Ollama server is reachable. Warming up model...")
        try:
            _warmup = requests.post(f'{OLLAMA_BASE_URL}/api/generate',
                                    json={'model': 'llama3.2:1b', 'prompt': 'Hello.', 'stream': False},
                                    timeout=120)
            if _warmup.status_code == 200:
                print("✅ AI brain is warm and ready.")
            else:
                print(f"⚠️  Warmup got status {_warmup.status_code}")
        except Exception as _we:
            print(f"⚠️  Warmup timed out: {_we}")
    else:
        print(f"⚠️  Ollama server returned status {_ping.status_code}.")
        speak("Warning: AI brain returned an unexpected status.")
except Exception as _e:
    print(f"❌ AI brain is UNREACHABLE at {OLLAMA_BASE_URL}: {_e}")
    speak("Warning: AI brain is offline.")

print("👂 Initializing Voice Recognition...")
vosk_model = Model("model")  # Uses relative path for portability
rec = KaldiRecognizer(vosk_model, 16000)
audio_queue = queue.Queue()


# --- 4. ACTION FUNCTIONS ---
def get_dist(b1, b2):
    c1 = [(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2]
    c2 = [(b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2]
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def send_session_summary():
    global cigar_count_session, thor_alert_count
    duration = round((time.time() - session_start_time) / 60, 1)

    msg = MIMEMultipart()
    msg['Subject'] = f"Jarvis Session Report - {datetime.now().strftime('%b %d')}"
    msg['From'] = GMAIL_USER
    msg['To'] = GMAIL_USER

    body = f"Ethan,\n\nSession concluded.\n- Duration: {duration} mins\n- Cigars Logged: {cigar_count_session}\n- Thor Alerts: {thor_alert_count}"
    msg.attach(MIMEText(body, 'plain'))

    if os.path.exists("best_capture.jpg"):
        with open("best_capture.jpg", 'rb') as f:
            msg.attach(MIMEImage(f.read(), name="session_moment.jpg"))

    try:
        if not GMAIL_USER or not GMAIL_APP_PW:
            print("Email credentials not set.")
            speak("Session summary email skipped as credentials are not configured.")
            return

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_USER, GMAIL_APP_PW)
            server.send_message(msg)
        speak("Summary email sent successfully.")
    except Exception as e:
        print(f"Email failed: {e}")


def conduct_research(query):
    speak(f"Researching {query} now.")
    try:
        summary = wikipedia.summary(query, sentences=2)
        speak("Here is what I found. " + summary)

        page = wikipedia.page(query)
        report_filename = f"Report_{query.replace(' ', '_')}.txt"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(f"--- RESEARCH REPORT: {page.title} ---\n\n")
            f.write(page.content)

        speak(f"I have saved a detailed report to {report_filename}.")
        m.add(f"Researched {query}", user_id="ethan")
    except Exception as e:
        speak("I encountered an error while researching.")


def find_recipe(query):
    model = route_model("general")
    speak(f"Consulting the AI chef for {query}.")
    try:
        response = requests.post(f'{OLLAMA_BASE_URL}/api/generate', json={
            'model': model,
            'prompt': f'Give me a clear, step-by-step recipe for {query}. Keep it concise.',
            'stream': False
        }, timeout=120)
        if response.status_code == 200:
            recipe_text = response.json().get('response', '')
            report_filename = f"Recipe_{query.replace(' ', '_')}.txt"
            with open(report_filename, "w", encoding="utf-8") as f:
                f.write(f"--- RECIPE: {query.upper()} ---\n\n")
                f.write(recipe_text)
            speak(f"I have successfully generated the recipe for {query}.")
            m.add(f"Found a recipe for {query}", user_id="ethan")
    except Exception as e:
        speak("I encountered an error searching for the recipe.")


def find_best_prices(query):
    speak(f"Shopping for {query}. Checking the web.")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query + " price", max_results=5))
        if results:
            speak("I found some shopping results and logged them.")
            m.add(f"Shopped for {query}", user_id="ethan")
        else:
            speak("I could not find any shopping results.")
    except Exception as e:
        speak("I encountered an error while shopping online.")


def generate_code(query):
    model = route_code_model(query)
    speak(f"Drafting your code for: {query}. Please wait, Ethan.")
    try:
        print(f"🌐 Contacting {model} for: {query}...")

        system_prompt = "You are an elite software engineer. Return ONLY the code requested inside a single standard markdown code block. Do not include explanations, greetings, or text outside the code block."

        response = requests.post(f'{OLLAMA_BASE_URL}/api/generate', json={
            'model': model,
            'system': system_prompt,
            'prompt': query,
            'stream': False
        }, timeout=180)

        if response.status_code == 200:
            raw_text = response.json().get('response', '')

            # Safely separated backticks to avoid UI parsing errors
            regex_pattern = r'``' + r'`(?:\w+)?\n(.*?)``' + r'`'
            code_matches = re.findall(regex_pattern, raw_text, re.DOTALL)

            if code_matches:
                final_code = code_matches[0].strip()
            else:
                final_code = raw_text.strip()

            ext = ".py"
            if "html" in query.lower() or "website" in query.lower():
                ext = ".html"
            elif "javascript" in query.lower() or " js" in query.lower():
                ext = ".js"
            elif "css" in query.lower():
                ext = ".css"
            elif "bash" in query.lower() or "script" in query.lower() and "python" not in query.lower():
                ext = ".sh"

            filename = f"jarvis_project_{int(time.time())}{ext}"

            with open(filename, "w", encoding="utf-8") as f:
                f.write(final_code)

            speak(f"Coding complete. I have saved the file as {filename} in your project folder.")
            print(f"✅ Code successfully written to {filename}")
            m.add(f"Wrote software code for: {query}", user_id="ethan")
            write_state()
        else:
            speak("I encountered a server issue generating the code.")
    except Exception as e:
        speak("My coding engine timed out or went offline.")
        print(f"❌ Code Gen Error: {e}")


# --- 5. THE VOICE LISTENER ---
def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)


def voice_listener():
    global cigar_count_session, latest_command
    p = pyaudio.PyAudio()
    stream = None

    try:
        dev_info = p.get_default_input_device_info()
        default_idx = int(dev_info['index'])
        for chans in [1, 2]:
            try:
                stream = p.open(format=pyaudio.paInt16, channels=chans, rate=16000,
                                input=True, input_device_index=default_idx,
                                frames_per_buffer=4000, stream_callback=audio_callback)
                print("✅ Mic successfully engaged via auto-detect.")
                break
            except Exception as _mic_e:
                print(f"⚠️  Mic open failed (channels={chans}): {_mic_e}")
                continue
    except Exception as _dev_e:
        print(f"❌ Could not query default input device: {_dev_e}")

    if not stream:
        print("🚨 Mic Failed.")
        return

    is_awake = False
    awake_time = 0
    AWAKE_DURATION = 20

    action_words = ["search for", "research", "look up", "recipe for", "how to cook", "how to bake", "record", "log",
                    "save", "cigar", "report", "status", "update", "close", "finish", "end", "stop", "shop for", "buy ",
                    "find prices for", "write a script", "code a", "build a"]

    while True:
        data = audio_queue.get()
        if time.time() < _speak_cooldown_until: continue

        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            cmd = res.get('text', '').lower()
            if not cmd: continue

            print(f"🗣️ Heard: {cmd}")

            if is_awake and (time.time() - awake_time > AWAKE_DURATION):
                is_awake = False
                speak("Going back to sleep.")

            if is_wake_word(cmd):
                if not is_awake: speak("Ready, Ethan.")
                is_awake = True
                awake_time = time.time()
                continue

            if is_awake and any(a in cmd for a in action_words):
                latest_command = cmd
                command_log.append(f"[{time.strftime('%H:%M:%S')}] {cmd}")
                is_awake = False

                if any(k in cmd for k in ["close", "finish", "end", "stop"]):
                    speak("Shutting down protocol. Sending email.")
                    send_session_summary()
                    time.sleep(3)
                    os._exit(0)

                elif any(k in cmd for k in ["record", "log", "save", "cigar"]):
                    cigar_count_session += 1
                    speak("Logged to memory, sir.")
                    write_state()
                    threading.Thread(target=m.add, args=("Manual Cigar Log",), kwargs={"user_id": "ethan"},
                                     daemon=True).start()

                elif any(k in cmd for k in ["report", "status", "update"]):
                    speak(f"Distance to Thor is {current_spatial_dist}. Logs: {cigar_count_session}.")

                elif any(k in cmd for k in ["search for", "research", "look up"]):
                    parts = cmd.split("for ", 1) if "for " in cmd else cmd.split("research ", 1)
                    query = parts[1].strip() if len(parts) > 1 else ""
                    if query:
                        threading.Thread(target=conduct_research, args=(query,), daemon=True).start()
                    else:
                        speak("What topic would you like me to research?")

                elif any(k in cmd for k in ["recipe for", "how to cook", "how to bake"]):
                    query = cmd.split("for ")[1].strip() if "for " in cmd else cmd.split("cook ")[1].strip()
                    threading.Thread(target=find_recipe, args=(query,), daemon=True).start()

                elif any(k in cmd for k in ["shop for", "buy ", "find prices for "]):
                    query = cmd.split("for ")[1].strip() if "for " in cmd else cmd.split("buy ")[1].strip()
                    threading.Thread(target=find_best_prices, args=(query,), daemon=True).start()

                # THIS IS THE MISSING DEV ASSISTANT TRIGGER
                # THIS IS THE DEV ASSISTANT TRIGGER
                elif any(k in cmd for k in ["write a script", "code a", "build a"]):
                    query = ""
                    # We strip out the trigger words more cleanly
                    if "write a script " in cmd:
                        query = cmd.split("write a script ")[1].strip()
                    elif "code a " in cmd:
                        query = cmd.split("code a ")[1].strip()
                    elif "build a " in cmd:
                        query = cmd.split("build a ")[1].strip()

                    if query:
                        # Clean up common spoken artifacts that confuse the AI
                        query = query.replace("jarvis ", "").strip()
                        if query.startswith("for a "): query = query.replace("for a ", "", 1)
                        if query.startswith("for "): query = query.replace("for ", "", 1)
                        if query.startswith("to "): query = query.replace("to ", "", 1)

                        threading.Thread(target=generate_code, args=(query,), daemon=True).start()
                    else:
                        speak("What would you like me to code for you?")


threading.Thread(target=voice_listener, daemon=True).start()


# --- 6. DASHBOARD TRIGGER POLLER ---
def trigger_poller():
    global cigar_count_session
    last_ts = 0
    while True:
        try:
            if os.path.exists(CMD_FILE):
                with open(CMD_FILE) as f:
                    payload = json.load(f)
                ts = payload.get("ts", 0)
                if ts > last_ts:
                    last_ts = ts
                    action = payload.get("action", "")
                    query = payload.get("query", "")
                    if action == "recipe":
                        threading.Thread(target=find_recipe, args=(query,), daemon=True).start()
                    elif action == "shop":
                        threading.Thread(target=find_best_prices, args=(query,), daemon=True).start()
                    elif action == "research":
                        threading.Thread(target=conduct_research, args=(query,), daemon=True).start()
                    elif action == "log_cigar":
                        m.add("Dashboard Cigar Log", user_id="ethan")
                        cigar_count_session += 1
                        speak("Cigar logged from dashboard.")
                        write_state()
                    elif action == "code":
                        threading.Thread(target=generate_code, args=(query,), daemon=True).start()
        except:
            pass
        time.sleep(1)


threading.Thread(target=trigger_poller, daemon=True).start()


# --- 7. MAIN VISION PROTOCOL ---
def start_protocol():
    global last_thor_warning, thor_alert_count, current_spatial_dist
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    frame_count = 0
    SAFE_DIST = 450

    speak("Systems online. Spatial guard active.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        # FIXED CPU BUG: Will now properly process exactly every 3rd frame
        if frame_count % 3 == 0:
            # FIXED GPU UTILIZATION: Removed device='cpu'
            gen_res = general_model.predict(source=frame, verbose=False, classes=[16], conf=0.20)
            cig_res = cigar_model.predict(source=frame, verbose=False, conf=0.25)

            thor_boxes = gen_res[0].boxes.xyxy.tolist()
            cig_boxes = cig_res[0].boxes.xyxy.tolist()

            current_spatial_dist = "Searching..."

            for box in thor_boxes:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, "THOR", (int(box[0]), int(box[1]) - 10), 0, 0.5, (0, 255, 0), 2)

            for box in cig_boxes:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 165, 255), 2)

            if thor_boxes and cig_boxes:
                d = get_dist(thor_boxes[0], cig_boxes[0])
                current_spatial_dist = f"{int(d)} px"

                c_center = (int((cig_boxes[0][0] + cig_boxes[0][2]) / 2), int((cig_boxes[0][1] + cig_boxes[0][3]) / 2))
                t_center = (int((thor_boxes[0][0] + thor_boxes[0][2]) / 2),
                            int((thor_boxes[0][1] + thor_boxes[0][3]) / 2))

                cv2.line(frame, c_center, t_center, (0, 0, 255) if d < SAFE_DIST else (255, 255, 255), 2)

                if d < SAFE_DIST:
                    if time.time() - last_thor_warning > 40:
                        thor_alert_count += 1
                        speak("Thor is getting too close.")
                        cv2.imwrite("best_capture.jpg", frame)
                        last_thor_warning = time.time()
                        write_state()

        cv2.rectangle(frame, (0, 0), (450, 160), (0, 0, 0), -1)
        cv2.putText(frame, f"SPATIAL DIST: {current_spatial_dist}", (10, 40), 0, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"CIGARS LOGGED: {cigar_count_session}", (10, 80), 0, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, f"COMMAND: {latest_command[:25]}", (10, 120), 0, 0.6, (0, 255, 0), 1)

        if frame_count % 90 == 0:
            write_state()

        cv2.imshow("Jarvis Master", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_protocol()