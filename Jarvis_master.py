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
from ddgs import DDGS
from dotenv import load_dotenv

# Load environment variables from .env file
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

# --- SHARED STATE (written to disk so the Streamlit dashboard can read it) ---
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_state.json")
CMD_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_trigger.json")
command_log = []

def write_state():
    """Atomically write current session state to jarvis_state.json."""
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
# Known Vosk mishears of "Jarvis" kept as a fast-path allow-list
_VOSK_ALIASES = {"jarvis", "travis", "jarvas", "jarvish"}
_WAKE_SIMILARITY_THRESHOLD = 0.72  # 0–1; higher = stricter

def is_wake_word(cmd: str) -> bool:
    """Return True if any word in cmd is close enough to 'jarvis'."""
    for word in cmd.lower().split():
        if word in _VOSK_ALIASES:
            return True
        score = SequenceMatcher(None, word, "jarvis").ratio()
        if score >= _WAKE_SIMILARITY_THRESHOLD:
            return True
    return False

# --- 2. VOICE WORKER (The Butler Thread) ---
speech_queue = queue.Queue()

# Self-mute: timestamp until which the voice listener should ignore input
# (prevents Jarvis from hearing his own TTS through the microphone)
_speak_cooldown_until = 0.0


def voice_worker():
    global _speak_cooldown_until
    engine = pyttsx3.init()
    engine.setProperty('rate', 165)
    while True:
        text = speech_queue.get()
        if text is None: break
        try:
            # Estimate speech duration: ~2.75 words/sec at 165wpm, +1s buffer
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


# --- 3. THE "EFFICIENT" BRAIN FIX ---
# Mem0 embedder is forced to CPU to stop the GTX 970 crash, everything else is natural
config = {
    "llm": {
        "provider": "ollama",
        "config": {"model": "llama3.2", "ollama_base_url": OLLAMA_BASE_URL}
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "model_kwargs": {"device": "cpu"}  # <--- This prevents the hard crash
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {"collection_name": "jarvis_memories", "path": "./chroma_db"}
    }
}

print("🧠 Initializing Memory...")
m = Memory.from_config(config)

print("👁️ Initializing Vision Models...")
# YOLO stays on CPU for efficiency because the GTX 970 will crash the video stream
general_model = YOLO('yolov8n.pt')
cigar_model = YOLO('runs/detect/jarvis_cigar_model/weights/best.pt')

print("👂 Initializing Voice Recognition...")
vosk_model = Model("/home/manrig-13/Antigravity/my_jarvis/model")
rec = KaldiRecognizer(vosk_model, 16000)
audio_queue = queue.Queue()

# --- WARM UP OLLAMA on the remote server so model is in RAM ---
# Quick reachability check (5s) so we know immediately if the server is down
print("🔌 Checking AI brain connectivity...")
try:
    _ping = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=5)
    if _ping.status_code == 200:
        print("✅ Ollama server is reachable. Warming up model...")
        try:
            _warmup = requests.post(f'{OLLAMA_BASE_URL}/api/generate',
                                    json={'model': 'llama3.2', 'prompt': 'Hello.', 'stream': False},
                                    timeout=120)
            if _warmup.status_code == 200:
                print("✅ AI brain is warm and ready.")
            else:
                print(f"⚠️  Warmup got status {_warmup.status_code} — LLM commands may be slow.")
        except Exception as _we:
            print(f"⚠️  Warmup timed out: {_we} — LLM commands may be slow on first use.")
    else:
        print(f"⚠️  Ollama server returned status {_ping.status_code}.")
        speak("Warning: AI brain returned an unexpected status. Recipe and shopping commands may not work.")
except Exception as _e:
    print(f"❌ AI brain is UNREACHABLE at {OLLAMA_BASE_URL}: {_e}")
    speak("Warning: AI brain is offline. Recipe, shopping, and research commands will not work until it is back online.")


# --- 4. MATH & EMAIL ---
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
            print("Email credentials not set. Skipping summary email.")
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
        # Get brief summary for speaking
        summary = wikipedia.summary(query, sentences=2)
        speak("Here is what I found. " + summary)
        
        # Get full page content for report
        page = wikipedia.page(query)
        report_filename = f"Report_{query.replace(' ', '_')}.txt"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(f"--- RESEARCH REPORT: {page.title} ---\n\n")
            f.write(page.content)
            
        speak(f"I have saved a detailed report to {report_filename}.")
        
        # Log to memory
        m.add(f"Researched {query}", user_id="ethan")
    except wikipedia.exceptions.DisambiguationError as e:
        speak(f"The topic {query} is too broad. Please be more specific.")
    except wikipedia.exceptions.PageError:
        speak(f"I could not find any information on {query}.")
    except Exception as e:
        speak("I encountered an error while researching.")
        print(f"Research error: {e}")


def find_recipe(query):
    speak(f"Consulting the AI chef for {query}.")
    try:
        print(f"🌐 Contacting Ubuntu server for recipe: {query}...")
        response = requests.post(f'{OLLAMA_BASE_URL}/api/generate', json={
            'model': 'llama3.2', 
            'prompt': f'Correct any obvious speech-to-text typos in the item name. Then, give me a clear, step-by-step recipe for {query} with a list of ingredients and baking/cooking instructions. Keep it concise.', 
            'stream': False
        }, timeout=120)
        print(f"✅ Recipe response received (status {response.status_code}).")
        if response.status_code == 200:
            recipe_text = response.json().get('response', '')
            report_filename = f"Recipe_{query.replace(' ', '_')}.txt"
            with open(report_filename, "w", encoding="utf-8") as f:
                f.write(f"--- RECIPE: {query.upper()} ---\n\n")
                f.write(recipe_text)
            speak(f"I have saved the recipe for {query} to your folder.")
            m.add(f"Found a recipe for {query}", user_id="ethan")
        else:
            speak("I could not generate the recipe right now.")
            print(f"Recipe API bad status: {response.status_code} - {response.text[:200]}")
    except Exception as e:
        speak("I encountered an error while searching for the recipe.")
        print(f"❌ Recipe error: {type(e).__name__}: {e}")

def find_best_prices(query):
    speak(f"Shopping for {query}. Checking the web for the best prices.")
    try:
        # Step 1: Ask LLM to correct the item name, pick stores, AND give exclusion words
        cat_prompt = (
            f"Given the shopping item: '{query}', do the following:\n"
            f"1. Correct any obvious speech-to-text typos (e.g., 'keyword' -> 'keyboard').\n"
            f"2. Give me the corrected item name.\n"
            f"3. Give me 3 popular online retailers where I could buy this.\n"
            f"4. Give me 3-5 words for similar but WRONG variants I should EXCLUDE from search results.\n\n"
            f"Reply in EXACTLY this format, nothing else:\n"
            f"ITEM: <corrected item name>\n"
            f"STORES: <store1> OR <store2> OR <store3>\n"
            f"EXCLUDE: <word1>, <word2>, <word3>\n\n"
            f"Example for 'whole eggs':\n"
            f"ITEM: whole eggs\n"
            f"STORES: AMAZON OR WALMART OR TARGET\n"
            f"EXCLUDE: powdered, liquid, substitute, easter, artificial"
        )
        print(f"🌐 Contacting Ubuntu server to categorize item: {query}...")
        cat_resp = requests.post(f'{OLLAMA_BASE_URL}/api/generate', json={'model': 'llama3.2', 'prompt': cat_prompt, 'stream': False}, timeout=120)
        print(f"✅ Category response received (status {cat_resp.status_code}).")
        
        corrected_query = query
        store_string = "Amazon OR Walmart OR Target"
        exclude_words = []

        if cat_resp.status_code == 200:
            llm_text = cat_resp.json().get('response', '').strip()
            # Parse ITEM line
            for line in llm_text.split('\n'):
                line = line.strip()
                if line.upper().startswith('ITEM:'):
                    corrected_query = line.split(':', 1)[1].strip()
                elif line.upper().startswith('STORES:'):
                    store_string = line.split(':', 1)[1].strip()
                elif line.upper().startswith('EXCLUDE:'):
                    exclude_words = [w.strip().lower() for w in line.split(':', 1)[1].split(',') if w.strip()]
            
            # Validate store string
            if len(store_string) > 100 or "OR" not in store_string:
                store_string = "Amazon OR Walmart OR Target"

        # Build site-specific search query with exclusions
        sites = [f"site:{s.strip().lower().replace(' ', '')}.com" for s in store_string.split("OR")]
        site_query = " OR ".join(sites)
        exclude_query = " ".join([f"-{w}" for w in exclude_words[:5]])
        search_query = f'{site_query} "{corrected_query}" price {exclude_query}'

        print(f"🔎 Search query: {search_query}")  # Debug
            
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=10))
        
        if not results:
            speak("I could not find any shopping results for that item.")
            return

        # Pre-filter results: remove any result whose title/body contains exclusion words
        filtered = []
        for r in results:
            title = (r.get('title') or '').lower()
            body = (r.get('body') or '').lower()
            combined = title + ' ' + body
            if not any(ex in combined for ex in exclude_words):
                filtered.append(r)
        
        if not filtered:
            filtered = results[:5]  # Fallback if over-filtered

        context = "\n".join([f"Title: {r.get('title')}\nBody: {r.get('body')}\nLink: {r.get('href')}\n" for r in filtered])
        
        prompt = (
            f"I want to buy EXACTLY this item: '{corrected_query}'. Based strictly on the search snippets below, build a shopping report.\n\n"
            f"STRICT RULES:\n"
            f"1. ONLY include results for the EXACT item '{corrected_query}'. Do NOT include variants like: {', '.join(exclude_words) if exclude_words else 'powdered, liquid, substitute, or unrelated versions'}.\n"
            f"2. ONLY include results from actual store product pages (Amazon, Walmart, Target, etc.).\n"
            f"3. Each row MUST have a real price extracted from the snippet. If no price is found, skip that result.\n"
            f"4. The Link column must use the EXACT URL from the snippet. Do NOT modify or shorten URLs.\n"
            f"5. Sort from lowest price to highest price.\n"
            f"6. Format as a clean HTML table with columns: Store, Item Name, Price, Link (clickable <a href> tag).\n"
            f"7. Return ONLY the HTML table. No extra text, no explanations, no markdown.\n\n"
            f"Snippets:\n{context}"
        )
        
        print(f"🌐 Contacting Ubuntu server to generate shopping report...")
        response = requests.post(f'{OLLAMA_BASE_URL}/api/generate', json={
            'model': 'llama3.2', 
            'prompt': prompt, 
            'stream': False
        }, timeout=60)
        print(f"✅ Shopping report response received (status {response.status_code}).")
        
        if response.status_code == 200:
            report_html = response.json().get('response', '').replace("```html", "").replace("```", "")
            
            try:
                if not GMAIL_USER or not GMAIL_APP_PW:
                    speak("Email credentials not set. I cannot email the report.")
                else:
                    msg = MIMEMultipart()
                    msg['Subject'] = f"Jarvis Shopping Report - {query.title()}"
                    msg['From'] = GMAIL_USER
                    msg['To'] = f"{GMAIL_USER}, ether2803@gmail.com"
                    
                    html_content = f"<h3>Jarvis Shopping Report: {query.upper()}</h3><br>{report_html}"
                    msg.attach(MIMEText(html_content, 'html'))
                    
                    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                        server.login(GMAIL_USER, GMAIL_APP_PW)
                        server.send_message(msg)
                    speak(f"I have emailed you the shopping report for {query}.")
            except Exception as e:
                print(f"Email sending failed: {e}")
                speak("I encountered an error trying to email the report.")

            m.add(f"Shopped for {query}", user_id="ethan")
        else:
            speak("I failed to generate the shopping report summary.")
    except Exception as e:
        speak("I encountered an error while shopping online.")
        print(f"❌ Shopping error: {type(e).__name__}: {e}")


# --- 5. VOICE LISTENER ---
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
                break
            except:
                continue
    except:
        pass

    if not stream: return

    is_awake = False
    awake_time = 0
    AWAKE_DURATION = 25  # seconds Jarvis listens after hearing his name

    action_words = ["search for", "research", "look up", "recipe for", "how to cook", "how to bake", "record", "log", "save", "cigar", "report", "status", "update", "close", "finish", "end", "stop", "shop for", "buy ", "find prices for "]

    while True:
        data = audio_queue.get()
        if rec.AcceptWaveform(data):
            # Self-mute: ignore audio while Jarvis is still speaking (prevents self-triggering)
            if time.time() < _speak_cooldown_until:
                continue

            res = json.loads(rec.Result())
            cmd = res.get('text', '').lower()
            if cmd:
                print(f"🗣️  Heard: {cmd}")  # Debug print so you know what vosk heard

            # Check if wake word logic timed out
            if is_awake and (time.time() - awake_time > AWAKE_DURATION):
                is_awake = False
                print("💤 Jarvis drifted back to sleep.")
                speak("Going back to sleep.")

            # If we hear the wake word, wake him up! (uses similarity scorer, not static list)
            if is_wake_word(cmd):
                # Don't say "Yes sir" repeatedly if he's already awake
                if not is_awake:
                    speak("Yes, sir?")
                is_awake = True
                awake_time = time.time()
                print("🚨 JARVIS IS AWAKE AND LISTENING")

            # Now, only process actions IF he is awake
            if is_awake and any(a in cmd for a in action_words):
                latest_command = cmd
                command_log.append(f"[{time.strftime('%H:%M:%S')}] {cmd}")
                print(f"🔊 Command Received: {cmd}")

                # We consume the command, so go back to sleep
                is_awake = False

                # ── CLOSE SESSION — checked FIRST so "stop"/"end" can't be swallowed by other branches ──
                if any(k in cmd for k in ["close", "finish", "end", "stop"]):
                    speak("Concluding session. Processing your email now.")
                    send_session_summary()
                    time.sleep(3)
                    os._exit(0)

                elif any(k in cmd for k in ["record", "log", "save", "cigar"]):
                    cigar_count_session += 1
                    speak("Logged to memory, sir.")
                    write_state()
                    # Run memory write in background so voice thread stays responsive
                    threading.Thread(target=m.add, args=("Manual Cigar Log",), kwargs={"user_id": "ethan"}, daemon=True).start()

                elif any(k in cmd for k in ["report", "status", "update"]):
                    speak(f"Distance to Thor is {current_spatial_dist}. You have {cigar_count_session} logs today.")

                elif any(k in cmd for k in ["search for", "research", "look up"]):
                    query = ""
                    if "search for " in cmd:
                        query = cmd.split("search for ")[1].strip()
                    elif "research " in cmd:
                        query = cmd.split("research ")[1].strip()
                    elif "look up " in cmd:
                        query = cmd.split("look up ")[1].strip()
                    
                    if query:
                        query = query.replace("jarvis ", "").strip()
                        speak("On it, sir. Researching now.")
                        threading.Thread(target=conduct_research, args=(query,), daemon=True).start()
                    else:
                        speak("What would you like me to research?")

                elif any(k in cmd for k in ["recipe for", "how to cook", "how to bake"]):
                    query = ""
                    if "recipe for " in cmd:
                        query = cmd.split("recipe for ")[1].strip()
                    elif "how to cook " in cmd:
                        query = cmd.split("how to cook ")[1].strip()
                    elif "how to bake " in cmd:
                        query = cmd.split("how to bake ")[1].strip()
                    
                    if query:
                        query = query.replace("jarvis ", "").strip()
                        speak("On it, sir. Contacting the AI chef now. This may take a moment.")
                        threading.Thread(target=find_recipe, args=(query,), daemon=True).start()
                    else:
                        speak("What recipe would you like me to find?")

                elif any(k in cmd for k in ["shop for", "buy ", "find prices for "]):
                    query = ""
                    if "shop for " in cmd:
                        query = cmd.split("shop for ")[1].strip()
                    elif "buy " in cmd:
                        query = cmd.split("buy ")[1].strip()
                    elif "find prices for " in cmd:
                        query = cmd.split("find prices for ")[1].strip()
                    
                    if query:
                        query = query.replace("jarvis ", "").strip()
                        speak("On it, sir. Searching the web for the best prices. This may take a moment.")
                        threading.Thread(target=find_best_prices, args=(query,), daemon=True).start()
                    else:
                        speak("What would you like me to shop for?")


threading.Thread(target=voice_listener, daemon=True).start()


# --- 7. DASHBOARD TRIGGER POLLER ---
def trigger_poller():
    """Poll jarvis_trigger.json for commands sent from the Streamlit dashboard."""
    global cigar_count_session
    last_ts = 0
    while True:
        try:
            if os.path.exists(CMD_FILE):
                with open(CMD_FILE) as f:
                    payload = json.load(f)
                ts = payload.get("ts", 0)
                if ts > last_ts:   # new trigger
                    last_ts = ts
                    action = payload.get("action", "")
                    query  = payload.get("query", "")
                    print(f"📲 Dashboard trigger: {action} → {query}")
                    if action == "recipe":
                        threading.Thread(target=find_recipe, args=(query,), daemon=True).start()
                    elif action == "shop":
                        threading.Thread(target=find_best_prices, args=(query,), daemon=True).start()
                    elif action == "research":
                        threading.Thread(target=conduct_research, args=(query,), daemon=True).start()
                    elif action == "log_cigar":
                        m.add("Dashboard Cigar Log", user_id="ethan")
                        cigar_count_session += 1
                        speak("Cigar logged from dashboard, sir.")
                        write_state()
        except Exception as e:
            print(f"Trigger poll error: {e}")
        time.sleep(1)


threading.Thread(target=trigger_poller, daemon=True).start()


# --- 6. MAIN VISION PROTOCOL ---
def start_protocol():
    global last_thor_warning, thor_alert_count, current_spatial_dist
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    frame_count = 0
    SAFE_DIST = 450  # Wide spatial net

    speak("Systems online. Spatial guard active.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        # Smooth frame skipping for efficiency (Every 3rd frame)
        if frame_count % 3 == 0:
            gen_res = general_model.predict(source=frame, device='cpu', verbose=False, classes=[16], conf=0.20)
            cig_res = cigar_model.predict(source=frame, device='cpu', verbose=False, conf=0.25)

            thor_boxes = gen_res[0].boxes.xyxy.tolist()
            cig_boxes = cig_res[0].boxes.xyxy.tolist()

            current_spatial_dist = "Searching..."

            # --- DRAW VISUAL DEBUGGER ---
            for box in thor_boxes:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, "THOR", (int(box[0]), int(box[1]) - 10), 0, 0.5, (0, 255, 0), 2)

            for box in cig_boxes:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 165, 255), 2)

            # --- SPATIAL MATH ---
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

        # HUD
        cv2.rectangle(frame, (0, 0), (450, 160), (0, 0, 0), -1)
        cv2.putText(frame, f"SPATIAL DIST: {current_spatial_dist}", (10, 40), 0, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"CIGARS LOGGED: {cigar_count_session}", (10, 80), 0, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, f"COMMAND: {latest_command[:25]}", (10, 120), 0, 0.6, (0, 255, 0), 1)

        # Write state every ~90 frames (~3 seconds) for the dashboard
        if frame_count % 90 == 0:
            write_state()

        cv2.imshow("Jarvis Master", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_protocol()