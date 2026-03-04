# 🤖 Jarvis AI Assistant

A Python-based AI assistant for home automation and personal productivity. Jarvis integrates local AI models (via Ollama), computer vision (YOLO), voice recognition (Vosk), and Gmail notifications into a single always-on system.

## ✨ Features

- **Voice Control** — Wake word detection ("Jarvis") with natural language command parsing
- **AI Recipes** — Ask Jarvis for a recipe and it consults an LLM and saves the result
- **Smart Shopping** — Searches the web for best prices and emails you an HTML report
- **Research Reports** — Wikipedia lookups saved to text files
- **Vision Guard** — YOLO-powered camera monitoring with spatial distance tracking (for Thor 🐕)
- **Email Alerts** — Session summaries and shopping reports sent via Gmail
- **Persistent Memory** — Logs events using [Mem0](https://mem0.ai/) + ChromaDB

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone git@github.com:man1328/my_jarvis.git
cd my_jarvis
```

### 2. Set up a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 🔐 Configuration

This project uses a `.env` file for secrets. **Never commit `.env` to git** — it is already in `.gitignore`.

### Set up your secrets
```bash
cp .env.example .env
```
Then edit `.env` with your real values:

| Variable | Description |
|---|---|
| `GMAIL_USER` | Your Gmail address |
| `GMAIL_APP_PW` | 16-character Gmail App Password (not your regular password) |
| `OLLAMA_BASE_URL` | URL of your Ollama server (e.g. `http://192.168.1.24:11434`) |

> **Note:** Ollama does **not** require an API key. It runs as a local server with no authentication by default.

### Gmail App Password
If you use 2-Step Verification on Gmail, you need an App Password:
1. Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
2. Generate a password for "Mail"
3. Paste the 16-character code into `.env`

## 📁 Project Structure

```
Jarvis_master.py    — Main application (voice, vision, commands)
keys_loader.py      — Secure credential loader from .env
app.py              — Streamlit dashboard (optional UI)
requirements.txt    — Python dependencies
.env.example        — Template for secrets (safe to commit)
config.py.example   — Template for local path config (safe to commit)
model/              — Vosk speech recognition model (local only)
chroma_db/          — Persistent memory vector store (local only)
data/               — Datasets (local only)
runs/               — Training output (local only)
```

## 🚀 Running Jarvis

Make sure your Ollama server is running on your Ubuntu machine, then:

```bash
source .venv/bin/activate
python Jarvis_master.py
```

Press `q` in the video window to shut down and send the session summary email.

## 📊 Live Dashboard

Run the Streamlit dashboard in a **second terminal** while Jarvis is running:

```bash
source .venv/bin/activate
streamlit run jarvis_dashboard.py
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser. The dashboard:

- Shows live session stats (uptime, cigars logged, Thor alerts, spatial distance)
- Displays a scrolling command log of everything Jarvis heard
- Lets you trigger **Recipe**, **Shopping**, **Research**, and **Log Cigar** with buttons — no voice needed
- Auto-refreshes every 3 seconds

> The dashboard communicates with Jarvis via `jarvis_state.json` (Jarvis → dashboard) and `jarvis_trigger.json` (dashboard → Jarvis). Both files are gitignored.


Next steps:
- 🗺️ Proposed Next Steps for Jarvis
Here's what I'd prioritize, roughly in order of impact:

🥇 High Impact (Do These Next)
1. Live Dashboard (Streamlit) The current 

app.py
 is an unrelated MNIST demo. Jarvis deserves a real dashboard showing:

Live camera feed from the vision guard
Command log ("last heard" / "last action")
Cigar & Thor alert counts for the session
Quick-fire buttons to trigger shopping/recipe/research without voice
2. Weather + Time Briefing on Startup When Jarvis says "Systems online" at startup, have it also announce: the time, today's weather, and any outstanding memory events from Mem0. Makes it feel much more like a real assistant.

3. Smarter Wake Word Handling Right now the wake word list includes fuzzy matches ("garbage", "harvest") because Vosk mishears "Jarvis." A better fix is to fine-tune the Vosk vocabulary or add a secondary fuzzy-match score so false positives are reduced.

🥈 Good Additions (After the Above)
4. Scheduled / Recurring Tasks Use schedule or APScheduler to let Jarvis do things on a timer — e.g., "Every morning at 8am, check the weather and email me" or "Remind me every 30 mins if Thor is near cigars."

5. Voice Command: "What did I do today?" Query Mem0 for today's logged events and have Jarvis read them back. The memory system is already wired up but barely used.

6. ServiceNow Integration The config already has a placeholder SERVICENOW_PW. If you use ServiceNow at work, Jarvis could open/close tickets or report on queue counts by voice.

🥉 Longer Term
7. Multi-camera Support Extend the vision guard to handle more than one camera — e.g., a second camera in another room.

8. Text-to-Image (Stable Diffusion) Since the Ubuntu server already has GPU power, hook up a Stable Diffusion endpoint so Jarvis can generate images on command ("Show me a diagram of...")

My top recommendation: Start with the Streamlit dashboard — it's the most immediately useful, makes everything Jarvis does visible, and would also let you control him from your phone or another machine on the network. Want me to build that?