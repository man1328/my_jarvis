# Jarvis — AI Home Assistant with Voice, Vision & Automation

A Python-based always-on assistant that combines **wake-word voice control (Vosk)**, **local LLM reasoning (Ollama)**, **YOLOv8 vision (dog/cigar detection)**, **Gmail notifications**, and **persistent memory (Mem0 + ChromaDB)** — with a live Streamlit dashboard for monitoring and control.

---

## Why this exists

I wanted a *local* assistant that actually operates my home: "Jarvis, what's the weather?" → speaks. "Jarvis, Thor near cigars?" → vision alert + email. "Jarvis, backup Nextcloud" → triggers cron + reports result. No cloud, no subscription, full privacy. Runs on an Ubuntu box with an RTX A3000.

---

## Demo

| Interface | What you see |
|-----------|--------------|
| **Voice** | "Jarvis" → Vosk wake word → natural language → Ollama → speaks response |
| **Vision** | YOLOv8 tracks dog (Thor) spatial distance from cigar zone → alert + annotated frame + email |
| **Dashboard** | Streamlit at `localhost:8501` — live camera feed, command log, cigar/Thor counters, one-click Recipe/Shopping/Research buttons |
| **Memory** | Mem0 + ChromaDB — "What did I do today?" → reads back logged events |
| **Email** | Session summary on shutdown; shopping reports; vision alerts |

> **30-second proof:** `python Jarvis_master.py` → say "Jarvis, recipe for carbonara" → watches it consult Ollama, save recipe, speak confirmation. Dashboard updates in real-time.

---

## Stack

| Layer | Tech |
|-------|------|
| **Voice** | Vosk (offline STT), pyttsx3 (TTS) |
| **LLM** | Ollama (host GPU) — `qwen2.5:7b`, `llama3.2:3b` |
| **Vision** | YOLOv8 (Ultralytics), OpenCV — custom classes: `thor`, `cigar` |
| **Memory** | Mem0 (embedding + retrieval) + ChromaDB (vector store) |
| **Email** | Gmail SMTP (App Password) — HTML reports |
| **Dashboard** | Streamlit — `jarvis_dashboard.py` (reads `jarvis_state.json`, writes `jarvis_trigger.json`) |
| **Scheduling** | Systemd service + health-check cron (emails if dead) |
| **Hardware** | Ubuntu Server, RTX A3000 6GB, USB camera |

---

## Quick Start

```bash
# 1. Prereqs (Ubuntu 24.04)
sudo apt-get update && sudo apt-get install -y python3-venv python3-opencv libcamera-dev

# 2. Clone & setup
git clone <repo>
cd my_jarvis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env: GMAIL_USER, GMAIL_APP_PW, OLLAMA_BASE_URL=http://<ollama-host>:11434

# 4. Download Vosk model (once)
# Place in model/ (gitignored) — https://alphacephei.com/vosk/models

# 5. Run
python Jarvis_master.py
# Press 'q' in video window to shutdown + email session summary

# 6. Dashboard (second terminal)
source .venv/bin/activate
streamlit run jarvis_dashboard.py
# Open http://localhost:8501
```

---

## Project Structure

```
my_jarvis/
├── Jarvis_master.py          # Main daemon: voice loop, vision, commands, email
├── jarvis_dashboard.py       # Streamlit dashboard (live feed, log, triggers)
├── keys_loader.py            # Secure .env loader
├── config.py.example         # Path config template (gitignored)
├── requirements.txt
├── .env.example              # Secrets template (Gmail, Ollama URL)
├── model/                    # Vosk model (local, gitignored)
├── chroma_db/                # Mem0 vector store (local, gitignored)
├── data/                     # Datasets (local)
├── runs/                     # Training output (local)
├── jarvis_state.json         # Jarvis → dashboard (gitignored)
├── jarvis_trigger.json       # Dashboard → Jarvis (gitignored)
└── README.md                 # This file
```

---

## Core Capabilities

| Command | Action |
|---------|--------|
| "Jarvis, recipe for *X*" | Asks Ollama, saves to file, speaks summary |
| "Jarvis, shop for *X*" | Scrapes web for best prices, emails HTML report |
| "Jarvis, research *X*" | Wikipedia lookup, saves to text file |
| "Jarvis, log cigar" | Increments session counter, persists |
| "Jarvis, what did I do today?" | Queries Mem0, reads back events |
| (Vision) | Continuous YOLO inference — Thor distance from cigar zone → alert + email |

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Vosk offline STT** | No cloud API, works fully offline, privacy-first. Tradeoff: accuracy ~85% vs. cloud 95%+. |
| **Secondary fuzzy-score filter** | Vosk mishears "Jarvis" as "garbage", "harvest", "harbor" → added Levenshtein filter on wake-word candidates, cut false positives 90%. |
| **Ollama on host, not container** | GPU passthrough complexity; host Ollama + `OLLAMA_BASE_URL` in `.env` works reliably. |
| **YOLOv8 custom classes** | Trained on ~200 labeled frames of Thor (dog) + cigars. Runs at ~15 FPS on RTX A3000. |
| **Mem0 + ChromaDB** | Mem0 handles embedding + retrieval logic; Chroma persists. "What did I do today?" works. |
| **Streamlit dashboard via JSON files** | Zero-coupling: Jarvis writes state, dashboard reads. No sockets, no Redis, survives restarts. |
| **Systemd + health-check cron** | `jarvis.service` keeps it alive; cron every 5 min → `systemctl is-active` + email if down. |

---

## What's Not Here (Known Limits)

- **No multi-user / auth** — single-user personal assistant.
- **Vosk accuracy ceiling** — noisy environments need better mic or fine-tuned vocabulary (future: fine-tune Vosk on "Jarvis" + household names).
- **Single camera** — vision guard watches one zone. Multi-cam would need process-per-camera or threaded inference.
- **No Stable Diffusion** — GPU VRAM (6 GB) tight for SD + YOLO + Ollama simultaneously. MusicGen works (separate service).
- **ServiceNow integration** — config has placeholder `SERVICENOW_PW` but not implemented.
- **Windows/macOS** — Linux-specific (Vosk model paths, systemd, `/dev/video0`).

---

## My Role

- Designed the master loop: voice → intent → action → memory → response
- Integrated Vosk + Ollama + YOLOv8 + Mem0 + ChromaDB + Gmail into one daemon
- Built the Streamlit dashboard (live feed, command log, one-click triggers)
- Trained custom YOLO classes (Thor, cigar) — labeled data, tuned confidence thresholds
- Implemented fuzzy wake-word filter to tame Vosk false positives
- Wired systemd service + health-check cron + email alerting
- AI-assisted: YOLO training script, Mem0 integration patterns, Streamlit component layout, email HTML templates

---

## What's Next (Roadmap)

| Priority | Item |
|----------|------|
| **High** | Streamlit dashboard: live camera feed, command log, quick-action buttons (recipe/shop/research/log) |
| **High** | Startup briefing: time + weather + outstanding Mem0 events on "Systems online" |
| **Medium** | APScheduler for recurring tasks: "Every morning 8 AM, weather + email" |
| **Medium** | "What did I do today?" voice command → Mem0 query → TTS response |
| **Low** | ServiceNow ticket open/close by voice |
| **Low** | Multi-camera support (second room) |
| **Low** | Stable Diffusion image generation (needs VRAM budgeting) |

---

## License

MIT — personal project, open for inspection.
