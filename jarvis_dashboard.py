"""
jarvis_dashboard.py — Jarvis Live Dashboard
Run with: streamlit run jarvis_dashboard.py
Reads jarvis_state.json written by Jarvis_master.py every ~3 seconds.
"""
import json
import os
import time
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
STATE_FILE = BASE_DIR / "jarvis_state.json"
CMD_FILE   = BASE_DIR / "jarvis_trigger.json"   # dashboard → master comms

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JARVIS Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS (dark HUD aesthetic) ──────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

    html, body, [class*="css"] {
        background-color: #020c14;
        color: #00d4ff;
        font-family: 'Share Tech Mono', monospace;
    }

    .stApp { background-color: #020c14; }

    .jarvis-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.8rem;
        font-weight: 900;
        color: #00d4ff;
        text-shadow: 0 0 20px #00d4ff88, 0 0 40px #00d4ff44;
        letter-spacing: 0.25em;
        text-align: center;
        padding: 0.5rem 0 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #005f7a;
        font-size: 0.75rem;
        letter-spacing: 0.3em;
        margin-bottom: 1.5rem;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #001a2e 0%, #00111f 100%);
        border: 1px solid #00d4ff33;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        box-shadow: 0 0 16px #00d4ff18;
    }
    div[data-testid="metric-container"] label {
        color: #005f7a !important;
        font-size: 0.7rem;
        letter-spacing: 0.2em;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem;
    }

    /* Status pill */
    .status-online  { color: #00ff88; font-weight: bold; font-size: 0.9rem; letter-spacing: 0.15em; }
    .status-offline { color: #ff4444; font-weight: bold; font-size: 0.9rem; letter-spacing: 0.15em; }

    /* Command log box */
    .log-box {
        background: #00090f;
        border: 1px solid #00d4ff22;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.78rem;
        color: #00d4ffbb;
        max-height: 300px;
        overflow-y: auto;
        line-height: 1.8;
    }

    /* Section headers */
    .section-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 0.35em;
        color: #005f7a;
        border-bottom: 1px solid #00d4ff22;
        padding-bottom: 0.3rem;
        margin: 1.2rem 0 0.7rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #001a2e, #002a40);
        color: #00d4ff;
        border: 1px solid #00d4ff55;
        border-radius: 6px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.1em;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: #003a55;
        border-color: #00d4ff;
        box-shadow: 0 0 12px #00d4ff44;
        color: #ffffff;
    }

    /* Text input */
    .stTextInput > div > div > input {
        background: #00090f;
        color: #00d4ff;
        border: 1px solid #00d4ff33;
        border-radius: 6px;
        font-family: 'Share Tech Mono', monospace;
    }

    /* Divider */
    hr { border-color: #00d4ff11; }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helper: load state ────────────────────────────────────────────────────────
def load_state() -> dict:
    if not STATE_FILE.exists():
        return {
            "status": "offline",
            "uptime_mins": 0,
            "latest_command": "—",
            "cigar_count": 0,
            "thor_alerts": 0,
            "spatial_dist": "—",
            "command_log": [],
            "last_updated": "—",
        }
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def send_trigger(action: str, query: str = ""):
    """Write a trigger file that Jarvis_master.py can poll and act on."""
    payload = {"action": action, "query": query, "ts": time.time()}
    try:
        with open(CMD_FILE, "w") as f:
            json.dump(payload, f)
        st.toast(f"⚡ Sent to Jarvis: {action} {query}", icon="🤖")
    except Exception as e:
        st.error(f"Could not write trigger: {e}")


# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown('<div class="jarvis-title">J.A.R.V.I.S</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">JUST A RATHER VERY INTELLIGENT SYSTEM  ·  LIVE DASHBOARD</div>',
            unsafe_allow_html=True)

state = load_state()
is_online = state.get("status") == "online"

# Status + last-updated row
col_s, col_t = st.columns([1, 3])
with col_s:
    pill_class = "status-online" if is_online else "status-offline"
    pill_icon  = "● ONLINE" if is_online else "● OFFLINE"
    st.markdown(f'<span class="{pill_class}">{pill_icon}</span>', unsafe_allow_html=True)
with col_t:
    st.markdown(
        f'<span style="color:#004455;font-size:0.72rem;">Last sync: {state.get("last_updated", "—")}</span>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Metric cards ─────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("⏱ UPTIME", f'{state.get("uptime_mins", 0)} min')
m2.metric("🚬 CIGARS LOGGED", state.get("cigar_count", 0))
m3.metric("🐕 THOR ALERTS", state.get("thor_alerts", 0))
m4.metric("📐 SPATIAL DIST", state.get("spatial_dist", "—"))

# ── Latest command ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">LAST COMMAND RECEIVED</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="log-box" style="max-height:60px;">'
    f'<span style="color:#00ff88;">▶ </span>{state.get("latest_command", "—")}'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Command log ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">COMMAND LOG</div>', unsafe_allow_html=True)
log_lines = state.get("command_log", [])
if log_lines:
    log_html = "<br>".join(
        f'<span style="color:#004455;">{line.split("]")[0]}]</span>'
        f'<span style="color:#00d4ffcc;">{"] ".join(line.split("]")[1:])}</span>'
        for line in reversed(log_lines)
    )
else:
    log_html = '<span style="color:#003344;">No commands logged yet.</span>'

st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

# ── Quick-fire controls ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">QUICK-FIRE CONTROLS</div>', unsafe_allow_html=True)

query_input = st.text_input(
    "Query / item name",
    placeholder='e.g. "scrambled eggs"  or  "mechanical keyboard"',
    label_visibility="collapsed",
)

b1, b2, b3, b4 = st.columns(4)
with b1:
    if st.button("🍳  GET RECIPE", key="btn_recipe"):
        if query_input.strip():
            send_trigger("recipe", query_input.strip())
        else:
            st.warning("Enter a dish name first.")
with b2:
    if st.button("🛒  SHOP FOR ITEM", key="btn_shop"):
        if query_input.strip():
            send_trigger("shop", query_input.strip())
        else:
            st.warning("Enter an item name first.")
with b3:
    if st.button("🔭  RESEARCH TOPIC", key="btn_research"):
        if query_input.strip():
            send_trigger("research", query_input.strip())
        else:
            st.warning("Enter a topic first.")
with b4:
    if st.button("🚬  LOG CIGAR", key="btn_cigar"):
        send_trigger("log_cigar")

st.markdown("---")

# ── Footer / auto-refresh ─────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;color:#003344;font-size:0.65rem;letter-spacing:0.2em;">'
    'AUTO-REFRESH EVERY 3 SECONDS  ·  JARVIS v2.0'
    '</div>',
    unsafe_allow_html=True,
)

# Auto-refresh — rerun every 1 second
time.sleep(1)
st.rerun()
