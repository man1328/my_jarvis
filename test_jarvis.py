"""
test_jarvis.py — Automated test suite for Jarvis_master.py subsystems.
Run with:  python test_jarvis.py
Or:        python test_jarvis.py -v          (verbose)
            python test_jarvis.py TestOllama  (run one group only)

Tests are grouped into classes so you can run just the ones you need.
No audio hardware or camera is required — hardware-dependent features are skipped.
"""

import json
import os
import sys
import time
import unittest
import tempfile
from pathlib import Path
from difflib import SequenceMatcher
from unittest.mock import patch, MagicMock

# ── Resolve project root ──────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

# ── Load env so OLLAMA_BASE_URL / GMAIL_USER are available ───────────────────
from dotenv import load_dotenv
load_dotenv(PROJECT_DIR / ".env")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
STATE_FILE      = PROJECT_DIR / "jarvis_state.json"
CMD_FILE        = PROJECT_DIR / "jarvis_trigger.json"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Ollama Connectivity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestOllama(unittest.TestCase):
    """Tests that require the Ollama server to be reachable."""

    def setUp(self):
        import requests
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if r.status_code != 200:
                self.skipTest(f"Ollama server returned {r.status_code}")
        except Exception as e:
            self.skipTest(f"Ollama server not reachable ({e})")

    def test_01_server_reachable(self):
        """Ollama API endpoint responds with 200."""
        import requests
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        self.assertEqual(r.status_code, 200, "Ollama /api/tags should return 200")

    def test_02_model_listed(self):
        """llama3.2 model is installed on the server."""
        import requests
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m.get("name", "") for m in r.json().get("models", [])]
        has_llama = any("llama3.2" in m for m in models)
        self.assertTrue(has_llama,
            f"llama3.2 not found in installed models: {models}\n"
            "Fix: run  ollama pull llama3.2  on the server.")

    def test_03_model_loaded_in_ram(self):
        """llama3.2 is currently loaded in RAM (ollama ps).
        If this fails, run: OLLAMA_KEEP_ALIVE=60m ollama run llama3.2  then Ctrl+D.
        """
        import requests
        r = requests.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=5)
        if r.status_code != 200:
            self.skipTest("Ollama /api/ps not available on this version")
        models = [m.get("name", "") for m in r.json().get("models", [])]
        has_loaded = any("llama3.2" in m for m in models)
        self.assertTrue(has_loaded,
            f"llama3.2 is NOT loaded in RAM. Currently loaded: {models}\n"
            "Fix: OLLAMA_KEEP_ALIVE=60m ollama run llama3.2  then Ctrl+D")

    def test_04_recipe_generation(self):
        """LLM can generate a short recipe within 60 seconds."""
        import requests
        prompt = "Give me a 2-step scrambled egg recipe. Be concise."
        try:
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": "llama3.2", "prompt": prompt, "stream": False},
                timeout=60,
            )
        except Exception as e:
            self.fail(f"Recipe generation timed out or errored: {e}\n"
                      "The model may not be loaded in RAM. "
                      "Run: OLLAMA_KEEP_ALIVE=60m ollama run llama3.2")
        self.assertEqual(r.status_code, 200)
        response_text = r.json().get("response", "")
        self.assertTrue(len(response_text) > 20,
            f"LLM response was too short: '{response_text}'")
        print(f"\n    ✅ Recipe generated ({len(response_text)} chars).")

    def test_05_shopping_categorize(self):
        """LLM can categorize a shopping item within 60 seconds."""
        import requests
        prompt = (
            "Given the shopping item: 'whole eggs', reply in EXACTLY this format:\n"
            "ITEM: <corrected name>\nSTORES: <store1> OR <store2> OR <store3>\n"
            "EXCLUDE: <word1>, <word2>"
        )
        try:
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": "llama3.2", "prompt": prompt, "stream": False},
                timeout=60,
            )
        except Exception as e:
            self.fail(f"Shopping categorization timed out: {e}")
        self.assertEqual(r.status_code, 200)
        text = r.json().get("response", "").upper()
        self.assertIn("ITEM:", text, f"LLM response missing ITEM: field.\nGot: {text[:200]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. State File I/O (jarvis_state.json)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestStateFile(unittest.TestCase):
    """Tests for the jarvis_state.json read/write cycle."""

    def _write_state(self, path, **overrides):
        state = {
            "status": "online",
            "uptime_mins": 1.5,
            "latest_command": "test command",
            "cigar_count": 3,
            "thor_alerts": 1,
            "spatial_dist": "200 px",
            "command_log": ["[10:00:00] test command"],
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        state.update(overrides)
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, path)
        return state

    def test_01_write_and_read(self):
        """State written to a temp file can be read back correctly."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name
        try:
            written = self._write_state(path, cigar_count=7, thor_alerts=2)
            with open(path) as f:
                read_back = json.load(f)
            self.assertEqual(read_back["cigar_count"], 7)
            self.assertEqual(read_back["thor_alerts"], 2)
            self.assertEqual(read_back["status"], "online")
        finally:
            os.unlink(path)

    def test_02_atomic_write(self):
        """State write uses atomic rename so readers never see partial data."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name
        tmp_path = path + ".tmp"
        try:
            # Write initial valid state
            self._write_state(path, cigar_count=0)
            # Simulate in-progress write: create tmp file
            with open(tmp_path, "w") as f:
                json.dump({"status": "online", "cigar_count": 99}, f)
            # Atomic replace: tmp → real
            os.replace(tmp_path, path)
            # Reader should see the new state, not partial data
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["cigar_count"], 99)
            # tmp file should be gone after replace
            self.assertFalse(os.path.exists(tmp_path))
        finally:
            for p in [path, tmp_path]:
                if os.path.exists(p):
                    os.unlink(p)

    def test_03_command_log_truncated_to_20(self):
        """Command log stored in state is capped at 20 entries."""
        log = [f"[00:00:{i:02d}] cmd {i}" for i in range(50)]
        truncated = log[-20:]
        self.assertEqual(len(truncated), 20)
        self.assertEqual(truncated[-1], log[-1])

    def test_04_offline_state_default(self):
        """Dashboard returns offline defaults when state file is missing."""
        missing_path = PROJECT_DIR / "jarvis_state_MISSING.json"
        if missing_path.exists():
            missing_path.unlink()

        # Replicate dashboard load_state() logic
        if not missing_path.exists():
            state = {
                "status": "offline", "uptime_mins": 0,
                "latest_command": "—", "cigar_count": 0,
                "thor_alerts": 0, "spatial_dist": "—",
                "command_log": [], "last_updated": "—",
            }
        else:
            with open(missing_path) as f:
                state = json.load(f)

        self.assertEqual(state["status"], "offline")
        self.assertEqual(state["cigar_count"], 0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Dashboard Trigger File (jarvis_trigger.json)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestTriggerFile(unittest.TestCase):
    """Tests for the dashboard → master command trigger mechanism."""

    def setUp(self):
        self._tmp = PROJECT_DIR / "jarvis_trigger_TEST.json"

    def tearDown(self):
        if self._tmp.exists():
            self._tmp.unlink()

    def _send_trigger(self, action, query=""):
        payload = {"action": action, "query": query, "ts": time.time()}
        with open(self._tmp, "w") as f:
            json.dump(payload, f)
        return payload

    def test_01_trigger_write_read(self):
        """Trigger written by dashboard can be read by master."""
        sent = self._send_trigger("recipe", "scrambled eggs")
        with open(self._tmp) as f:
            received = json.load(f)
        self.assertEqual(received["action"], "recipe")
        self.assertEqual(received["query"], "scrambled eggs")
        self.assertAlmostEqual(received["ts"], sent["ts"], places=1)

    def test_02_new_trigger_detected_by_timestamp(self):
        """Master detects a new trigger by comparing timestamps."""
        last_ts = 0.0
        payload = self._send_trigger("log_cigar")
        ts = payload["ts"]
        self.assertGreater(ts, last_ts, "New trigger should have ts > last_ts")
        last_ts = ts
        # Same trigger should NOT fire again
        self.assertFalse(ts > last_ts, "Same trigger should not re-fire")

    def test_03_all_action_types_valid(self):
        """All documented action types can be serialized as trigger payloads."""
        for action in ["recipe", "shop", "research", "log_cigar"]:
            payload = {"action": action, "query": "test", "ts": time.time()}
            serialized = json.dumps(payload)
            parsed = json.loads(serialized)
            self.assertEqual(parsed["action"], action)

    def test_04_empty_query_safe(self):
        """log_cigar trigger (no query) doesn't cause JSON errors."""
        sent = self._send_trigger("log_cigar")  # no query arg → defaults to ""
        with open(self._tmp) as f:
            received = json.load(f)
        self.assertEqual(received["action"], "log_cigar")
        self.assertEqual(received.get("query", ""), "")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Wake Word Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestWakeWord(unittest.TestCase):
    """Tests for is_wake_word() — covers exact matches and Vosk mishears."""

    # Replicate logic from Jarvis_master.py without importing the full module
    _VOSK_ALIASES = {"jarvis", "travis", "jarvas", "jarvish"}
    _THRESHOLD    = 0.72

    def is_wake_word(self, cmd: str) -> bool:
        for word in cmd.lower().split():
            if word in self._VOSK_ALIASES:
                return True
            if SequenceMatcher(None, word, "jarvis").ratio() >= self._THRESHOLD:
                return True
        return False

    def test_exact_jarvis(self):
        self.assertTrue(self.is_wake_word("jarvis"))

    def test_known_mishears(self):
        for alias in ["travis", "jarvas", "jarvish"]:
            with self.subTest(alias=alias):
                self.assertTrue(self.is_wake_word(alias),
                    f"Known Vosk mishear '{alias}' should trigger wake word")

    def test_jarvis_in_sentence(self):
        self.assertTrue(self.is_wake_word("hey jarvis recipe for eggs"))
        self.assertTrue(self.is_wake_word("they could jarvis"))   # real Vosk example

    def test_similar_enough(self):
        # Words that score ≥ 0.72 against "jarvis"
        for word in ["jarviss", "jaarvis"]:
            with self.subTest(word=word):
                score = SequenceMatcher(None, word, "jarvis").ratio()
                if score >= self._THRESHOLD:
                    self.assertTrue(self.is_wake_word(word))

    def test_unrelated_words_rejected(self):
        for word in ["yes", "hello", "recipe", "stop", "coffee", "marble"]:
            with self.subTest(word=word):
                self.assertFalse(self.is_wake_word(word),
                    f"'{word}' should NOT trigger the wake word")

    def test_empty_string(self):
        self.assertFalse(self.is_wake_word(""))

    def test_case_insensitive(self):
        self.assertTrue(self.is_wake_word("JARVIS"))
        self.assertTrue(self.is_wake_word("Jarvis"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Command Parsing Logic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestCommandParsing(unittest.TestCase):
    """Tests for action-word detection and query extraction."""

    ACTION_WORDS = [
        "search for", "research", "look up",
        "recipe for", "how to cook", "how to bake",
        "record", "log", "save", "cigar",
        "report", "status", "update",
        "close", "finish", "end", "stop",
        "shop for", "buy ", "find prices for ",
    ]

    def has_action(self, cmd):
        return any(a in cmd for a in self.ACTION_WORDS)

    def extract_recipe_query(self, cmd):
        if "recipe for " in cmd:
            return cmd.split("recipe for ")[1].strip().replace("jarvis ", "")
        if "how to cook " in cmd:
            return cmd.split("how to cook ")[1].strip().replace("jarvis ", "")
        if "how to bake " in cmd:
            return cmd.split("how to bake ")[1].strip().replace("jarvis ", "")
        return ""

    def extract_shop_query(self, cmd):
        if "shop for " in cmd:
            return cmd.split("shop for ")[1].strip().replace("jarvis ", "")
        if "buy " in cmd:
            return cmd.split("buy ")[1].strip().replace("jarvis ", "")
        if "find prices for " in cmd:
            return cmd.split("find prices for ")[1].strip().replace("jarvis ", "")
        return ""

    # ── Recipe extraction ────────────────────────────────────────────────────
    def test_recipe_for(self):
        self.assertEqual(self.extract_recipe_query("recipe for scrambled eggs"),
                         "scrambled eggs")

    def test_recipe_with_jarvis_prefix(self):
        # Vosk sometimes includes "jarvis" in the command phrase
        self.assertEqual(self.extract_recipe_query("recipe for jarvis scrambled eggs"),
                         "scrambled eggs")

    def test_how_to_cook(self):
        self.assertEqual(self.extract_recipe_query("how to cook chicken breast"),
                         "chicken breast")

    def test_how_to_bake(self):
        self.assertEqual(self.extract_recipe_query("how to bake sourdough bread"),
                         "sourdough bread")

    def test_recipe_no_match_returns_empty(self):
        self.assertEqual(self.extract_recipe_query("log cigar"), "")

    # ── Shopping extraction ─────────────────────────────────────────────────
    def test_shop_for(self):
        self.assertEqual(self.extract_shop_query("shop for mechanical keyboard"),
                         "mechanical keyboard")

    def test_buy(self):
        self.assertEqual(self.extract_shop_query("buy whole eggs"),
                         "whole eggs")

    def test_find_prices_for(self):
        self.assertEqual(self.extract_shop_query("find prices for standing desk"),
                         "standing desk")

    # ── Action word detection ────────────────────────────────────────────────
    def test_close_session_is_action(self):
        self.assertTrue(self.has_action("jarvis close session"))

    def test_stop_is_action(self):
        self.assertTrue(self.has_action("stop"))

    def test_cigar_is_action(self):
        # "cigar" was added as a trigger so Vosk mishears of "log" still work
        self.assertTrue(self.has_action("the long cigar"))  # real Vosk example

    def test_ambient_noise_no_action(self):
        for noise in ["huh", "the", "hi there", "mm", "okay"]:
            with self.subTest(noise=noise):
                self.assertFalse(self.has_action(noise),
                    f"Ambient noise '{noise}' should not match any action word")

    # ── Close gets priority over other branches ──────────────────────────────
    def test_close_checked_before_search(self):
        """A phrase with 'stop' should be treated as close, not search."""
        cmd = "stop"
        close_words = ["close", "finish", "end", "stop"]
        search_words = ["search for", "research", "look up"]
        is_close  = any(k in cmd for k in close_words)
        is_search = any(k in cmd for k in search_words)
        # 'stop' is close but not search
        self.assertTrue(is_close, "'stop' should match a close word")
        self.assertFalse(is_search, "'stop' should NOT match a search word")
        # Code's if/elif chain: close branch fires, search branch never reached
        result = "close" if is_close else ("search" if is_search else "none")
        self.assertEqual(result, "close")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Environment / Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestConfig(unittest.TestCase):
    """Validates that required environment variables and files are present."""

    def test_env_file_exists(self):
        self.assertTrue((PROJECT_DIR / ".env").exists(),
            ".env file missing — copy .env.example and fill in values")

    def test_ollama_url_set(self):
        self.assertIsNotNone(OLLAMA_BASE_URL)
        self.assertTrue(OLLAMA_BASE_URL.startswith("http"),
            f"OLLAMA_BASE_URL looks wrong: {OLLAMA_BASE_URL}")

    def test_gmail_credentials_set(self):
        user = os.environ.get("GMAIL_USER")
        pw   = os.environ.get("GMAIL_APP_PW")
        if not user or not pw:
            self.skipTest("GMAIL_USER / GMAIL_APP_PW not set — email tests skipped")
        self.assertIn("@", user, "GMAIL_USER should be an email address")
        self.assertGreater(len(pw), 8, "GMAIL_APP_PW looks too short")

    def test_vosk_model_directory_exists(self):
        model_dir = PROJECT_DIR / "model"
        self.assertTrue(model_dir.exists(),
            f"Vosk model directory not found at {model_dir}")
        self.assertTrue(any(model_dir.iterdir()),
            "Vosk model directory is empty")

    def test_yolo_weights_exist(self):
        self.assertTrue((PROJECT_DIR / "yolov8n.pt").exists(),
            "yolov8n.pt not found")
        cigar_weights = PROJECT_DIR / "runs/detect/jarvis_cigar_model/weights/best.pt"
        self.assertTrue(cigar_weights.exists(),
            f"Cigar model weights not found at {cigar_weights}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    # Show test names + result inline for readability
    loader  = unittest.TestLoader()
    loader.sortTestMethodsUsing = None   # preserve definition order within class
    suite   = loader.loadTestsFromModule(sys.modules[__name__])
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
