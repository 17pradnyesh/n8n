# app/config/__init__.py

import os
import json
from typing import Any, Optional

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
NODE_REGISTRY_PATH = os.path.join(CONFIG_DIR, "node_registry.json")
WORKFLOW_SCHEMA_PATH = os.path.join(CONFIG_DIR, "workflow_schema.json")
INDEX_PATH = os.path.join(BASE_DIR, "index.html")

# --- GEMINI API CONFIG ---
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDrA125KgTm_SKp8JmgbzL7Dix276ODieU")

FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "models/gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17",
    "models/gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.5-flash-preview-05-20",
    "models/gemini-2.5-flash-preview-05-20",
]

def load_json_file(path: str) -> Optional[Any]:
    """Load JSON file with error handling"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {path}: {e}")
        return None

# Load configuration files
NODE_REGISTRY = load_json_file(NODE_REGISTRY_PATH) or {}
WORKFLOW_SCHEMA = load_json_file(WORKFLOW_SCHEMA_PATH)

print(f"Loaded environment variables from: {os.path.join(BASE_DIR, '.env')}")