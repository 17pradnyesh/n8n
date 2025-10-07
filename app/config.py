# app/config.py

import os
from dotenv import load_dotenv

# Load .env file from package root (n8n/.env) so environment variables are available
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
print(f"Loaded environment variables from: {os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')}")
import json
from typing import Any, Optional

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
NODE_REGISTRY_PATH = os.path.join(CONFIG_DIR, "node_registry.json")
WORKFLOW_SCHEMA_PATH = os.path.join(CONFIG_DIR, "workflow_schema.json")
INDEX_PATH = os.path.join(BASE_DIR, "index.html")

# --- GEMINI API CONFIG ---
GEMINI_MODEL_NAME = "gemini-2.5-flash"
# Make sure to create a .env file with your GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

FALLBACK_MODELS = [
    "gemini-1.5-flash",
    "models/gemini-1.5-flash",
]

# --- HELPER TO LOAD JSON FILES ---
def load_json_file(path: str) -> Optional[Any]:
    """Load JSON file with error handling"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {path}: {e}")
        return None

# --- LOADED SCHEMAS ---
NODE_REGISTRY = load_json_file(NODE_REGISTRY_PATH) or {}
WORKFLOW_SCHEMA = load_json_file(WORKFLOW_SCHEMA_PATH)