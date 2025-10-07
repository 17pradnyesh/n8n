# n8n Workflow Generator API

AI-powered n8n workflow generator that uses Google Gemini (Generative AI) to create fully-configured, production-ready n8n workflows from plain-English instructions.

This repository contains a small FastAPI application (`n8n/main.py`) that:
- Accepts natural-language instructions and extracts useful parameters (emails, URLs, cron times, API keys, sheet names, etc.)
- Builds a detailed prompt and calls the Gemini generative model to produce a complete n8n workflow JSON
- Validates, auto-configures, and "auto-heals" the returned workflow JSON so it is ready to import into n8n

## Getting started

These instructions assume you are on Windows and there is a local Python virtual environment under `env/` (this repo already includes a virtualenv under `env/`). The project uses FastAPI and the Google generative AI client.

Prerequisites
- Python 3.11+
- (Optional) A Google Cloud Generative AI API key (export to `GEMINI_API_KEY` environment variable)

Install dependencies (if you need to recreate the virtualenv):

```powershell
# from repository root
python -m venv env
env\Scripts\Activate.ps1
pip install -r n8n/requirements.txt
```

Environment
- GEMINI_API_KEY: API key used by the `google.generativeai` client. If not set, the code tries a default placeholder but the model initialization will likely fail.

Configuration files
- `n8n/config/workflow_schema.json`: Optional JSON Schema used to validate generated workflows. If present, the application will attempt to validate model output against it.
- `n8n/config/node_registry.json`: Loaded at startup but not currently used elsewhere in the code. It can be used in the future to map or validate node types.

## Run locally

You can run the API with Uvicorn:

```powershell
# Activate virtualenv first
env\Scripts\Activate.ps1
python -m n8n.main
# or directly with uvicorn
env\Scripts\Activate.ps1
uvicorn n8n.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/` to view the `index.html` (if present) or `http://localhost:8000/docs` for the automatic Swagger UI.

## Endpoints

- POST /generate-workflow
  - Request body: `{ "instructions": "<natural language instructions>" }`
  - Response: JSON object representing an n8n workflow ready to import.
  - Behavior: extracts parameters, constructs a prompt, calls Gemini to generate JSON, validates and auto-configures nodes, returns workflow JSON.

- GET /health
  - Quick health check returning `status`, `gemini_initialized`, and `model` name.

- GET /models
  - Lists available Gemini models (calls `genai.list_models()` internally).

- GET /
  - Serves `index.html` in the package if present, otherwise a small JSON welcome message.

- GET /openapi.json
  - Returns the OpenAPI spec for this FastAPI app.

## Important modules and functions (quick reference)
- `n8n/main.py`
  - `initialize_gemini()` — configures the `google.generativeai` client and picks a working model.
  - `extract_parameters_from_instructions()` — heuristic extraction for emails, URLs, schedule times, API keys, Slack channels, sheet names.
  - `create_enhanced_prompt()` — builds the long prompt sent to Gemini to request a fully-configured workflow JSON.
  - `extract_json_from_response()` — extracts JSON (tries code blocks or first/last braces), parses it, and validates it.
  - `validate_n8n_workflow()` — minimal structural validation plus optional JSON Schema validation using `config/workflow_schema.json`.
  - `smart_configure_nodes()` — post-processes nodes to fill missing parameters like emails, http urls, cron expressions, Slack channels, and Google Sheets fields.
  - `auto_heal_workflow()` — ensures nodes have `id`, `name`, `position`, and populates basic `connections` when missing.

## Notes & recommendations
- The generator relies on the model following the instruction to output JSON only. The `extract_json_from_response()` function attempts to be robust, but model outputs may still fail parsing.
- `workflow_schema.json` is useful to enforce stricter validation. Currently a schema validation error only prints a warning; consider making schema failures fatal if you need stricter enforcement.
- `node_registry.json` is loaded but not used; if you don't plan to use it, you may remove it. Alternatively, add validation or mapping logic that references it.
- Add tests for `extract_parameters_from_instructions`, `smart_configure_nodes`, and `extract_json_from_response` to guard against regressions.

## Contributing
- Open an issue or PR to improve parameter extraction, prompt design, or schema validation.

## License
- No license file included. Add one (e.g., MIT) if you intend to make this public.

---

If you want, I can:
- Open the `config/` files and include a brief summary of their current contents in the README.
- Make schema validation strict (return 400 on schema violations).
- Add a tiny test harness for the main helper functions.

Which of those would you like next?