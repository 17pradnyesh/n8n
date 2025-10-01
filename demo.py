from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# FIX: Use conditional imports to find APIError, as its location varies by SDK version.
# We try the most common locations. If none work, we define a dummy class to prevent 
# a hard crash, letting the higher-level exception handler catch the error during the API call.
try:
    from google.generativeai.client import APIError
except ImportError:
    try:
        from google.generativeai import APIError
    except ImportError:
        try:
            from google.generativeai.errors import APIError
        except ImportError:
            class APIError(Exception):
                """Dummy class if the APIError cannot be imported."""
                pass

import json
import re
import os
from typing import Dict, Any, List
from jsonschema import validate as jsonschema_validate, ValidationError
from urllib.parse import urlparse

# --- Configuration & Setup ---

# Load optional config files (registry, schema)
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
NODE_REGISTRY_PATH = os.path.join(CONFIG_DIR, "node_registry.json")
WORKFLOW_SCHEMA_PATH = os.path.join(CONFIG_DIR, "workflow_schema.json")

def load_json_file(path: str) -> Any:
    """Safely loads a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Warning: Could not load optional config file {path}. Error: {e}")
        return None

NODE_REGISTRY = load_json_file(NODE_REGISTRY_PATH) or {}
WORKFLOW_SCHEMA = load_json_file(WORKFLOW_SCHEMA_PATH) or None

app = FastAPI(title="n8n Workflow Generator", version="1.0.0")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini API configuration
def setup_gemini_api():
    """Set up and validate Gemini API configuration."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    return api_key

# Configure Gemini API
model = None

def initialize_gemini():
    """Initialize the Gemini API with error handling."""
    global model
    try:
        # Get API key from environment
        api_key = setup_gemini_api()
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Initialize the model with proper error handling
        try:
            # Use gemini-pro model which is stable and widely available
            model = genai.GenerativeModel('gemini-flash')
            
            # Verify the model works with a simple test
            test_response = model.generate_content("Test connection")
            if not test_response or not test_response.text:
                raise ValueError("Model returned empty response")
                
            print("✅ Gemini API initialized successfully")
            return True
            
        except Exception as model_error:
            print(f"❌ Error creating model: {str(model_error)}")
            raise
            
    except Exception as e:
        print(f"❌ Error configuring Gemini API: {str(e)}")
        return False

# Initialize the model during startup
print("Initializing Gemini API...")
if not initialize_gemini():
    print("Warning: Gemini API initialization failed. Please check your API key and internet connection.")

class WorkflowRequest(BaseModel):
    instructions: str

# --- Prompt Generation Logic ---

def create_n8n_prompt(instructions: str, extracted_params: Dict[str, Any] = None) -> str:
    """Create a highly-constrained prompt for workflow generation."""
    return f"""You are an expert n8n workflow architect. Create a sophisticated n8n workflow for: \"{instructions}\"

**CRITICAL OUTPUT REQUIREMENTS:**
- Output MUST be a single, valid JSON object only
- NO markdown formatting or code blocks
- NO explanations or text before/after JSON
- JSON must be production-ready for n8n import

**MANDATORY CONNECTION RULES:**
1. EVERY node (except the last node) MUST have outgoing connections
2. EVERY node (except triggers) MUST have incoming connections
3. Node names in connections MUST exactly match node names in the nodes array
4. Connection structure MUST follow the exact pattern provided in the examples.

**VALIDATION CHECKLIST:**
✅ Every node has `id`, `type`, `name`, `position`
✅ Connections are complete and consistent
✅ JSON is valid and importable into n8n
"""

# --- Parameter Extraction and Validation Logic ---

def extract_parameters_from_instructions(instructions: str) -> Dict[str, Any]:
    """Extract API keys, URLs, query parameters, and specific values from user instructions."""
    params = {
        'api_keys': {},
        'urls': {},
        'query_params': {},
        'values': {}
    }
    
    # 1. Extract API keys (common formats like "key = XXX" or "api_key: XXX")
    api_key_patterns = [
        r'(?:api[_-]?key|key|token|client[_-]?id|client[_-]?secret)[\s:=]+["\']?([\w-]+)["\']?',
        r'key[\s:=]+["\']?(AIza[\w-]+)["\']?'  # Google API key format
    ]
    for pattern in api_key_patterns:
        matches = re.finditer(pattern, instructions, re.IGNORECASE)
        for match in matches:
            key_name_or_value = match.group(0).split(match.group(1))[0].strip().split()[-1].strip(':=').strip('\'"')
            key_value = match.group(1)
            
            service_name = "google" if key_value.startswith("AIza") else key_name_or_value or "default_key"
            
            if len(key_value) > 8:
                 params['api_keys'][service_name] = key_value

    # 2. Extract URLs (http/https)
    url_pattern = r'(?:url|endpoint|api)[\s:=]+["\']?(https?://[^\s"\'>]+)["\']?'
    matches = re.finditer(url_pattern, instructions, re.IGNORECASE)
    for match in matches:
        url = match.group(1).strip()
        
        try:
            domain = urlparse(url).netloc.replace("www.", "")
            if domain and url not in params['urls'].values():
                params['urls'][domain] = url
        except Exception:
            pass # Skip invalid URL

    # 3. Extract query parameters
    query_param_pattern = r'(?:param|parameter|query)[\s:=]+(\w+)[\s:=]+["\']?([^"\'\s]+)["\']?'
    matches = re.finditer(query_param_pattern, instructions, re.IGNORECASE)
    for match in matches:
        param_name = match.group(1).strip()
        param_value = match.group(2).strip()
        if param_name.lower() not in ('key', 'api_key', 'apikey') and param_value not in params['api_keys'].values():
            params['query_params'][param_name] = param_value
    
    # 4. Extract specific values
    EXCLUDE_KEYWORDS = {'key', 'api_key', 'url', 'param', 'parameter', 'query', 'node', 'type', 'id', 'name', 'position', 'method'}
    value_pattern = r'(\w+)[\s:=]+["\']?([^"\'\s]+)["\']?'
    matches = re.finditer(value_pattern, instructions, re.IGNORECASE)
    for match in matches:
        key = match.group(1).strip()
        value = match.group(2).strip()
        
        if key.lower() not in EXCLUDE_KEYWORDS and \
           key.lower() not in params['query_params'] and \
           value not in params['api_keys'].values() and \
           not value.startswith(('http://', 'https://')):
            
            params['values'][key.lower()] = value

    return params

def validate_parameters(params: Dict[str, Any]) -> None:
    """Validate extracted parameters to ensure they are properly formatted."""
    
    # Validate API keys
    for service, key in params.get('api_keys', {}).items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Invalid API key for service {service}")
        if service == 'google' and not key.startswith('AIza'):
            raise ValueError("Invalid Google API key format (must start with AIza)")

    # Validate URLs
    for domain, url in params.get('urls', {}).items():
        if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL for domain {domain}: {url}")
        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError(f"URL format incomplete: {url}")
        except Exception as e:
            raise ValueError(f"URL validation failed for {url}: {str(e)}")


def validate_n8n_workflow(workflow: Dict[Any, Any]) -> None:
    """Basic validation of n8n workflow structure."""
    required_fields = ["name", "nodes", "connections"]
    for field in required_fields:
        if field not in workflow:
            raise ValueError(f"Missing required field: {field}")

    if not isinstance(workflow["nodes"], list) or len(workflow["nodes"]) == 0:
        raise ValueError("Workflow must contain a non-empty 'nodes' array.")

    # Validate each node has required fields
    for node in workflow["nodes"]:
        if not all(key in node for key in ["id", "type", "name", "position"]):
            raise ValueError(f"Node {node.get('name', 'unknown')} is missing required fields.")
        if not isinstance(node["position"], list) or len(node["position"]) != 2:
            raise ValueError(f"Node {node['name']} has an invalid position.")

def extract_json_from_response(response: str) -> Dict[Any, Any]:
    """Extract and validate JSON from Gemini response (with error handling)"""
    text = response if isinstance(response, str) else str(response)

    try:
        # 1. Prefer fenced ```json blocks
        json_fence = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if json_fence:
            json_str = json_fence.group(1)
        else:
            # 2. Accept any fenced block if language tag missing
            any_fence = re.search(r"```\s*([\s\S]*?)\s*```", text)
            if any_fence:
                json_str = any_fence.group(1)
            else:
                # 3. Fallback: take substring from first '{' to last '}'
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_str = text[start:end + 1]
                else:
                    json_str = text.strip()

        # Trim BOM/whitespace
        json_str = json_str.strip('\ufeff').strip()

        workflow_json = json.loads(json_str)
        validate_n8n_workflow(workflow_json)
        return workflow_json
    except json.JSONDecodeError as e:
        # Error 1: Failed to parse JSON
        raise ValueError(f"Invalid JSON format returned by AI: {str(e)}. Raw output did not contain valid JSON.")
    except Exception as e:
        # Error 2: Failed workflow structure validation
        raise ValueError(f"Workflow structure validation failed: {str(e)}. Raw response: {response}")


# --- Post-Processing / Auto-Healing Logic ---

def apply_native_node_preferences(workflow: Dict[str, Any], instructions: str) -> Dict[str, Any]:
    """Prefer native n8n nodes over generic HTTP for well-known services."""
    text = instructions.lower()
    nodes = workflow.get("nodes", [])
    keyword_to_node = [
        ("google sheets", {"type": "n8n-nodes-base.googleSheets", "name": "Google Sheets"}),
        ("http", {"type": "n8n-nodes-base.httpRequest", "name": "HTTP Request"}),
    ]
    preferred = next((target for keyword, target in keyword_to_node if keyword in text), None)

    if preferred:
        for node in nodes:
            node_type = str(node.get("type", ""))
            node_name = str(node.get("name", ""))
            is_generic_http = "httpRequest" in node_type or node_name.lower() in ("http request", "api call", "fetch data")
            is_trigger = "trigger" in node_type.lower()
            preferred_is_trigger = "trigger" in preferred["type"].lower()

            if preferred_is_trigger and is_trigger:
                node["type"] = preferred["type"]
                node["name"] = preferred["name"]
            elif (not preferred_is_trigger) and is_generic_http and not is_trigger:
                node["type"] = preferred["type"]
                if node_name.lower() in ("http request", "api call", "fetch data"):
                    node["name"] = preferred["name"]
    return workflow

def needs_ai_processing(text: str) -> bool:
    """Detect if the instruction implies AI usage (summarize, generate, classify, Gemini)."""
    t = text.lower()
    keywords = [
        "summarize", "summary", "summarisation", "ai ", " llm", "gemini",
        "classify", "categorize", "extract", "rewrite", "generate", "analyze"
    ]
    return any(k in t for k in keywords)

def auto_heal_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Light auto-heal: ensure ids, positions, and minimal connection wiring exist."""
    nodes = workflow.get("nodes", [])
    connections = workflow.get("connections", {})

    # Ensure workflow has a name
    workflow.setdefault("name", "Generated Workflow")

    for idx, node in enumerate(nodes):
        # Ensure every node has a unique ID
        node.setdefault("id", f"node-{idx+1}")

        # Ensure every node has a position
        node.setdefault("position", [250 + idx * 250, 300])

    # Add missing connections
    for i in range(len(nodes) - 1):
        current_node = nodes[i]
        next_node = nodes[i + 1]

        connections.setdefault(current_node["name"], {}).setdefault("main", [[]])
        if not any(conn.get("node") == next_node["name"] for conn in connections[current_node["name"]]["main"][0]):
            connections[current_node["name"]]["main"][0].append({"node": next_node["name"], "type": "main", "index": 0})

    workflow["nodes"] = nodes
    workflow["connections"] = connections
    return workflow

# NOTE: The full logic for these functions must be available in the execution environment
# as they were part of the previous code context. Simplified placeholders are used here.
def ensure_ai_node_present(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """If no AI node exists, inject an AI Model node and wire it into the main path."""
    return workflow 

def ensure_ai_agent_present(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """If no AI Agent exists, inject Google Gemini Chat Model + AI Agent and connect them."""
    return workflow

def apply_extracted_parameters(workflow: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Applies extracted parameters to the workflow nodes, enforcing manual values."""
    nodes = workflow.get("nodes", [])
    
    for node in nodes:
        node_type = node.get("type", "")
        parameters = node.get("parameters", {})
        
        # 1. Apply URLs to HTTP Request nodes
        if params.get("urls") and "httpRequest" in node_type:
            for url in params["urls"].values():
                parameters["url"] = url
                break
        
        # 2. Apply API keys and Query Parameters to HTTP Request nodes
        if "httpRequest" in node_type:
            parameters.setdefault("queryParameters", {}).setdefault("parameters", [])
            query_params: List[Dict[str, Any]] = parameters["queryParameters"]["parameters"]

            # Apply API keys
            if params.get("api_keys"):
                for name, key in params["api_keys"].items():
                    common_key_names = ["key", "api_key", "apikey", name.lower()]
                    param_found = False
                    for param in query_params:
                        if param.get("name", "").lower() in common_key_names:
                            param["value"] = key
                            param_found = True
                            break
                    if not param_found:
                        key_name = name if name not in ("google", "default_key") else "key"
                        query_params.append({"name": key_name, "value": key, "type": "string"})

            # Apply explicit Query parameters
            if params.get("query_params"):
                for name, value in params["query_params"].items():
                    param_found = False
                    for param in query_params:
                        if param.get("name", "").lower() == name.lower():
                            param["value"] = value
                            param_found = True
                            break
                    if not param_found:
                        query_params.append({"name": name, "value": value, "type": "string"})
            
            parameters["queryParameters"]["parameters"] = query_params

        # 3. Apply Specific values to Set nodes, Email nodes, etc.
        if params.get("values"):
            if "set" in node_type.lower():
                parameters.setdefault("assignments", {}).setdefault("assignments", [])
                assignments: List[Dict[str, Any]] = parameters["assignments"]["assignments"]
                
                for key, value in params["values"].items():
                    assignment_found = False
                    for assignment in assignments:
                        if assignment.get("name", "").lower() == key.lower():
                            assignment["value"] = value
                            assignment_found = True
                            break
                    if not assignment_found:
                        assignments.append({
                            "id": f"{key}-{len(assignments)+1}",
                            "name": key,
                            "value": value,
                            "type": "string"
                        })
                parameters["assignments"]["assignments"] = assignments

            elif "emailSend" in node_type or "slack" in node_type:
                for key, value in params["values"].items():
                    if key.lower() in ("toemail", "recipient", "receiver"):
                        parameters["toEmail"] = value
                    elif key.lower() in ("fromemail", "sender"):
                        parameters["fromEmail"] = value
                    elif key.lower() in ("subject", "title"):
                        parameters["subject"] = value
                    elif key.lower() in ("channel", "slackchannel"):
                        parameters["channel"] = value

        node["parameters"] = parameters
    
    workflow["nodes"] = nodes
    return workflow

# --- API Endpoint with comprehensive Error Handling ---

@app.post("/generate-workflow", response_model=Dict[str, Any])
async def generate_workflow(request: WorkflowRequest):
    """Generate an n8n workflow based on user instructions."""
    global model
    
    # Ensure model is initialized
    if model is None:
        print("Model not initialized, attempting to initialize...")
        if not initialize_gemini():
            print("Failed to initialize Gemini API")
            raise HTTPException(
                status_code=500,
                detail="Unable to initialize AI service. Please try again in a few moments."
            )
        print("Successfully initialized Gemini API")

    instructions = request.instructions.strip()
    if not instructions:
        raise HTTPException(status_code=400, detail="Instructions cannot be empty")

    try:
        # Extract and validate parameters
        extracted_params = extract_parameters_from_instructions(instructions)
        
        # Create prompt with extracted parameters
        prompt = create_n8n_prompt(instructions, extracted_params)
        
        # Generate workflow using Gemini
        try:
            generation = model.generate_content(prompt)
            if not generation or not generation.text:
                raise ValueError("Empty response from AI model")
            workflow_text = generation.text
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error communicating with AI service: {str(e)}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during parameter extraction: {str(e)}")


    # 2. Create prompt for Gemini
    prompt = create_n8n_prompt(instructions, extracted_params)

    # 3. Call Gemini API (try/except for external service errors)
    try:
        if not model:
            # Model config failed earlier
            raise APIError("Gemini model object failed to initialize. Check API key.")

        response = model.generate_content(prompt)
        response_text = getattr(response, "text", None) or str(response)
        
        if not response_text:
            raise ValueError("AI returned an empty response.")

    except APIError as e:
        # Catch specific API errors (e.g., authentication, invalid request, rate limits)
        print(f"Gemini API Error: {e}")
        raise HTTPException(status_code=502, detail=f"Gemini API call failed. Check API key and service status. Error: {e}")
    except Exception as e:
        # Catch other exceptions (e.g., network issues)
        print(f"General AI Call Error: {e}")
        raise HTTPException(status_code=502, detail=f"Error communicating with AI service: {str(e)}")

    # 4. Extract and validate JSON (try/except for malformed output)
    try:
        workflow_json = extract_json_from_response(response_text)
    except ValueError as e:
        # Raised by extract_json_from_response on JSONDecodeError or validation errors
        print(f"JSON/Validation Error: {e}")
        raise HTTPException(status_code=400, detail=f"AI returned invalid workflow data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during workflow parsing: {str(e)}")


    # 5. Post-processing and Auto-healing (try/except for internal processing errors)
    try:
        workflow_json = apply_native_node_preferences(workflow_json, instructions)
        workflow_json = auto_heal_workflow(workflow_json)

        if needs_ai_processing(instructions):
            workflow_json = ensure_ai_node_present(workflow_json)
            workflow_json = ensure_ai_agent_present(workflow_json) 

        # 6. Apply extracted parameters to the workflow
        workflow_json = apply_extracted_parameters(workflow_json, extracted_params)
    except Exception as e:
        print(f"Post-processing Error: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred while finalizing the workflow structure: {str(e)}")

    return workflow_json

# Serve OpenAPI schema as JSON
@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_json():
    return JSONResponse(app.openapi())

# Serve index.html
@app.get("/")
async def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(index_path)