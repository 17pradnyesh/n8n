from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
import json
import re
import os
from typing import Dict, Any, Optional
from jsonschema import validate as jsonschema_validate, ValidationError

# ==================== CONFIGURATION ====================

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
NODE_REGISTRY_PATH = os.path.join(CONFIG_DIR, "node_registry.json")
WORKFLOW_SCHEMA_PATH = os.path.join(CONFIG_DIR, "workflow_schema.json")

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

# ==================== HELPER FUNCTIONS ====================

def load_json_file(path: str) -> Optional[Any]:
    """Load JSON file with error handling"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {path}: {e}")
        return None

NODE_REGISTRY = load_json_file(NODE_REGISTRY_PATH) or {}
WORKFLOW_SCHEMA = load_json_file(WORKFLOW_SCHEMA_PATH)

# ==================== FASTAPI APP SETUP ====================

app = FastAPI(
    title="n8n Workflow Generator",
    version="2.0.0",
    description="AI-powered n8n workflow generator with smart auto-configuration"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GEMINI CONFIGURATION ====================

model = None

def initialize_gemini() -> bool:
    """Initialize Gemini API with proper error handling and fallback models"""
    global model, GEMINI_MODEL_NAME
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        try:
            available_models = genai.list_models()
            model_names = [m.name for m in available_models if 'generateContent' in m.supported_generation_methods]
            print(f"📋 Available models: {[m.replace('models/', '') for m in model_names[:8]]}")
            
            preferred_models = [
                m for m in model_names 
                if 'flash' in m.lower() 
                and 'preview' not in m.lower() 
                and 'lite' not in m.lower()
            ]
            if preferred_models:
                GEMINI_MODEL_NAME = preferred_models[0].replace('models/', '')
                print(f"🎯 Selected preferred model: {GEMINI_MODEL_NAME}")
        except Exception as list_error:
            print(f"⚠️  Could not list models: {list_error}")
        
        models_to_try = list(dict.fromkeys([GEMINI_MODEL_NAME] + FALLBACK_MODELS))
        
        for model_name in models_to_try:
            try:
                test_model = genai.GenerativeModel(model_name)
                test_response = test_model.generate_content("Hi")
                if test_response and test_response.text:
                    model = test_model
                    GEMINI_MODEL_NAME = model_name
                    print(f"✅ Gemini API initialized successfully with: {model_name}")
                    return True
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    print(f"⚠️  Model {model_name}: Quota exceeded, trying next...")
                elif "404" in error_msg:
                    print(f"⚠️  Model {model_name}: Not found, trying next...")
                else:
                    print(f"⚠️  Model {model_name}: {error_msg[:80]}")
                continue
        
        print(f"❌ All models failed to initialize")
        print(f"💡 Tip: Check your API quota at https://aistudio.google.com/")
        return False
        
    except Exception as e:
        print(f"❌ Failed to configure Gemini API: {str(e)}")
        return False

print("=" * 60)
print("Initializing n8n Workflow Generator API...")
if not initialize_gemini():
    print("⚠️  Warning: Gemini API initialization failed")
print("=" * 60)

# ==================== REQUEST MODELS ====================

class WorkflowRequest(BaseModel):
    instructions: str

# ==================== PARAMETER EXTRACTION ====================

def extract_parameters_from_instructions(instructions: str) -> Dict[str, Any]:
    """Extract specific parameters mentioned in user instructions"""
    params = {}
    text = instructions.lower()
    
    # Extract emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, instructions, re.IGNORECASE)
    if emails:
        params['emails'] = emails
        params['from_email'] = emails[0] if len(emails) > 0 else None
        params['to_email'] = emails[-1] if len(emails) > 1 else emails[0]
    
    # Extract URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, instructions)
    if urls:
        params['urls'] = urls
        params['api_url'] = urls[0]
    
    # Extract time/schedule patterns
    time_pattern = r'(\d{1,2})\s*(?:am|pm|AM|PM|:00)?'
    times = re.findall(time_pattern, text)
    if times and any(word in text for word in ['daily', 'every day', 'each day', 'at']):
        hour = int(times[0])
        if 'pm' in text and hour < 12:
            hour += 12
        elif 'am' in text and hour == 12:
            hour = 0
        params['schedule_hour'] = hour
        params['cron_expression'] = f"0 {hour} * * *"
    
    # Extract API keys (look for patterns like "api key: xxx" or "key=xxx")
    api_key_patterns = [
        r'api[_\s-]?key[:\s=]+([A-Za-z0-9_-]{20,})',
        r'token[:\s=]+([A-Za-z0-9_-]{20,})',
        r'key[:\s=]+([A-Za-z0-9_-]{20,})'
    ]
    for pattern in api_key_patterns:
        matches = re.findall(pattern, instructions, re.IGNORECASE)
        if matches:
            params['api_key'] = matches[0]
            break
    
    # Extract database connection strings
    if 'mongodb://' in instructions:
        mongo_match = re.search(r'(mongodb://[^\s]+)', instructions)
        if mongo_match:
            params['mongodb_connection'] = mongo_match.group(1)
    
    # Extract Slack channel
    slack_pattern = r'#([a-z0-9-]+)|channel[:\s]+([a-z0-9-]+)'
    slack_matches = re.findall(slack_pattern, text)
    if slack_matches:
        params['slack_channel'] = next((m for group in slack_matches for m in group if m), None)
    
    # Extract spreadsheet/sheet names
    sheet_pattern = r'(?:spreadsheet|sheet)[:\s]+["\']?([^"\']+)["\']?'
    sheet_match = re.search(sheet_pattern, text, re.IGNORECASE)
    if sheet_match:
        params['sheet_name'] = sheet_match.group(1).strip()
    
    return params

# ==================== PROMPT GENERATION ====================

def create_enhanced_prompt(instructions: str, extracted_params: Dict[str, Any]) -> str:
    """Generate enhanced prompt with extracted parameters"""
    
    param_context = ""
    if extracted_params:
        param_context = f"\n**EXTRACTED PARAMETERS TO USE:**\n"
        for key, value in extracted_params.items():
            param_context += f"- {key}: {value}\n"
    
    return f"""You are an expert n8n workflow architect. Create a COMPLETE, PRODUCTION-READY workflow that requires ZERO manual configuration.

**USER REQUEST:** "{instructions}"
{param_context}

**CRITICAL REQUIREMENTS:**
1. Output ONLY valid JSON (no markdown, no explanations)
2. ALL parameters must be FULLY CONFIGURED with working values
3. Use extracted parameters from user input when available
4. Use smart defaults for missing parameters
5. Every node must be executable without manual configuration

**PARAMETER CONFIGURATION RULES:**

**Email Nodes (n8n-nodes-base.emailSend):**
- MUST include: fromEmail, toEmail, subject, text/html
- Use extracted emails if provided, otherwise use placeholders like "workflow@n8n.local" and "user@example.com"
- Subject should be descriptive based on workflow purpose
- Text should use n8n expressions like "={{$json.data}}" to include dynamic data

**HTTP Request Nodes (n8n-nodes-base.httpRequest):**
- MUST include: url, method, authentication (if API key provided)
- If API key extracted, add authentication: {{"type": "genericCredentialType", "credentials": {{"apiKey": "EXTRACTED_KEY"}}}}
- Use realistic API endpoints based on context
- Add headers if needed: {{"Content-Type": "application/json"}}

**Schedule Trigger (n8n-nodes-base.scheduleTrigger):**
- MUST include fully configured cron expression
- Use extracted time if provided (e.g., "0 8 * * *" for 8 AM daily)
- Default to "0 9 * * *" (9 AM daily) if not specified

**Webhook (n8n-nodes-base.webhook):**
- MUST include: httpMethod (POST/GET), path
- Set path to descriptive name like "/process-data" or "/webhook-handler"
- Set responseMode to "responseNode" for proper flow

**Google Sheets (n8n-nodes-base.googleSheets):**
- MUST include: operation, sheetId, range
- Use extracted sheet name if provided
- Default operation based on context (append for saving, read for fetching)
- Use range "A:Z" for full sheet access

**Slack (n8n-nodes-base.slack):**
- MUST include: resource, operation, channel, text
- Use extracted channel if provided (add # prefix)
- Default to "#general" if not specified
- Text should be dynamic: "={{$json.message}}"

**Code Nodes (n8n-nodes-base.code):**
- MUST include complete working JavaScript
- Handle data transformation properly
- Return properly formatted data

**Set Nodes (n8n-nodes-base.set):**
- Configure actual field mappings
- Use descriptive field names
- Include type conversions if needed

**COMPLETE EXAMPLE WITH FULL PARAMETERS:**
{{
  "name": "Daily Weather Email Report",
  "nodes": [
    {{
      "parameters": {{
        "rule": {{
          "interval": [{{
            "field": "cronExpression",
            "cronExpression": "0 8 * * *"
          }}]
        }}
      }},
      "name": "Daily at 8 AM",
      "type": "n8n-nodes-base.scheduleTrigger",
      "typeVersion": 1.1,
      "id": "trigger-1",
      "position": [250, 300]
    }},
    {{
      "parameters": {{
        "url": "https://api.openweathermap.org/data/2.5/weather",
        "method": "GET",
        "sendQuery": true,
        "queryParameters": {{
          "parameters": [
            {{"name": "q", "value": "London"}},
            {{"name": "appid", "value": "YOUR_API_KEY"}},
            {{"name": "units", "value": "metric"}}
          ]
        }},
        "options": {{}}
      }},
      "name": "Fetch Weather Data",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "id": "http-1",
      "position": [500, 300]
    }},
    {{
      "parameters": {{
        "jsCode": "const data = $input.first().json;\\nconst temp = data.main.temp;\\nconst description = data.weather[0].description;\\nconst city = data.name;\\n\\nreturn {{\\n  summary: `Weather in ${{city}}: ${{temp}}°C, ${{description}}`,\\n  temperature: temp,\\n  condition: description\\n}};"
      }},
      "name": "Format Weather Data",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "id": "code-1",
      "position": [750, 300]
    }},
    {{
      "parameters": {{
        "fromEmail": "weather-bot@n8n.local",
        "toEmail": "user@example.com",
        "subject": "Daily Weather Report - ={{$json.condition}}",
        "text": "={{$json.summary}}",
        "options": {{}}
      }},
      "name": "Send Weather Email",
      "type": "n8n-nodes-base.emailSend",
      "typeVersion": 2.1,
      "id": "email-1",
      "position": [1000, 300]
    }}
  ],
  "connections": {{
    "Daily at 8 AM": {{
      "main": [[{{"node": "Fetch Weather Data", "type": "main", "index": 0}}]]
    }},
    "Fetch Weather Data": {{
      "main": [[{{"node": "Format Weather Data", "type": "main", "index": 0}}]]
    }},
    "Format Weather Data": {{
      "main": [[{{"node": "Send Weather Email", "type": "main", "index": 0}}]]
    }}
  }},
  "active": false,
  "settings": {{}},
  "versionId": "1"
}}

**SMART DEFAULTS TO USE:**

Emails:
- From: "workflow@n8n.local" or extracted email
- To: "user@example.com" or extracted email
- Subject: Descriptive based on workflow purpose
- Body: Dynamic using n8n expressions

API URLs:
- Use realistic public APIs (weather: openweathermap.org, news: newsapi.org, etc.)
- Include required parameters in queryParameters
- Add authentication if API key provided

Times:
- Default to 9 AM daily if not specified: "0 9 * * *"
- Use extracted time if provided

Channels:
- Slack: "#general" or extracted channel
- Discord: "general" or extracted channel

Sheets:
- Range: "A:Z" for full access
- Operation: "append" for saving, "read" for fetching

**REMEMBER:**
- Every parameter MUST be filled with actual values
- No placeholder text like "YOUR_API_KEY" unless no key was provided
- Use extracted parameters when available
- Make workflow executable without any manual changes
- Include proper error handling where needed

Generate the complete, fully-configured workflow JSON now:"""

# ==================== JSON EXTRACTION ====================

def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """Extract and parse JSON from Gemini response"""
    try:
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = response_text[start:end + 1]
            else:
                json_str = response_text
        
        json_str = json_str.strip().strip('\ufeff')
        workflow = json.loads(json_str)
        validate_n8n_workflow(workflow)
        return workflow
        
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON: {str(e)}\n\nResponse preview: {response_text[:500]}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Workflow validation failed: {str(e)}"
        )

# ==================== WORKFLOW VALIDATION ====================

def validate_n8n_workflow(workflow: Dict[str, Any]) -> None:
    """Validate n8n workflow structure"""
    if WORKFLOW_SCHEMA:
        try:
            jsonschema_validate(instance=workflow, schema=WORKFLOW_SCHEMA)
            return
        except ValidationError as e:
            print(f"Schema validation warning: {e}")
    
    required = ['name', 'nodes', 'connections']
    for field in required:
        if field not in workflow:
            raise ValueError(f"Missing required field: {field}")
    
    if not isinstance(workflow['nodes'], list) or not workflow['nodes']:
        raise ValueError("Workflow must have at least one node")
    
    for i, node in enumerate(workflow['nodes']):
        required_node_fields = ['id', 'name', 'type', 'position']
        for field in required_node_fields:
            if field not in node:
                raise ValueError(f"Node {i} missing field: {field}")

# ==================== WORKFLOW ENHANCEMENT ====================

def smart_configure_nodes(workflow: Dict[str, Any], extracted_params: Dict[str, Any]) -> Dict[str, Any]:
    """Intelligently configure node parameters based on extracted data"""
    nodes = workflow.get("nodes", [])
    
    for node in nodes:
        node_type = node.get("type", "")
        params = node.get("parameters", {})
        
        # Configure Email nodes
        if "emailSend" in node_type:
            if not params.get("fromEmail"):
                params["fromEmail"] = extracted_params.get("from_email", "workflow@n8n.local")
            if not params.get("toEmail"):
                params["toEmail"] = extracted_params.get("to_email", "user@example.com")
            if not params.get("subject"):
                params["subject"] = "Workflow Result"
            if not params.get("text") and not params.get("html"):
                params["text"] = "={{$json}}"
        
        # Configure HTTP Request nodes
        elif "httpRequest" in node_type:
            if not params.get("url") and extracted_params.get("api_url"):
                params["url"] = extracted_params["api_url"]
            if not params.get("method"):
                params["method"] = "GET"
            
            # Ensure sendQuery is true if queryParameters exist
            if params.get("queryParameters"):
                params["sendQuery"] = True
            
            # Add API key authentication if provided
            if extracted_params.get("api_key") and not params.get("authentication"):
                params["authentication"] = "genericCredentialType"
                params.setdefault("genericAuthType", "httpHeaderAuth")
                params.setdefault("httpHeaderAuth", {
                    "name": "Authorization",
                    "value": f"Bearer {extracted_params['api_key']}"
                })
        
        # Configure Schedule triggers
        elif "scheduleTrigger" in node_type:
            if not params.get("rule"):
                cron = extracted_params.get("cron_expression", "0 9 * * *")
                params["rule"] = {
                    "interval": [{
                        "field": "cronExpression",
                        "cronExpression": cron
                    }]
                }
        
        # Configure Webhook nodes
        elif "webhook" in node_type:
            if not params.get("path"):
                params["path"] = "webhook-handler"
            if not params.get("httpMethod"):
                params["httpMethod"] = "POST"
            if not params.get("responseMode"):
                params["responseMode"] = "responseNode"
        
        # Configure Slack nodes
        elif "slack" in node_type:
            if not params.get("channel"):
                channel = extracted_params.get("slack_channel", "general")
                params["channel"] = f"#{channel}" if not channel.startswith("#") else channel
            if not params.get("text"):
                params["text"] = "={{$json.message || $json}}"
        
        # Configure Google Sheets nodes
        elif "googleSheets" in node_type:
            if not params.get("operation"):
                params["operation"] = "append"
            if not params.get("range"):
                params["range"] = "A:Z"
            if extracted_params.get("sheet_name") and not params.get("sheetName"):
                params["sheetName"] = extracted_params["sheet_name"]
        
        # Configure Set nodes with proper values structure
        elif "set" in node_type.lower():
            # Ensure values are in the correct structure
            if params.get("values") and isinstance(params["values"], dict):
                if not params["values"].get("string"):
                    # Convert to proper structure if needed
                    pass
        
        node["parameters"] = params
    
    workflow["nodes"] = nodes
    return workflow

def auto_heal_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure workflow has proper structure and connections"""
    nodes = workflow.get("nodes", [])
    connections = workflow.get("connections", {})
    
    for idx, node in enumerate(nodes):
        node.setdefault("id", f"node-{idx + 1}")
        node.setdefault("position", [250 + 250 * idx, 300])
        node.setdefault("name", f"Node {idx + 1}")
        node.setdefault("parameters", {})
    
    for node in nodes[:-1]:
        name = node.get("name")
        if name and name not in connections:
            connections.setdefault(name, {"main": [[]]})
    
    workflow["nodes"] = nodes
    workflow["connections"] = connections
    return workflow

# ==================== API ENDPOINTS ====================

@app.post("/generate-workflow")
async def generate_workflow(request: WorkflowRequest) -> Dict[str, Any]:
    """Generate fully-configured n8n workflow from natural language instructions"""
    
    instructions = request.instructions.strip()
    
    if not instructions:
        raise HTTPException(status_code=400, detail="Instructions cannot be empty")
    
    if model is None:
        if not initialize_gemini():
            raise HTTPException(
                status_code=500,
                detail="Gemini API not available. Please check configuration."
            )
    
    try:
        # Extract parameters from user input
        extracted_params = extract_parameters_from_instructions(instructions)
        print(f"\n📊 Extracted parameters: {extracted_params}")
        
        # Generate enhanced prompt with extracted parameters
        prompt = create_enhanced_prompt(instructions, extracted_params)
        
        print(f"\n{'='*60}")
        print(f"Generating workflow for: {instructions}")
        print(f"{'='*60}\n")
        
        # Call Gemini API
        response = model.generate_content(prompt)
        
        if not response or not hasattr(response, 'text'):
            raise HTTPException(
                status_code=500,
                detail="No valid response from Gemini API"
            )
        
        # Extract and validate workflow
        workflow = extract_json_from_response(response.text)
        
        # Apply smart configuration with extracted parameters
        workflow = smart_configure_nodes(workflow, extracted_params)
        workflow = auto_heal_workflow(workflow)
        
        print(f"✅ Workflow generated successfully: {workflow['name']}")
        print(f"   - Nodes: {len(workflow.get('nodes', []))}")
        print(f"   - Connections: {len(workflow.get('connections', {}))}")
        print(f"   - Configured parameters: {len(extracted_params)}\n")
        
        return workflow
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating workflow: {str(e)}"
        )

@app.get("/")
async def serve_index():
    """Serve the main HTML interface"""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        return JSONResponse(
            {"message": "n8n Workflow Generator API", "version": "2.0.0"},
            status_code=200
        )
    return FileResponse(index_path)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gemini_initialized": model is not None,
        "model": GEMINI_MODEL_NAME
    }

@app.get("/models")
async def list_available_models():
    """List all available Gemini models"""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        models = genai.list_models()
        
        model_list = []
        for m in models:
            model_list.append({
                "name": m.name,
                "display_name": m.display_name,
                "supported_methods": m.supported_generation_methods
            })
        
        return {
            "available_models": model_list,
            "current_model": GEMINI_MODEL_NAME,
            "total_count": len(model_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_json():
    """Serve OpenAPI specification"""
    return JSONResponse(app.openapi())

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting n8n Workflow Generator API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)