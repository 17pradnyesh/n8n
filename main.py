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

# Use Gemini 2.5 models (available in your API)
GEMINI_MODEL_NAME = "gemini-2.5-flash"  # Updated to use Gemini 2.5
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDrA125KgTm_SKp8JmgbzL7Dix276ODieU")

# Alternative models to try if the above fails (in priority order)
FALLBACK_MODELS = [
    "gemini-2.5-flash",  # Best option - free tier friendly
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
    description="AI-powered n8n workflow generator using Google Gemini"
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
        
        # Try to list available models first
        try:
            available_models = genai.list_models()
            model_names = [m.name for m in available_models if 'generateContent' in m.supported_generation_methods]
            print(f"📋 Available models: {[m.replace('models/', '') for m in model_names[:8]]}")
            
            # Prefer non-preview, non-pro models for free tier
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
        
        # Try primary model first, then fallbacks
        models_to_try = list(dict.fromkeys([GEMINI_MODEL_NAME] + FALLBACK_MODELS))  # Remove duplicates
        
        for model_name in models_to_try:
            try:
                test_model = genai.GenerativeModel(model_name)
                # Test with a minimal prompt to verify it works
                test_response = test_model.generate_content("Hi")
                if test_response and test_response.text:
                    model = test_model  # Use the working model
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

# Initialize at startup
print("=" * 60)
print("Initializing n8n Workflow Generator API...")
if not initialize_gemini():
    print("⚠️  Warning: Gemini API initialization failed")
print("=" * 60)

# ==================== REQUEST MODELS ====================

class WorkflowRequest(BaseModel):
    instructions: str

# ==================== PROMPT GENERATION ====================

def create_n8n_prompt(instructions: str) -> str:
    """Generate optimized prompt for Gemini API"""
    return f"""You are an expert n8n workflow architect. Create a production-ready n8n workflow for: "{instructions}"

**CRITICAL OUTPUT REQUIREMENTS:**
- Output MUST be valid JSON ONLY
- NO markdown code blocks, NO explanations
- Start with {{ and end with }}
- Must be importable directly into n8n

**MANDATORY STRUCTURE:**
{{
  "name": "Descriptive Workflow Name",
  "nodes": [
    // Complete node definitions with proper IDs, names, types, positions
  ],
  "connections": {{
    // COMPLETE connection mapping between all nodes
  }},
  "active": false,
  "settings": {{}},
  "versionId": "1"
}}

**CONNECTION RULES (CRITICAL):**
1. EVERY node except the last MUST have outgoing connections
2. EVERY node except triggers MUST have incoming connections
3. Node names in connections MUST exactly match node names in nodes array
4. Use this exact connection format:

"connections": {{
  "Node Name": {{
    "main": [
      [
        {{
          "node": "Next Node Name",
          "type": "main",
          "index": 0
        }}
      ]
    ]
  }}
}}

**ESSENTIAL NODE TYPES:**

Triggers:
- n8n-nodes-base.manualTrigger (Manual Trigger)
- n8n-nodes-base.scheduleTrigger (Schedule Trigger)
- n8n-nodes-base.webhook (Webhook)

Data Processing:
- n8n-nodes-base.httpRequest (HTTP Request)
- n8n-nodes-base.set (Set/Transform Data)
- n8n-nodes-base.code (JavaScript Code)

Logic:
- n8n-nodes-base.if (Conditional)
- n8n-nodes-base.switch (Switch)
- n8n-nodes-base.merge (Merge Data)

AI/LLM (use when AI/summarization/analysis needed):
- @n8n/n8n-nodes-langchain.lmChatGoogleGemini (Gemini Chat Model)
- @n8n/n8n-nodes-langchain.agent (AI Agent)

Output:
- n8n-nodes-base.emailSend (Send Email)
- n8n-nodes-base.slack (Slack)
- n8n-nodes-base.googleSheets (Google Sheets)

**POSITIONING PATTERN:**
- Start at [250, 300]
- Increment X by 250-300 for each sequential step
- Use Y=500 for error/alternative paths

**EXAMPLE - Simple Scheduled Workflow:**
{{
  "name": "Daily Weather Report",
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
      "name": "Schedule Trigger",
      "type": "n8n-nodes-base.scheduleTrigger",
      "typeVersion": 1,
      "id": "schedule-1",
      "position": [250, 300]
    }},
    {{
      "parameters": {{
        "url": "https://api.weather.com/data",
        "method": "GET"
      }},
      "name": "Fetch Weather",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 1,
      "id": "http-1",
      "position": [500, 300]
    }},
    {{
      "parameters": {{
        "fromEmail": "bot@example.com",
        "toEmail": "user@example.com",
        "subject": "Daily Weather Report",
        "text": "={{$json}}"
      }},
      "name": "Send Email",
      "type": "n8n-nodes-base.emailSend",
      "typeVersion": 1,
      "id": "email-1",
      "position": [750, 300]
    }}
  ],
  "connections": {{
    "Schedule Trigger": {{
      "main": [[{{"node": "Fetch Weather", "type": "main", "index": 0}}]]
    }},
    "Fetch Weather": {{
      "main": [[{{"node": "Send Email", "type": "main", "index": 0}}]]
    }}
  }}
}}

**AI-POWERED EXAMPLE (with Gemini):**
{{
  "name": "AI Content Summarizer",
  "nodes": [
    {{...webhook trigger...}},
    {{...http fetch content...}},
    {{
      "parameters": {{}},
      "name": "Gemini Chat Model",
      "type": "@n8n/n8n-nodes-langchain.lmChatGoogleGemini",
      "typeVersion": 1,
      "id": "gemini-1",
      "position": [500, 400]
    }},
    {{
      "parameters": {{}},
      "name": "AI Agent",
      "type": "@n8n/n8n-nodes-langchain.agent",
      "typeVersion": 2.2,
      "id": "agent-1",
      "position": [750, 300]
    }},
    {{...send result...}}
  ],
  "connections": {{
    "Gemini Chat Model": {{
      "ai_languageModel": [[{{"node": "AI Agent", "type": "ai_languageModel", "index": 0}}]]
    }},
    "Fetch Content": {{
      "main": [[{{"node": "AI Agent", "type": "main", "index": 0}}]]
    }},
    "AI Agent": {{
      "main": [[{{"node": "Send Result", "type": "main", "index": 0}}]]
    }}
  }}
}}

**IMPORTANT:**
- Include AI nodes when instructions mention: summarize, analyze, classify, generate, Gemini
- Connect Gemini Chat Model to AI Agent using "ai_languageModel" connection type
- Ensure ALL nodes are properly connected
- Use realistic, production-ready configurations

Generate the complete workflow JSON now:"""

# ==================== JSON EXTRACTION ====================

def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """Extract and parse JSON from Gemini response"""
    try:
        # Remove markdown code blocks if present
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object boundaries
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = response_text[start:end + 1]
            else:
                json_str = response_text
        
        # Clean and parse
        json_str = json_str.strip().strip('\ufeff')
        workflow = json.loads(json_str)
        
        # Validate structure
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
    # Use JSON schema if available
    if WORKFLOW_SCHEMA:
        try:
            jsonschema_validate(instance=workflow, schema=WORKFLOW_SCHEMA)
            return
        except ValidationError as e:
            print(f"Schema validation warning: {e}")
    
    # Basic validation
    required = ['name', 'nodes', 'connections']
    for field in required:
        if field not in workflow:
            raise ValueError(f"Missing required field: {field}")
    
    if not isinstance(workflow['nodes'], list) or not workflow['nodes']:
        raise ValueError("Workflow must have at least one node")
    
    # Validate node structure
    for i, node in enumerate(workflow['nodes']):
        required_node_fields = ['id', 'name', 'type', 'position']
        for field in required_node_fields:
            if field not in node:
                raise ValueError(f"Node {i} missing field: {field}")

# ==================== WORKFLOW ENHANCEMENT ====================

def needs_ai_processing(text: str) -> bool:
    """Check if instructions require AI processing"""
    keywords = [
        "summarize", "summary", "ai", "llm", "gemini",
        "classify", "categorize", "analyze", "generate",
        "extract", "rewrite", "intelligent"
    ]
    return any(kw in text.lower() for kw in keywords)

def auto_heal_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure workflow has proper structure and connections"""
    nodes = workflow.get("nodes", [])
    connections = workflow.get("connections", {})
    
    # Ensure all nodes have required fields
    for idx, node in enumerate(nodes):
        node.setdefault("id", f"node-{idx + 1}")
        node.setdefault("position", [250 + 250 * idx, 300])
        node.setdefault("name", f"Node {idx + 1}")
        node.setdefault("parameters", {})
    
    # Ensure connections exist for non-final nodes
    for node in nodes[:-1]:  # All except last node
        name = node.get("name")
        if name and name not in connections:
            connections.setdefault(name, {"main": [[]]})
    
    workflow["nodes"] = nodes
    workflow["connections"] = connections
    return workflow

def apply_native_node_preferences(workflow: Dict[str, Any], instructions: str) -> Dict[str, Any]:
    """Replace generic HTTP nodes with native integrations when appropriate"""
    text = instructions.lower()
    nodes = workflow.get("nodes", [])
    
    # Service mappings
    service_map = [
        ("google sheets", {"type": "n8n-nodes-base.googleSheets", "name": "Google Sheets"}),
        ("mongodb", {"type": "n8n-nodes-base.mongoDb", "name": "MongoDB"}),
        ("postgres", {"type": "n8n-nodes-base.postgres", "name": "Postgres"}),
        ("mysql", {"type": "n8n-nodes-base.mySql", "name": "MySQL"}),
        ("slack", {"type": "n8n-nodes-base.slack", "name": "Slack"}),
        ("discord", {"type": "n8n-nodes-base.discord", "name": "Discord"}),
    ]
    
    preferred = None
    for keyword, target in service_map:
        if keyword in text:
            preferred = target
            break
    
    if not preferred:
        return workflow
    
    # Replace generic HTTP nodes with native integration
    for node in nodes:
        is_http = "httpRequest" in node.get("type", "")
        is_trigger = "trigger" in node.get("type", "").lower()
        
        if is_http and not is_trigger:
            node["type"] = preferred["type"]
            node["name"] = preferred["name"]
            node.setdefault("parameters", {})
    
    return workflow

# ==================== API ENDPOINTS ====================

@app.post("/generate-workflow")
async def generate_workflow(request: WorkflowRequest) -> Dict[str, Any]:
    """Generate n8n workflow from natural language instructions"""
    
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
        # Generate prompt
        prompt = create_n8n_prompt(instructions)
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
        
        # Apply enhancements
        workflow = apply_native_node_preferences(workflow, instructions)
        workflow = auto_heal_workflow(workflow)
        
        print(f"✅ Workflow generated successfully: {workflow['name']}")
        print(f"   - Nodes: {len(workflow.get('nodes', []))}")
        print(f"   - Connections: {len(workflow.get('connections', {}))}\n")
        
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

# ==================== STARTUP ====================

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting n8n Workflow Generator API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)