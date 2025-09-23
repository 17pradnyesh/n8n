from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
import json
import re
import os
from typing import Dict, Any

app = FastAPI(title="n8n Workflow Generator", version="1.0.0")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API - Replace with your actual API key
GEMINI_API_KEY = "AIzaSyChYMUFD4sCJty11DeiAmdHSUMciM0mEd8"  # Replace this with your actual API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

class WorkflowRequest(BaseModel):
    instructions: str

def create_n8n_prompt(instructions: str) -> str:
    """
    Creates a highly-constrained prompt for the Gemini API to generate 
    a valid n8n workflow JSON with proper node connections.
    """
    return f"""You are an expert n8n workflow architect. Create a sophisticated n8n workflow for: "{instructions}"

**CRITICAL OUTPUT REQUIREMENTS:**
- Output MUST be a single, valid JSON object only
- NO markdown formatting or code blocks
- NO explanations or text before/after JSON
- JSON must be production-ready for n8n import

**MANDATORY CONNECTION RULES:**
1. EVERY node (except the last node) MUST have outgoing connections
2. EVERY node (except triggers) MUST have incoming connections
3. Node names in connections MUST exactly match node names in the nodes array
4. Connection structure MUST follow this exact pattern:

**CONNECTION STRUCTURE EXAMPLE:**
```
"connections": {{
  "Trigger Node Name": {{
    "main": [
      [
        {{
          "node": "Next Node Name",
          "type": "main",
          "index": 0
        }}
      ]
    ]
  }},
  "Next Node Name": {{
    "main": [
      [
        {{
          "node": "Final Node Name",
          "type": "main",
          "index": 0
        }}
      ]
    ]
  }}
}}
```

**NODE NAMING CONSISTENCY:**
- Use EXACT same names in "nodes" array and "connections" object
- Names are case-sensitive and must match perfectly
- Common pattern: "Schedule Trigger" → "HTTP Request" → "Set Data" → "Send Email"

**COMPREHENSIVE NODE TYPES:**

**Triggers:**
- n8n-nodes-base.manualTrigger (name: "Manual Trigger")
- n8n-nodes-base.scheduleTrigger (name: "Schedule Trigger") 
- n8n-nodes-base.webhook (name: "Webhook")

**Data Processing:**
- n8n-nodes-base.httpRequest (name: "HTTP Request", "API Call", "Fetch Data")
- n8n-nodes-base.set (name: "Set Data", "Transform Data", "Process Data")
- n8n-nodes-base.code (name: "Code", "JavaScript", "Process Items")

**Logic Nodes:**
- n8n-nodes-base.if (name: "IF", "Check Condition")
- n8n-nodes-base.switch (name: "Switch")
- n8n-nodes-base.merge (name: "Merge", "Combine Data")

**AI / LLM Nodes (use when prompts mention AI / summarize / classify / generate):**
- @n8n/n8n-nodes-ai.aiModel (name: "AI Model") — set provider to "googleGemini" and model to "gemini-1.5-flash" or "gemini-1.5-pro"; prefer temperature 0.2
- @n8n/n8n-nodes-ai.aiAgent (name: "AI Agent") — connect an AI Model node to power the agent for reasoning/tool-use

When the user mentions AI, Gemini, summarize, generate text, classify, extract, etc., you MUST include an AI Model or AI Agent node between data fetching/processing and the final output node.

Configure AI Model minimal parameters example:
{{"provider":"googleGemini","model":"gemini-1.5-flash","temperature":0.2,"prompt":"Summarize the following in 5 bullet points."}}

**Output Nodes:**
- n8n-nodes-base.emailSend (name: "Send Email")
- n8n-nodes-base.slack (name: "Slack", "Send Slack Message")
- n8n-nodes-base.googleSheets (name: "Google Sheets", "Save to Sheets")

**Error Handling:**
- n8n-nodes-base.stopAndError (name: "Error Handler")
- n8n-nodes-base.noOp (name: "Log Error")

**WORKFLOW CONNECTION PATTERNS:**

**Linear Flow (most common):**
Trigger → Process → Transform → Output

**With Error Handling:**
Trigger → Try Process → Success Path → Output
                   → Error Path → Error Handler

**Conditional Flow:**
Trigger → Fetch → IF → True Path → Action
                   → False Path → Alternative Action

**DETAILED EXAMPLES:**

**Example 1 - Simple Email Workflow:**
```json
{{
  "name": "Daily Email Report",
  "nodes": [
    {{
      "parameters": {{
        "rule": {{
          "interval": [{{
            "field": "cronExpression",
            "cronExpression": "0 9 * * *"
          }}]
        }}
      }},
      "name": "Daily Trigger",
      "type": "n8n-nodes-base.scheduleTrigger",
      "typeVersion": 1,
      "id": "schedule-1",
      "position": [250, 300]
    }},
    {{
      "parameters": {{
        "url": "https://api.example.com/data",
        "method": "GET"
      }},
      "name": "Fetch Data",
      "type": "n8n-nodes-base.httpRequest", 
      "typeVersion": 1,
      "id": "http-1",
      "position": [500, 300]
    }},
    {{
      "parameters": {{
        "fromEmail": "sender@example.com",
        "toEmail": "recipient@example.com",
        "subject": "Daily Report",
        "text": "Here is your daily report"
      }},
      "name": "Send Report",
      "type": "n8n-nodes-base.emailSend",
      "typeVersion": 1,
      "id": "email-1", 
      "position": [750, 300]
    }}
  ],
  "connections": {{
    "Daily Trigger": {{
      "main": [
        [{{
          "node": "Fetch Data",
          "type": "main",
          "index": 0
        }}]
      ]
    }},
    "Fetch Data": {{
      "main": [
        [{{
          "node": "Send Report", 
          "type": "main",
          "index": 0
        }}]
      ]
    }}
  }}
}}
```

**CONNECTION VALIDATION CHECKLIST:**
✅ Every trigger connects to next node
✅ Every processing node connects to next node
✅ Node names match exactly between nodes and connections
✅ Final nodes (email, slack, etc.) don't need outgoing connections
✅ Error handlers are connected from processing nodes

**POSITIONING GUIDELINES:**
- Triggers: [250, 300]
- Processing: [500, 300], [750, 300], [1000, 300]
- Error paths: [500, 500], [750, 500]
- Increment X by 250-300 for each step
- Use Y=500 for error/alternative paths

**MANDATORY STRUCTURE:**
{{
  "name": "Descriptive Workflow Name",
  "nodes": [
    // All nodes with proper IDs, names, types, positions
  ],
  "connections": {{
    // COMPLETE connection mapping - this is CRITICAL
    // Every node must be properly connected
  }},
  "active": false,
  "settings": {{}},
  "versionId": "1"
}}

**Example 2 - Summarize with Gemini then email:**
```json
{{
  "name": "Daily News Summary",
  "nodes": [
    {{"parameters":{{"rule":{{"interval":[{{"field":"cronExpression","cronExpression":"0 9 * * *"}}]}}}},"name":"Schedule Trigger","type":"n8n-nodes-base.scheduleTrigger","typeVersion":1,"id":"cron-1","position":[250,300]}},
    {{"parameters":{{"url":"https://api.example.com/news","method":"GET"}},"name":"Fetch News","type":"n8n-nodes-base.httpRequest","typeVersion":1,"id":"http-1","position":[500,300]}},
    {{"parameters":{{"provider":"googleGemini","model":"gemini-1.5-flash","temperature":0.2,"prompt":"Summarize the following news in 5 concise bullet points."}},"name":"AI Model","type":"@n8n/n8n-nodes-ai.aiModel","typeVersion":1,"id":"ai-1","position":[750,300]}},
    {{"parameters":{{"fromEmail":"sender@example.com","toEmail":"recipient@example.com","subject":"Daily News Summary","text":"={{$json.summary || $json.data}}"}},"name":"Send Email","type":"n8n-nodes-base.emailSend","typeVersion":1,"id":"email-1","position":[1000,300]}}
  ],
  "connections":{{
    "Schedule Trigger":{{"main":[[{{"node":"Fetch News","type":"main","index":0}}]]}},
    "Fetch News":{{"main":[[{{"node":"AI Model","type":"main","index":0}}]]}},
    "AI Model":{{"main":[[{{"node":"Send Email","type":"main","index":0}}]]}}
  }}
}}
```

ALWAYS pick the most suitable nodes for the request. If summarization or AI reasoning is needed, include AI Model or AI Agent configured for Google Gemini as shown. Keep connections complete and consistent.

**GENERATE COMPLETE WORKFLOW WITH PROPER CONNECTIONS NOW:**"""

def extract_json_from_response(response: str) -> Dict[Any, Any]:
    """Extract and validate JSON from Gemini response"""
    try:
        text = response if isinstance(response, str) else str(response)

        # Prefer fenced ```json blocks
        json_fence = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if json_fence:
            json_str = json_fence.group(1)
        else:
            # Accept any fenced block if language tag missing
            any_fence = re.search(r"```\s*([\s\S]*?)\s*```", text)
            if any_fence:
                json_str = any_fence.group(1)
            else:
                # Fallback: take substring from first '{' to last '}'
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
        raise HTTPException(status_code=400, detail=f"Invalid JSON generated: {str(e)}")
    except Exception as e:
        # Log the raw response for debugging
        raise HTTPException(status_code=400, detail=f"Workflow validation failed: {str(e)}. Raw response: {response}")

def apply_native_node_preferences(workflow: Dict[str, Any], instructions: str) -> Dict[str, Any]:
    """Prefer native n8n nodes over generic HTTP for well-known services.

    Non-invasive: only updates nodes which are clearly generic HTTP (or generic names),
    and only when the instruction strongly indicates a specific integration.
    """
    text = instructions.lower()
    nodes = workflow.get("nodes", [])

    keyword_to_node = [
        ("google sheets", {"type": "n8n-nodes-base.googleSheets", "name": "Google Sheets"}),
        ("sheet", {"type": "n8n-nodes-base.googleSheets", "name": "Google Sheets"}),
        ("mongodb", {"type": "n8n-nodes-base.mongoDb", "name": "MongoDB"}),
        ("mysql", {"type": "n8n-nodes-base.mySql", "name": "MySQL"}),
        ("postgres", {"type": "n8n-nodes-base.postgres", "name": "Postgres"}),
        ("perplexity", {"type": "@watzon/n8n-nodes-perplexity", "name": "Perplexity"}),
        ("aimlapi", {"type": "n8n-nodes-aimlapi", "name": "AI/ML API"}),
        ("discord", {"type": "n8n-nodes-base.discord", "name": "Discord"}),
        ("webhook", {"type": "n8n-nodes-base.webhook", "name": "Webhook"}),
        ("schedule", {"type": "n8n-nodes-base.scheduleTrigger", "name": "Schedule Trigger"}),
        ("http", {"type": "n8n-nodes-base.httpRequest", "name": "HTTP Request"}),
    ]

    # Determine preferred target if any keyword matches
    preferred = None
    for keyword, target in keyword_to_node:
        if keyword in text:
            preferred = target
            # Prefer most specific: break early for non-generic keywords
            if keyword not in ("http", "sheet", "schedule"):
                break

    if not preferred:
        return workflow

    # Replace obvious generic HTTP nodes with the preferred native node
    for node in nodes:
        node_type = str(node.get("type", ""))
        node_name = str(node.get("name", ""))
        is_generic_http = "httpRequest" in node_type or node_name.lower() in ("http request", "api call", "fetch data")

        # Do not change triggers when the preferred is not a trigger
        is_trigger = "trigger" in node_type.lower()

        # If preferred is a trigger, only adjust an existing trigger-like node name
        preferred_is_trigger = "trigger" in preferred["type"].lower()

        if preferred_is_trigger and is_trigger:
            node["type"] = preferred["type"]
            node["name"] = preferred["name"]
        elif (not preferred_is_trigger) and is_generic_http and not is_trigger:
            node["type"] = preferred["type"]
            # Preserve meaningful names if set; otherwise adopt a sensible name
            if node_name.lower() in ("http request", "api call", "fetch data"):
                node["name"] = preferred["name"]

        # Minimal parameter normalization for certain nodes
        if node["type"] == "n8n-nodes-base.googleSheets":
            node.setdefault("parameters", {}).setdefault("operation", "append")
        if node["type"] == "n8n-nodes-base.mongoDb":
            node.setdefault("parameters", {}).setdefault("operation", "insert")
        if node["type"] == "n8n-nodes-base.mySql":
            node.setdefault("parameters", {}).setdefault("operation", "executeQuery")
        if node["type"] == "n8n-nodes-base.postgres":
            node.setdefault("parameters", {}).setdefault("operation", "executeQuery")

    workflow["nodes"] = nodes
    return workflow

def needs_ai_processing(text: str) -> bool:
    """Detect if the instruction implies AI usage (summarize, generate, classify, Gemini)."""
    t = text.lower()
    keywords = [
        "summarize", "summary", "summarisation", "ai ", " llm", "gemini",
        "classify", "categorize", "extract", "rewrite", "generate", "analyze"
    ]
    return any(k in t for k in keywords)

def ensure_ai_node_present(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """If no AI node exists, inject an AI Model node and wire it into the main path."""
    nodes = workflow.get("nodes", [])
    connections = workflow.get("connections", {})

    # Already has AI node
    for n in nodes:
        if isinstance(n.get("type", ""), str) and ("aiModel" in n["type"] or "aiAgent" in n["type"]):
            return workflow

    # Find a likely fetch/process node and a likely sink (email/slack)
    fetch_idx = next((i for i, n in enumerate(nodes) if "httpRequest" in n.get("type", "")), None)
    sink_idx = next((i for i, n in enumerate(nodes) if any(x in n.get("type", "") for x in ["emailSend", "slack"])) , None)

    # Fallbacks: use first non-trigger as fetch, last node as sink
    if fetch_idx is None:
        fetch_idx = next((i for i, n in enumerate(nodes) if "trigger" not in n.get("type", "").lower()), 0)
    if sink_idx is None:
        sink_idx = max(0, len(nodes) - 1)

    fetch_node = nodes[fetch_idx] if nodes else None
    sink_node = nodes[sink_idx] if nodes else None

    # Create AI node
    ai_node = {
        "parameters": {
            "provider": "googleGemini",
            "model": "gemini-1.5-flash",
            "temperature": 0.2,
            "prompt": "Summarize the following input in 5 concise bullet points."
        },
        "name": "AI Model",
        "type": "@n8n/n8n-nodes-ai.aiModel",
        "typeVersion": 1,
        "id": f"ai-{len(nodes)+1}",
        "position": [
            (fetch_node.get("position", [500, 300])[0] + 250) if fetch_node else 750,
            (fetch_node.get("position", [500, 300])[1]) if fetch_node else 300
        ]
    }
    nodes.insert(fetch_idx + 1, ai_node)

    # Rewire connections: fetch -> AI -> sink
    # Remove direct fetch -> sink if present
    if fetch_node and fetch_node["name"] in connections:
        for path in connections.get(fetch_node["name"], {}).get("main", []):
            path[:] = [p for p in path if p.get("node") != (sink_node or {}).get("name")]

    # Ensure connections dict structure
    connections.setdefault(fetch_node["name"], {}).setdefault("main", [[]])
    connections.setdefault(ai_node["name"], {}).setdefault("main", [[]])

    # Connect fetch -> AI
    if not any(p.get("node") == ai_node["name"] for group in connections[fetch_node["name"]]["main"] for p in group):
        connections[fetch_node["name"]]["main"][0].append({"node": ai_node["name"], "type": "main", "index": 0})

    # Connect AI -> sink
    if sink_node:
        connections[ai_node["name"]]["main"][0].append({"node": sink_node["name"], "type": "main", "index": 0})

    workflow["nodes"] = nodes
    workflow["connections"] = connections
    return workflow

def ensure_ai_agent_present(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """If no AI Agent exists, inject Google Gemini Chat Model + AI Agent and connect them.

    Follows the user's exported structure:
    - Node types:
      - "@n8n/n8n-nodes-langchain.lmChatGoogleGemini" ("Google Gemini Chat Model")
      - "@n8n/n8n-nodes-langchain.agent" ("AI Agent")
    - Special connection key: ai_languageModel from Chat Model to AI Agent
    - Also wire main path: fetch -> AI Agent -> sink
    """
    nodes = workflow.get("nodes", [])
    connections = workflow.get("connections", {})

    # Return if already present
    has_agent = any("n8n-nodes-langchain.agent" in n.get("type", "") for n in nodes)
    if has_agent:
        return workflow

    # Locate fetch and sink
    fetch_idx = next((i for i, n in enumerate(nodes) if "httpRequest" in n.get("type", "")), None)
    if fetch_idx is None:
        fetch_idx = next((i for i, n in enumerate(nodes) if "trigger" not in n.get("type", "").lower()), 0)
    sink_idx = next((i for i, n in enumerate(nodes) if any(x in n.get("type", "") for x in ["emailSend", "slack"])) , None)
    if sink_idx is None:
        sink_idx = max(0, len(nodes) - 1)

    fetch_node = nodes[fetch_idx] if nodes else None
    sink_node = nodes[sink_idx] if nodes else None

    # Create nodes
    chat_node = {
        "parameters": {"options": {}},
        "type": "@n8n/n8n-nodes-langchain.lmChatGoogleGemini",
        "typeVersion": 1,
        "position": [
            (fetch_node.get("position", [500, 300])[0] - 150) if fetch_node else 350,
            (fetch_node.get("position", [500, 300])[1] + 100) if fetch_node else 400
        ],
        "id": f"gemini-chat-{len(nodes)+1}",
        "name": "Google Gemini Chat Model"
    }

    agent_node = {
        "parameters": {"options": {}},
        "type": "@n8n/n8n-nodes-langchain.agent",
        "typeVersion": 2.2,
        "position": [
            (fetch_node.get("position", [500, 300])[0] + 250) if fetch_node else 750,
            (fetch_node.get("position", [500, 300])[1]) if fetch_node else 300
        ],
        "id": f"agent-{len(nodes)+2}",
        "name": "AI Agent"
    }

    # Insert agent after fetch, and chat before/near it
    insert_index = fetch_idx + 1
    nodes.insert(insert_index, agent_node)
    nodes.insert(insert_index, chat_node)

    # Ensure dict structures
    connections.setdefault(chat_node["name"], {})
    connections[chat_node["name"]].setdefault("ai_languageModel", [[]])

    # Add chat -> agent link (special connection key)
    connections[chat_node["name"]]["ai_languageModel"][0].append({
        "node": agent_node["name"],
        "type": "ai_languageModel",
        "index": 0
    })

    # Main path: fetch -> agent
    if fetch_node:
        connections.setdefault(fetch_node["name"], {}).setdefault("main", [[]])
        # Remove direct fetch -> sink
        for group in connections[fetch_node["name"]]["main"]:
            group[:] = [p for p in group if p.get("node") != (sink_node or {}).get("name")]
        # Add fetch -> agent if missing
        if not any(p.get("node") == agent_node["name"] for group in connections[fetch_node["name"]]["main"] for p in group):
            connections[fetch_node["name"]]["main"][0].append({"node": agent_node["name"], "type": "main", "index": 0})

    # Wire agent to sink
    if sink_node:
        connections.setdefault(agent_node["name"], {}).setdefault("main", [[]])
        connections[agent_node["name"]]["main"][0].append({"node": sink_node["name"], "type": "main", "index": 0})

    workflow["nodes"] = nodes
    workflow["connections"] = connections
    return workflow

def validate_n8n_workflow(workflow: Dict[Any, Any]) -> None:
    """Basic validation of n8n workflow structure"""
    required_fields = ['name', 'nodes', 'connections']
    for field in required_fields:
        if field not in workflow:
            raise ValueError(f"Missing required field: {field}")

    if not isinstance(workflow['nodes'], list) or len(workflow['nodes']) == 0:
        raise ValueError("Workflow must have at least one node")

    # Validate each node has required fields
    for i, node in enumerate(workflow['nodes']):
        node_required = ['id', 'name', 'type', 'position']
        for field in node_required:
            if field not in node:
                raise ValueError(f"Node {i} missing required field: {field}")

@app.post("/generate-workflow", response_model=Dict[str, Any])
async def generate_workflow(request: WorkflowRequest):
    instructions = request.instructions.strip()

    # Basic validation
    if not instructions:
        raise HTTPException(status_code=400, detail="Instructions cannot be empty")

    # Create prompt for Gemini
    prompt = create_n8n_prompt(instructions)
    print(f"Generated Prompt for Gemini: {prompt}")  # Debug: log the prompt

    # Call Gemini API
    try:
        response = model.generate_content(prompt)
        print(f"Raw response from Gemini: {response}")  # Debug: log the raw response

        # Extract text content from Gemini response
        try:
            response_text = getattr(response, "text", None)
            if response_text is None:
                # Fallback: attempt to stringify
                response_text = str(response)
        except Exception:
            response_text = str(response)

        # Extract and validate JSON from response text
        workflow_json = extract_json_from_response(response_text)

        # Prefer native nodes for known services based on instructions
        workflow_json = apply_native_node_preferences(workflow_json, instructions)

        # Auto-inject AI processing nodes if needed
        if needs_ai_processing(instructions):
            workflow_json = ensure_ai_node_present(workflow_json)

        # Auto-inject AI Agent if needed
        workflow_json = ensure_ai_agent_present(workflow_json)

        print(f"Final workflow JSON: {json.dumps(workflow_json, indent=2)}")  # Debug: log the final JSON
        return workflow_json
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating workflow: {str(e)}")

# Serve OpenAPI schema as JSON
@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_json():
    return JSONResponse(app.openapi())


@app.get("/")
async def serve_index():
    # Point directly to your index.html
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(index_path)

