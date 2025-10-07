# app/repositories/workflow_repository.py

import google.generativeai as genai
import re
import json
from typing import Dict, Any
from jsonschema import validate as jsonschema_validate, ValidationError
from fastapi import HTTPException

from app.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    FALLBACK_MODELS,
    WORKFLOW_SCHEMA,
)

# Global model instance
model: genai.GenerativeModel | None = None
current_model_name: str = GEMINI_MODEL_NAME

def initialize_gemini() -> bool:
    """Initialize Gemini API with proper error handling and fallback models"""
    global model, current_model_name
    if not GEMINI_API_KEY:
        print(" ERROR: GEMINI_API_KEY is not set in your environment variables.")
        return False
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
                current_model_name = preferred_models[0].replace('models/', '')
                print(f"🎯 Selected preferred model: {current_model_name}")
        except Exception as list_error:
            print(f"⚠️  Could not list models: {list_error}")
        
        models_to_try = list(dict.fromkeys([current_model_name] + FALLBACK_MODELS))

        # Try to list models first and prefer a 'flash' model when available
        available_model_names = []
        try:
            listed = genai.list_models()
            for m in listed:
                # Some SDKs return attributes differently; be defensive
                name = getattr(m, 'name', None) or m.get('name') if isinstance(m, dict) else None
                supported = getattr(m, 'supported_generation_methods', None) or m.get('supported_generation_methods') if isinstance(m, dict) else None
                if name and supported and 'generateContent' in supported:
                    available_model_names.append(name)
            if available_model_names:
                print(f" Available models: {[n.replace('models/', '') for n in available_model_names[:8]]}")
                # Prefer flash (non-preview, non-lite) models
                preferred = [n for n in available_model_names if 'flash' in n.lower() and 'preview' not in n.lower() and 'lite' not in n.lower()]
                if preferred:
                    # use the first preferred model as the primary candidate
                    current_model_name = preferred[0].replace('models/', '')
                    print(f" Selected preferred model from list: {current_model_name}")
        except Exception as list_err:
            # Listing models may fail depending on API/version; fall back to configured names
            print(f" Could not list models: {str(list_err)[:200]}")

        # Build the ordered list of model names to try
        candidates = []
        # Start with the current configured name, then any discovered preferred name, then fallbacks
        if GEMINI_MODEL_NAME:
            candidates.append(GEMINI_MODEL_NAME)
        if current_model_name and current_model_name not in candidates:
            candidates.append(current_model_name)
        candidates.extend(FALLBACK_MODELS)
        models_to_try = list(dict.fromkeys([c for c in candidates if c]))

        for model_name in models_to_try:
            try:
                test_model = genai.GenerativeModel(model_name)
                test_response = test_model.generate_content("Hi")
                # Some SDKs return an object with .text
                text_ok = bool(getattr(test_response, 'text', None) or (isinstance(test_response, dict) and test_response.get('text')))
                if text_ok:
                    model = test_model
                    current_model_name = model_name
                    print(f" Gemini API initialized successfully with: {model_name}")
                    return True
                else:
                    print(f" Model {model_name} responded but no usable text returned")
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower():
                    print(f" Model {model_name}: Quota exceeded or rate-limited, trying next...")
                elif "404" in err or "not found" in err.lower():
                    print(f" Model {model_name}: Not found, trying next...")
                else:
                    print(f" Model {model_name} failed: {err[:200]}")
                continue

        print(" All models failed to initialize. Check API key and quotas.")
        return False
    except Exception as e:
        print(f" Failed to configure Gemini API: {str(e)}")
        return False

def generate_workflow_from_instructions(instructions: str) -> Dict[str, Any]:
    """Generate fully-configured n8n workflow from natural language instructions"""
    global model
    if model is None:
        if not initialize_gemini():
            raise HTTPException(
                status_code=503,
                detail="Gemini API is not available. Please check server configuration."
            )
    
    instructions = instructions.strip()
    if not instructions:
        raise HTTPException(status_code=400, detail="Instructions cannot be empty")
    
    try:
        # Extract parameters from user input
        extracted_params = extract_parameters_from_instructions(instructions)
        print(f"\n📊 Extracted parameters: {extracted_params}")
        
        # Generate enhanced prompt with extracted parameters
        prompt = create_enhanced_prompt(instructions, extracted_params)
        
        print(f"\n{'='*60}")
        print(f"Generating workflow for: {instructions}")
        print(f"{'='*60}\n")
        
        # Call Gemini API with safety configuration
        generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            candidate_count=1,
            max_output_tokens=8192  # Increased for complex workflows
        )
        
        # Generate workflow
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        if not response or not hasattr(response, 'text'):
            raise HTTPException(status_code=500, detail="No valid response from Gemini API")
        
        # Extract and validate workflow
        workflow = extract_json_from_response(response.text)
        
        # Apply smart configuration with extracted parameters
        workflow = smart_configure_nodes(workflow, extracted_params)
        workflow = auto_heal_workflow(workflow)
        
        print(f"✅ Workflow generated successfully: {workflow.get('name', 'Untitled')}")
        print(f"   - Nodes: {len(workflow.get('nodes', []))}")
        print(f"   - Connections: {len(workflow.get('connections', {}))}")
        print(f"   - Configured parameters: {len(extracted_params)}\n")
        
        return workflow
        
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            # Attempt emergency fallback generation
            try:
                print(" Attempting emergency fallback generation...")
                return generate_simple_workflow(instructions, extracted_params)
            except:
                raise HTTPException(
                    status_code=504,
                    detail="The workflow is too complex. Please try one of these simpler approaches:\n"
                          "1. Break down into multiple smaller workflows\n"
                          "2. Generate the basic workflow first, then add complexity\n"
                          "3. Specify one main workflow path at a time"
                )
        raise HTTPException(status_code=500, detail=f"Workflow generation failed: {error_msg}")

def generate_workflow_skeleton(instructions: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a basic workflow skeleton quickly"""
    # Create a simplified prompt focused on core workflow structure
    skeleton_prompt = f"""Create a basic n8n workflow skeleton with just the essential nodes for:
    {instructions}
    
    Include only:
    1. Main trigger node
    2. Core processing node
    3. Final output node
    
    Keep it minimal but functional."""
    
    try:
        response = model.generate_content(
            skeleton_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,  # Lower temperature for more focused output
                candidate_count=1,
                max_output_tokens=2048,  # Smaller limit for faster response
                top_k=20,
                top_p=0.8
            )
        )
        
        if response and hasattr(response, 'text'):
            workflow = extract_json_from_response(response.text)
            return workflow
    except Exception as e:
        print(f" Skeleton generation failed: {str(e)}")
        return None

def enhance_workflow_incrementally(base_workflow: Dict[str, Any], 
                                instructions: str, 
                                params: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance a basic workflow by improving each part separately"""
    try:
        # 1. Enhance the trigger node configuration
        if base_workflow["nodes"]:
            trigger_node = base_workflow["nodes"][0]
            enhanced_trigger = enhance_node_config(trigger_node, "trigger", instructions, params)
            base_workflow["nodes"][0] = enhanced_trigger

        # 2. Add necessary data processing nodes
        processing_nodes = generate_processing_nodes(instructions, params)
        if processing_nodes:
            insert_point = 1  # After trigger node
            base_workflow["nodes"][insert_point:insert_point] = processing_nodes
            
        # 3. Enhance output nodes
        if len(base_workflow["nodes"]) > 1:
            output_node = base_workflow["nodes"][-1]
            enhanced_output = enhance_node_config(output_node, "output", instructions, params)
            base_workflow["nodes"][-1] = enhanced_output
            
        # 4. Update connections
        base_workflow = auto_heal_workflow(base_workflow)
        
        return base_workflow
        
    except Exception as e:
        print(f" Enhancement failed: {str(e)}")
        return base_workflow  # Return original if enhancement fails

def generate_simple_workflow(instructions: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a simplified version of the requested workflow"""
    # Extract the core functionality
    core_instruction = simplify_workflow_request(instructions)
    
    simple_prompt = f"""Create a simplified n8n workflow for: {core_instruction}
    Focus on core functionality only. Keep it minimal but working."""
    
    try:
        response = model.generate_content(
            simple_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                candidate_count=1,
                max_output_tokens=2048,
                top_k=20,
                top_p=0.8
            )
        )
        
        if response and hasattr(response, 'text'):
            workflow = extract_json_from_response(response.text)
            workflow = smart_configure_nodes(workflow, params)
            workflow = auto_heal_workflow(workflow)
            return workflow
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate even a simplified workflow. Please try with simpler requirements."
        )

def simplify_workflow_request(instructions: str) -> str:
    """Extract the core functionality from complex workflow instructions"""
    # Look for main action words
    main_actions = re.findall(r'(send|save|process|create|update|get|fetch|monitor|check|notify)', 
                            instructions.lower())
    
    # Look for key nouns/objects
    key_objects = re.findall(r'(email|slack|sheet|database|file|data|message|notification|webhook)', 
                           instructions.lower())
    
    if main_actions and key_objects:
        return f"{main_actions[0]} {key_objects[0]}"
    
    # Fallback to first sentence or clause
    parts = re.split(r'[.,;]', instructions)
    return parts[0].strip()

    try:
        # Initialize base workflow structure
        base_workflow = {
            "name": "Generated Workflow",
            "nodes": [],
            "connections": {},
            "active": False,
            "settings": {},
            "versionId": "1"
        }

        # Configure generation parameters for better quality
        generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=4096,
            candidate_count=1
        )

        for idx, component in enumerate(workflow_components):
            try:
                # Generate component-specific prompt
                component_prompt = create_component_prompt(component, extracted_params)
                print(f" Generating component {idx + 1}/{len(workflow_components)}: {component['type']}")

                # Generate component with proper configuration
                response = model.generate_content(
                    contents=component_prompt,
                    generation_config=generation_config
                )

                if response and hasattr(response, 'text'):
                    # Extract and process component JSON
                    component_json = extract_json_from_response(response.text)
                    if component_json and "nodes" in component_json:
                        # Merge component nodes into base workflow
                        base_workflow = merge_workflow_components(base_workflow, component_json)
                
            except Exception as comp_error:
                print(f" Warning: Component {idx + 1} generation had an issue: {str(comp_error)}")
                continue

        if not base_workflow["nodes"]:
            # If component approach failed, try one-shot generation
            print(" Attempting full workflow generation...")
            prompt = create_enhanced_prompt(instructions, extracted_params, is_complex=True)
            
            # Use more focused generation parameters for full workflow
            full_generation_config = genai.types.GenerationConfig(
                temperature=0.5,  # More focused output
                top_p=0.9,
                top_k=20,
                max_output_tokens=8192,  # Allow larger output
                candidate_count=1
            )
            
            response = model.generate_content(
                contents=prompt,
                generation_config=full_generation_config
            )
            
            if not response or not hasattr(response, 'text'):
                raise HTTPException(status_code=500, detail="No valid response from Gemini API")
            
        # Extract and validate the base workflow
        workflow = extract_json_from_response(response.text)
        
        # Process each component and enhance the workflow
        for component in workflow_components:
            workflow = enhance_workflow_component(workflow, component, extracted_params)
        
        # Apply smart configuration and auto-healing
        workflow = smart_configure_nodes(workflow, extracted_params)
        workflow = auto_heal_workflow(workflow)

        print(f" Workflow generated successfully: {workflow.get('name', 'Untitled')}")
        return workflow
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        if "timeout" in str(e).lower():
            raise HTTPException(
                status_code=504,
                detail="The workflow generation took too long. Please try breaking down your request into simpler parts."
            )
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

def break_down_complex_workflow(instructions: str) -> list:
    """Break down complex workflow instructions into manageable components."""
    components = []
    
    # Split on keywords that indicate different parts of the workflow
    keywords = ["if", "then", "else", "when", "and", "or", "upload", "save", "send", "process"]
    
    # First, try to identify the trigger/input part
    if any(word in instructions.lower() for word in ["webhook", "upload", "form", "trigger", "schedule"]):
        components.append({
            "type": "trigger",
            "description": instructions.split("then")[0] if "then" in instructions else instructions
        })
    
    # Look for conditional logic
    if "if" in instructions.lower():
        conditions = instructions.lower().split("if")[1:]
        for condition in conditions:
            if "then" in condition:
                cond_part = condition.split("then")[0]
                action_part = condition.split("then")[1].split("else")[0]
                components.append({
                    "type": "condition",
                    "condition": cond_part.strip(),
                    "action": action_part.strip()
                })
                
                # Check for else clause
                if "else" in condition:
                    else_action = condition.split("else")[1]
                    components.append({
                        "type": "else",
                        "action": else_action.strip()
                    })
    
    # Look for processing steps
    processing_keywords = ["process", "analyze", "transform", "calculate", "format"]
    for keyword in processing_keywords:
        if keyword in instructions.lower():
            components.append({
                "type": "processing",
                "description": extract_relevant_part(instructions, keyword)
            })
    
    # Look for output/actions
    output_keywords = ["save", "send", "notify", "store", "upload"]
    for keyword in output_keywords:
        if keyword in instructions.lower():
            components.append({
                "type": "output",
                "description": extract_relevant_part(instructions, keyword)
            })
    
    # If no components were identified, treat it as a single component
    if not components:
        components.append({
            "type": "single",
            "description": instructions
        })
    
    return components

def extract_relevant_part(text: str, keyword: str) -> str:
    """Extract the relevant part of the instruction containing the keyword."""
    parts = text.split('.')
    for part in parts:
        if keyword in part.lower():
            return part.strip()
    return text

def enhance_workflow_component(workflow: Dict[str, Any], component: Dict[str, Any], 
                             extracted_params: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance a specific component of the workflow with additional configuration."""
    
    if component["type"] == "condition":
        # Find or create IF node
        if_node = None
        for node in workflow["nodes"]:
            if "if" in node["type"].lower():
                if_node = node
                break
                
        if if_node:
            # Enhance IF node configuration
            if_node["parameters"] = if_node.get("parameters", {})
            if "amount" in component["condition"].lower():
                if_node["parameters"]["conditions"] = {
                    "conditions": [{
                        "value1": "={{$json.amount}}",
                        "operation": "smaller",
                        "value2": extract_amount(component["condition"])
                    }]
                }
    
    elif component["type"] == "trigger":
        # Configure trigger nodes
        for node in workflow["nodes"]:
            if any(trigger in node["type"].lower() for trigger in ["webhook", "form", "trigger"]):
                node["parameters"] = node.get("parameters", {})
                if "webhook" in node["type"].lower():
                    node["parameters"]["path"] = "/incoming-data"
                    node["parameters"]["responseMode"] = "lastNode"
                elif "form" in node["type"].lower():
                    node["parameters"]["title"] = "Data Input Form"
    
    return workflow

def extract_amount(condition: str) -> float:
    """Extract numerical amount from condition text."""
    import re
    amounts = re.findall(r'[\d,.]+', condition)
    if amounts:
        # Remove commas and convert to float
        return float(amounts[0].replace(',', ''))
    return 0

# ==============================================================================
# ALL HELPER FUNCTIONS FROM YOUR ORIGINAL SCRIPT ARE BELOW
# ==============================================================================

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
    
    # Extract API keys
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
    
    # Extract other details...
    if 'mongodb://' in instructions:
        mongo_match = re.search(r'(mongodb://[^\s]+)', instructions)
        if mongo_match:
            params['mongodb_connection'] = mongo_match.group(1)
    
    slack_pattern = r'#([a-z0-9-]+)|channel[:\s]+([a-z0-9-]+)'
    slack_matches = re.findall(slack_pattern, text)
    if slack_matches:
        params['slack_channel'] = next((m for group in slack_matches for m in group if m), None)
    
    sheet_pattern = r'(?:spreadsheet|sheet)[:\s]+["\']?([^"\']+)["\']?'
    sheet_match = re.search(sheet_pattern, text, re.IGNORECASE)
    if sheet_match:
        params['sheet_name'] = sheet_match.group(1).strip()
        
    return params

def create_component_prompt(component: Dict[str, Any], extracted_params: Dict[str, Any]) -> str:
    """Generate a focused prompt for a specific workflow component"""
    param_context = ""
    if extracted_params:
        param_context = f"\n**EXTRACTED PARAMETERS TO USE:**\n"
        for key, value in extracted_params.items():
            param_context += f"- {key}: {value}\n"
    
    component_type = component["type"]
    description = component.get("description", "")
    
    type_specific_guide = {
        "trigger": """
**TRIGGER NODE GUIDELINES:**
1. Configure webhook/scheduler with proper parameters
2. Set up proper error handling
3. Include data validation where needed
""",
        "condition": """
**CONDITION NODE GUIDELINES:**
1. Set up IF node with proper conditions
2. Include both true/false branches
3. Configure data comparisons correctly
""",
        "processing": """
**PROCESSING NODE GUIDELINES:**
1. Include proper data transformation
2. Handle all potential data formats
3. Include error checking
""",
        "output": """
**OUTPUT NODE GUIDELINES:**
1. Configure all required parameters
2. Include proper data mapping
3. Handle potential errors
"""
    }.get(component_type, "")

    # Base prompt structure
    return f"""You are an expert n8n workflow architect. Create a focused workflow component that handles the following specific task:

**COMPONENT TYPE:** {component_type}
**DESCRIPTION:** {description}

{param_context}
{type_specific_guide}

**REQUIREMENTS:**
1. Output only the JSON for this specific component
2. Include all necessary node configurations
3. Ensure proper error handling
4. Make all parameters production-ready

Generate the component JSON now:"""

def enhance_instructions_with_structure(instructions: str) -> str:
    """Add structural hints to the instructions to improve generation quality"""
    # Add structure markers for different parts of the workflow
    parts = []
    
    # Identify trigger condition
    if any(word in instructions.lower() for word in ['when', 'if', 'on', 'after']):
        trigger_part = re.split(r'\s*(?:then|,|\band\b)\s*', instructions)[0]
        parts.append(f"TRIGGER: {trigger_part}")
    
    # Identify conditions
    if 'if' in instructions.lower():
        condition_matches = re.findall(r'if\s+([^,\s]+(?:\s+[^,\s]+)*)', instructions, re.IGNORECASE)
        for condition in condition_matches:
            parts.append(f"CONDITION: {condition}")
    
    # Identify actions
    action_keywords = ['then', 'send', 'save', 'create', 'update', 'process']
    for keyword in action_keywords:
        if keyword in instructions.lower():
            # Fixed the invalid escape sequence by using raw string
            action_matches = re.findall(fr"{keyword}\s+([^,]+)", instructions, re.IGNORECASE)
            for action in action_matches:
                parts.append(f"ACTION: {action}")
    
    # Combine with original instructions
    structured = instructions + "\n\n" + "\n".join(parts)
    return structured

def create_enhanced_prompt(instructions: str, extracted_params: Dict[str, Any], is_complex: bool = False) -> str:
    """Generate enhanced prompt with extracted parameters and complexity handling"""
    param_context = ""
    if extracted_params:
        param_context = f"\n**EXTRACTED PARAMETERS TO USE:**\n"
        for key, value in extracted_params.items():
            param_context += f"- {key}: {value}\n"
    
    # Add specific format requirements
    format_guide = """
**JSON FORMAT REQUIREMENTS:**
1. Use only ASCII characters in the JSON
2. Ensure all property names and values are properly quoted
3. No trailing commas in objects or arrays
4. No undefined or empty values
5. All nodes must have complete parameters
"""
    
    complexity_guide = """
**COMPLEX WORKFLOW GUIDELINES:**
1. Break down complex conditions into clear IF/ELSE nodes
2. Use Set nodes to prepare data between steps
3. Include error handling nodes where needed
4. Use clear node naming for complex flows
5. Implement proper data transformations between steps
""" if is_complex else ""
    
    # --- THIS IS THE FULL, UNTRUNCATED PROMPT WITH COMPLEXITY HANDLING ---
    return f"""You are an expert n8n workflow architect. Create a COMPLETE, PRODUCTION-READY workflow that requires ZERO manual configuration.
{complexity_guide}

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

def merge_workflow_components(base_workflow: Dict[str, Any], component: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a component workflow into the base workflow while maintaining proper connections"""
    # Keep track of existing node names to avoid duplicates
    existing_names = {node["name"] for node in base_workflow["nodes"]}
    
    # Update node IDs and names to avoid conflicts
    next_id = len(base_workflow["nodes"]) + 1
    for node in component.get("nodes", []):
        # Ensure unique node name
        original_name = node["name"]
        while node["name"] in existing_names:
            node["name"] = f"{original_name}_{next_id}"
        existing_names.add(node["name"])
        
        # Update node ID
        node["id"] = f"node-{next_id}"
        next_id += 1
        
        # Adjust node position
        if "position" in node:
            node["position"] = [
                node["position"][0] + (len(base_workflow["nodes"]) * 160),
                node["position"][1]
            ]
    
    # Merge nodes
    base_workflow["nodes"].extend(component.get("nodes", []))
    
    # Merge connections
    for from_node, connections in component.get("connections", {}).items():
        # Update connection names if they were changed
        updated_from_node = from_node
        for node in component.get("nodes", []):
            if node.get("name") == from_node and node.get("name") != original_name:
                updated_from_node = node["name"]
                break
                
        if updated_from_node not in base_workflow["connections"]:
            base_workflow["connections"][updated_from_node] = connections
        else:
            # Merge connection arrays
            for conn_type, conn_array in connections.items():
                if conn_type not in base_workflow["connections"][updated_from_node]:
                    base_workflow["connections"][updated_from_node][conn_type] = conn_array
                else:
                    base_workflow["connections"][updated_from_node][conn_type].extend(conn_array)
    
    return base_workflow

def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """Extract and parse JSON from Gemini response with improved error handling"""
    try:
        # First try to find JSON in code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no code block, look for JSON structure
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = response_text[start:end + 1]
            else:
                raise ValueError("No valid JSON structure found in response")

        # Clean and normalize the JSON string
        def clean_json_string(s: str) -> str:
            # Remove any Unicode control characters
            s = ''.join(char for char in s if ord(char) >= 32 or char in '\n\r\t')
            
            # Normalize whitespace while preserving structure
            s = re.sub(r'\s+(?=(?:[^"]*"[^"]*")*[^"]*$)', ' ', s)
            
            # Fix common JSON issues
            s = re.sub(r',(\s*[}\]])', r'\1', s)  # Remove trailing commas
            s = re.sub(r'\\\n\s*', '', s)  # Remove escaped newlines
            s = re.sub(r'\\(?!["\\/bfnrtu])', '', s)  # Remove invalid escapes
            s = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', s)  # Remove control chars
            
            # Ensure property names are properly quoted
            s = re.sub(r'(\{|\,)\s*([a-zA-Z0-9_]+)\s*:', r'\1 "\2":', s)
            
            # Fix missing values
            s = re.sub(r':\s*,', ': null,', s)
            s = re.sub(r':\s*\}', ': null}', s)
            
            return s.strip()

        # Clean the JSON string
        json_str = clean_json_string(json_str)
        
        try:
            # First attempt: direct parse
            workflow = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f" Initial JSON parse failed: {str(e)}")
            
            try:
                # Second attempt: stricter cleaning
                json_str = re.sub(r'[^\x20-\x7E\n\r\t]', '', json_str)
                workflow = json.loads(json_str)
            except json.JSONDecodeError as e2:
                print(f" Second JSON parse attempt failed: {str(e2)}")
                
                # Third attempt: try to salvage partial JSON
                try:
                    # Find all complete objects
                    valid_json_pattern = r'\{(?:[^{}]|(?R))*\}'
                    matches = re.finditer(valid_json_pattern, json_str)
                    
                    for match in matches:
                        try:
                            partial = json.loads(match.group())
                            if "nodes" in partial or "connections" in partial:
                                print(" Successfully extracted partial workflow")
                                return partial
                        except:
                            continue
                            
                    raise e2  # If no valid partial JSON found, raise the original error
                except:
                    raise e2

        # Validate the workflow structure
        validate_n8n_workflow(workflow)
        return workflow
        
    except json.JSONDecodeError as e:
        print(f" JSON parsing error: {str(e)}\nAttempting to fix...")
        # Try to extract partial JSON
        try:
            matches = re.findall(r'{[^{}]*}', response_text)
            if matches:
                for match in matches:
                    try:
                        partial = json.loads(match)
                        if "nodes" in partial or "connections" in partial:
                            return partial
                    except:
                        continue
        except:
            pass
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON from AI: {e}\n\nPreview: {response_text[:500]}"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Workflow validation failed: {e}")

def validate_n8n_workflow(workflow: Dict[str, Any]) -> None:
    """Validate n8n workflow structure"""
    if WORKFLOW_SCHEMA:
        try:
            jsonschema_validate(instance=workflow, schema=WORKFLOW_SCHEMA)
            return
        except ValidationError as e:
            print(f"Schema validation warning: {e.path} - {e.message}")
    
    for field in ['name', 'nodes', 'connections']:
        if field not in workflow:
            raise ValueError(f"Missing required field: {field}")
    if not isinstance(workflow.get('nodes'), list) or not workflow['nodes']:
        raise ValueError("Workflow must have at least one node")
    for i, node in enumerate(workflow['nodes']):
        for field in ['id', 'name', 'type', 'position']:
            if field not in node:
                raise ValueError(f"Node {i} missing field: {field}")

def smart_configure_nodes(workflow: Dict[str, Any], extracted_params: Dict[str, Any]) -> Dict[str, Any]:
    """Intelligently configure node parameters based on extracted data"""
    nodes = workflow.get("nodes", [])
    
    # First, ensure node types are correct
    for node in nodes:
        # Fix node types to match n8n's expectations
        if "webhook" in node.get("name", "").lower():
            node["type"] = "n8n-nodes-base.webhook"
            node["typeVersion"] = 1
        elif "sentiment" in node.get("name", "").lower() and "http" in node.get("type", "").lower():
            node["type"] = "n8n-nodes-base.httpRequest"
            node["typeVersion"] = 3
        elif "if" in node.get("type", "").lower():
            node["type"] = "n8n-nodes-base.if"
            node["typeVersion"] = 1
        elif "email" in node.get("type", "").lower():
            node["type"] = "n8n-nodes-base.emailSend"
            node["typeVersion"] = 1
        elif "slack" in node.get("type", "").lower():
            node["type"] = "n8n-nodes-base.slack"
            node["typeVersion"] = 1
        elif "set" in node.get("type", "").lower():
            node["type"] = "n8n-nodes-base.set"
            node["typeVersion"] = 2
            
    # Then configure parameters
    for node in nodes:
        node_type = node.get("type", "").lower()
        params = node.get("parameters", {})
        
        if "emailsend" in node_type:
            params.setdefault("fromEmail", extracted_params.get("from_email", "workflow@n8n.local"))
            params.setdefault("toEmail", extracted_params.get("to_email", "user@example.com"))
            params.setdefault("subject", "Workflow Result")
            if not params.get("text") and not params.get("html"):
                params["text"] = "={{$json}}"
        
        elif "httprequest" in node_type:
            params.setdefault("url", extracted_params.get("api_url"))
            params.setdefault("method", "GET")
            if params.get("queryParameters"):
                params["sendQuery"] = True
            if extracted_params.get("api_key") and "authentication" not in params:
                params["authentication"] = "genericCredentialType"
                params.setdefault("genericAuthType", "httpHeaderAuth")
                params.setdefault("httpHeaderAuth", {"name": "Authorization", "value": f"Bearer {extracted_params['api_key']}"})
        
        elif "googleforms" in node_type:
            # Ensure Google Forms trigger is configured
            params.setdefault("operation", "getResponses")
            params.setdefault("range", "A:Z")  # Full range
        
        elif "code" in node_type or "function" in node_type:
            # Ensure code nodes have proper JavaScript
            if not params.get("jsCode") and ("sentiment" in node.get("name", "").lower() or "analyze" in node.get("name", "").lower()):
                # Add sentiment analysis code
                params["jsCode"] = """
// Analyze sentiment using a simple scoring approach
function analyzeSentiment(text) {
    const positive = ['great', 'good', 'excellent', 'happy', 'satisfied', 'awesome', 'love'];
    const negative = ['bad', 'poor', 'unhappy', 'disappointed', 'terrible', 'hate'];
    
    text = text.toLowerCase();
    let score = 0;
    
    positive.forEach(word => {
        const regex = new RegExp('\\\\b' + word + '\\\\b', 'g');
        score += (text.match(regex) || []).length;
    });
    
    negative.forEach(word => {
        const regex = new RegExp('\\\\b' + word + '\\\\b', 'g');
        score -= (text.match(regex) || []).length;
    });
    
    return {
        text,
        score,
        sentiment: score > 0 ? 'positive' : score < 0 ? 'negative' : 'neutral',
        summary: `Sentiment score: ${score} (${score > 0 ? 'positive' : score < 0 ? 'negative' : 'neutral'})`
    };
}

// Get the feedback text from input
const item = $input.first();
const feedbackText = item.json?.feedback || item.json?.text || item.json?.response || item.json;

// Return analysis result
return {
    json: {
        ...analyzeSentiment(feedbackText),
        original: feedbackText
    }
};
"""
        
        elif "ai" in node_type or "openai" in node_type or "gemini" in node_type:
            # Configure AI-related nodes (adjust based on your n8n version's node types)
            params.setdefault("authentication", "genericCredentialType")
            params.setdefault("prompt", "Analyze the following feedback and determine the sentiment (positive/negative/neutral): {{$json.text}}")
            params.setdefault("output", "json")
        
        elif "slack" in node_type:
            params.setdefault("channel", extracted_params.get("slack_channel", "general"))
            if not params.get("text"):
                params["text"] = "={{$json.message || $json.summary || JSON.stringify($json)}}"
        
        node["parameters"] = params
    
    workflow["nodes"] = nodes
    return workflow

def auto_heal_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure workflow has proper structure and connections.
    - Assigns missing node IDs and positions
    - Ensures all nodes are connected (no orphans)
    - Creates logical flow based on node types and names with proper branching
    """
    nodes = workflow.get("nodes", [])
    connections = workflow.get("connections", {})
    
    # 1. First pass: assign IDs and positions with better layout
    x_spacing = 280  # Horizontal spacing between nodes
    y_spacing = 200  # Vertical spacing for branches
    current_x = 250
    current_y = 300
    max_y = current_y
    
    for idx, node in enumerate(nodes):
        # Ensure unique node IDs
        node.setdefault("id", f"node-{str(idx + 1).zfill(3)}")
        
        # Assign node name if missing
        if "name" not in node:
            node_type = node.get("type", "").split(".")[-1]
            node["name"] = f"{node_type.title()} {idx + 1}"
            
        # Initialize parameters if missing
        node.setdefault("parameters", {})
        
        # Smart position assignment based on node type and connections
        node_type = node.get("type", "").lower()
        if "trigger" in node_type or "webhook" in node_type or idx == 0:
            # Place triggers at the start
            node["position"] = [current_x, current_y]
            current_x += x_spacing
        elif "if" in node_type:
            # Place IF nodes with space for branches
            node["position"] = [current_x, current_y]
            current_x += x_spacing
            max_y = max(max_y, current_y + y_spacing)  # Reserve space for true/false branches
        elif any(x in node_type for x in ["email", "slack", "discord", "send"]):
            # Place output nodes at the end of their branch
            if idx > 0:
                prev_node = nodes[idx - 1]
                if "if" in prev_node.get("type", "").lower():
                    # Position in the appropriate branch
                    branch_y = current_y + (y_spacing if idx % 2 == 0 else -y_spacing)
                    node["position"] = [current_x, branch_y]
                else:
                    node["position"] = [current_x, current_y]
            else:
                node["position"] = [current_x, current_y]
            current_x += x_spacing
        else:
            # Standard positioning for other nodes
            node["position"] = [current_x, current_y]
            current_x += x_spacing
    
    # 2. Second pass: ensure all nodes are properly connected
    def find_nodes_by_type(pattern: str) -> list:
        return [n for n in nodes if pattern.lower() in n.get("type", "").lower()]
    
    def find_nodes_by_name(pattern: str) -> list:
        return [n for n in nodes if pattern.lower() in n.get("name", "").lower()]
    
    # Identify node categories
    trigger_nodes = (find_nodes_by_type("trigger") + 
                    find_nodes_by_type("webhook") + 
                    find_nodes_by_type("form"))
    
    processing_nodes = (find_nodes_by_type("code") + 
                       find_nodes_by_type("function") + 
                       find_nodes_by_type("http") +  # Include HTTP nodes
                       find_nodes_by_type("ai") + 
                       find_nodes_by_type("set"))
    
    output_nodes = (find_nodes_by_type("email") + 
                   find_nodes_by_type("slack") + 
                   find_nodes_by_type("discord") +
                   find_nodes_by_name("send") +
                   find_nodes_by_name("notify"))
    
    # If no clear triggers found, use first node
    if not trigger_nodes and nodes:
        trigger_nodes = [nodes[0]]
    
    # Categorize remaining nodes
    remaining = [n for n in nodes if n not in trigger_nodes + processing_nodes + output_nodes]
    for node in remaining:
        name_lower = node.get("name", "").lower()
        node_type = node.get("type", "").lower()
        
        if "if" in node_type:
            # Keep IF nodes in their original position
            continue
        elif any(x in name_lower for x in ["process", "analyze", "extract", "transform", "get", "fetch"]):
            processing_nodes.append(node)
        elif any(x in name_lower for x in ["send", "notify", "alert", "output", "write"]):
            output_nodes.append(node)
        else:
            processing_nodes.append(node)
    
    # Reset and rebuild connections with proper branching
    connections.clear()
    
    # Track processed nodes to avoid duplicates
    processed_nodes = set()
    
    def add_connection(from_node: Dict[str, Any], to_node: Dict[str, Any], 
                      branch_index: int = 0, connection_type: str = "main"):
        """Helper to add a connection between nodes"""
        if from_node["name"] not in connections:
            connections[from_node["name"]] = {connection_type: [[]]}
        
        # Ensure we have enough arrays for the branch index
        while len(connections[from_node["name"]][connection_type]) <= branch_index:
            connections[from_node["name"]][connection_type].append([])
            
        connection = {
            "node": to_node["name"],
            "type": connection_type,
            "index": 0
        }
        
        # Add the connection if it doesn't exist
        if connection not in connections[from_node["name"]][connection_type][branch_index]:
            connections[from_node["name"]][connection_type][branch_index].append(connection)
    
    # Connect nodes with proper branching logic
    for idx, node in enumerate(nodes):
        if node["name"] in processed_nodes:
            continue
            
        node_type = node.get("type", "").lower()
        
        if "if" in node_type:
            # Handle IF node branching
            true_branch = []
            false_branch = []
            remaining_nodes = nodes[idx + 1:]
            
            # Find the next nodes for true/false branches
            for next_node in remaining_nodes:
                next_pos = next_node["position"]
                if next_pos[0] > node["position"][0]:  # Only look at nodes after this one
                    if next_pos[1] < node["position"][1]:  # Upper branch (true)
                        true_branch.append(next_node)
                    elif next_pos[1] > node["position"][1]:  # Lower branch (false)
                        false_branch.append(next_node)
            
            # Connect the branches
            if true_branch:
                add_connection(node, true_branch[0], branch_index=0)
                processed_nodes.add(true_branch[0]["name"])
            if false_branch:
                add_connection(node, false_branch[0], branch_index=1)
                processed_nodes.add(false_branch[0]["name"])
                
            processed_nodes.add(node["name"])
            
        elif idx < len(nodes) - 1:  # Not the last node
            next_node = nodes[idx + 1]
            if not any(["if" in n.get("type", "").lower() for n in [node, next_node]]):
                # Standard sequential connection
                add_connection(node, next_node)
                processed_nodes.add(node["name"])
    
    # Ensure all nodes have connection entries
    for node in nodes:
        if node["name"] not in connections:
            connections[node["name"]] = {"main": [[]]}
    
    workflow["nodes"] = nodes
    workflow["connections"] = connections
    return workflow
    return workflow