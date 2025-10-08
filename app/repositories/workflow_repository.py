# app/repositories/workflow_repository.py

import google.generativeai as genai
import re
import json
import logging
from typing import Dict, Any, Optional
from jsonschema import validate as jsonschema_validate, ValidationError
from fastapi import HTTPException

from app.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    FALLBACK_MODELS,
    WORKFLOW_SCHEMA,
    ERRORLOG,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkflowGenerationError(Exception):
    """Base exception for workflow generation errors"""
    pass

class ModelInitializationError(WorkflowGenerationError):
    """Failed to initialize Gemini model"""
    pass

class ValidationError(WorkflowGenerationError):
    """Workflow validation failed"""
    pass

class ParameterExtractionError(WorkflowGenerationError):
    """Failed to extract parameters from instructions"""
    pass

class WorkflowGenerator:
    def __init__(self):
        """Initialize WorkflowGenerator with empty model."""
        self.model: Optional[genai.GenerativeModel] = None
        self.current_model_name: str = GEMINI_MODEL_NAME
        
    def initialize_gemini(self) -> bool:
        """Initialize the Gemini model with API key and fallback handling.
        
        Returns:
            bool: True if initialization successful, False otherwise
            
        Raises:
            ModelInitializationError: If initialization fails with all models
        """
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY is not set in environment variables")
            raise ModelInitializationError("API key not configured")
            
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            models_to_try = list(dict.fromkeys([self.current_model_name] + FALLBACK_MODELS))
            
            for model_name in models_to_try:
                try:
                    test_model = genai.GenerativeModel(model_name)
                    # Test the model with a simple request
                    test_model.generate_content("Hi")
                    self.model = test_model
                    self.current_model_name = model_name
                    logger.info(f"Gemini API initialized successfully with: {model_name}")
                    return True
                except Exception as e:
                    logger.warning(f"Model {model_name} failed: {str(e)[:100]}")
                    continue
            
            logger.error("All models failed to initialize")
            raise ModelInitializationError("No working models found")
            
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {str(e)}")
            raise ModelInitializationError(f"API configuration failed: {str(e)}")

    def generate_workflow(self, instructions: str) -> Dict[str, Any]:
        """Generate a complete n8n workflow from natural language instructions.
        
        Args:
            instructions (str): Natural language description of the desired workflow
            
        Returns:
            Dict[str, Any]: Complete, validated n8n workflow configuration
            
        Raises:
            HTTPException: If workflow generation fails
        """
        if self.model is None:
            try:
                if not self.initialize_gemini():
                    raise ModelInitializationError("Failed to initialize Gemini")
            except ModelInitializationError as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Gemini API not available: {str(e)}"
                )
                
        try:
            extracted_params = self.extract_parameters_from_instructions(instructions)
            logger.info(f"Extracted parameters: {extracted_params}")

            prompt = self.create_enhanced_prompt(instructions, extracted_params)
            
            response = self.model.generate_content(prompt)
            if not response or not hasattr(response, 'text'):
                raise HTTPException(
                    status_code=500,
                    detail="Invalid response from Gemini API"
                )
                
            workflow = self.extract_json_from_response(response.text)
            workflow = self.smart_configure_nodes(workflow, extracted_params)
            workflow = self.auto_heal_workflow(workflow)

            logger.info(f"Workflow generated successfully: {workflow.get('name', 'Untitled')}")
            # Return both generated workflow and the standard errorlog JSON
            return {
                "workflow": workflow,
                "errorlog": ERRORLOG or {}
            }
            
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            logger.error(f"Workflow generation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Workflow generation failed: {str(e)}"
            )

    def extract_parameters_from_instructions(self, instructions: str) -> Dict[str, Any]:
        """Extract specific parameters mentioned in user instructions.
        
        Args:
            instructions (str): Natural language workflow instructions
            
        Returns:
            Dict[str, Any]: Extracted parameters and their values
            
        Raises:
            ParameterExtractionError: If parameter extraction fails
        """
        try:
            params = {}
            text = instructions.lower()
            
            # Enhanced email pattern with better TLD handling
            EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
            emails = re.findall(EMAIL_PATTERN, instructions, re.IGNORECASE)
            if emails:
                params['emails'] = emails
                params['from_email'] = emails[0]
                params['to_email'] = emails[-1] if len(emails) > 1 else emails[0]
            
            # Enhanced URL pattern with better protocol and path handling
            URL_PATTERN = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*'
            urls = re.findall(URL_PATTERN, instructions)
            if urls:
                params['urls'] = urls
                params['api_url'] = urls[0]
            
            # Time extraction with AM/PM handling
            time_pattern = r'(\d{1,2})(?:\s*:\s*\d{2})?\s*(?:am|pm|AM|PM)?'
            times = re.findall(time_pattern, text)
            time_indicators = ['daily', 'every day', 'each day', 'at']
            
            if times and any(indicator in text for indicator in time_indicators):
                hour = int(times[0])
                if 'pm' in text.split(times[0])[1][:3] and hour < 12:
                    hour += 12
                elif 'am' in text.split(times[0])[1][:3] and hour == 12:
                    hour = 0
                params['schedule_hour'] = hour
                params['cron_expression'] = f"0 {hour} * * *"
            
            # API key extraction with multiple formats
            API_KEY_PATTERNS = [
                r'api[_\s-]?key[:\s=]+([A-Za-z0-9_\-\.]{20,})',
                r'token[:\s=]+([A-Za-z0-9_\-\.]{20,})',
                r'key[:\s=]+([A-Za-z0-9_\-\.]{20,})'
            ]
            
            for pattern in API_KEY_PATTERNS:
                matches = re.findall(pattern, instructions, re.IGNORECASE)
                if matches:
                    params['api_key'] = matches[0]
                    break
            
            # Database connection strings
            if 'mongodb://' in instructions:
                mongo_match = re.search(r'(mongodb://[^\s]+)', instructions)
                if mongo_match:
                    params['mongodb_connection'] = mongo_match.group(1)
            
            # Channel names (Slack/Discord)
            channel_pattern = r'#([a-z0-9-]+)|channel[:\s]+([a-z0-9-]+)'
            channel_matches = re.findall(channel_pattern, text)
            if channel_matches:
                params['slack_channel'] = next((m for group in channel_matches for m in group if m), None)
            
            # Spreadsheet names
            sheet_pattern = r'(?:spreadsheet|sheet)[:\s]+["\']?([^"\']+)["\']?'
            sheet_match = re.search(sheet_pattern, text, re.IGNORECASE)
            if sheet_match:
                params['sheet_name'] = sheet_match.group(1).strip()
            
            return params
            
        except Exception as e:
            logger.error(f"Parameter extraction failed: {str(e)}")
            raise ParameterExtractionError(f"Failed to extract parameters: {str(e)}")

    def create_enhanced_prompt(self, instructions: str, extracted_params: Dict[str, Any]) -> str:
        """Generate an enhanced prompt for the Gemini model with parameter context.
        
        Args:
            instructions (str): Original user instructions
            extracted_params (Dict[str, Any]): Parameters extracted from instructions
            
        Returns:
            str: Complete prompt for the model
        """
        param_context = ""
        if extracted_params:
            param_context = f"\n**EXTRACTED PARAMETERS TO USE:**\n"
            for key, value in extracted_params.items():
                param_context += f"- {key}: {value}\n"
    
    # --- THIS IS THE FULL, UNTRUNCATED PROMPT ---
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

    def extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract and parse JSON from Gemini model response.
        
        Args:
            response_text (str): Raw text response from the model
            
        Returns:
            Dict[str, Any]: Parsed and validated workflow JSON
            
        Raises:
            HTTPException: If JSON extraction or validation fails
        """
        try:
            # Try multiple patterns to extract JSON
            patterns = [
                r"```json\s*([\s\S]*?)\s*```",  # JSON code block
                r"```\s*([\s\S]*?)\s*```",      # Any code block
                r"({[\s\S]*})"                  # Any JSON-like structure
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response_text)
                if match:
                    json_str = match.group(1).strip()
                    try:
                        workflow = json.loads(json_str)
                        self.validate_n8n_workflow(workflow)
                        return workflow
                    except json.JSONDecodeError:
                        continue
            
            # If no pattern matched, try the raw text
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = response_text[start:end + 1]
                workflow = json.loads(json_str)
                self.validate_n8n_workflow(workflow)
                return workflow
                
            raise ValueError("No valid JSON found in response")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON from AI: {str(e)}\n\nPreview: {response_text[:500]}"
            )
        except Exception as e:
            logger.error(f"Workflow validation failed: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Workflow validation failed: {str(e)}"
            )

    def validate_n8n_workflow(self, workflow: Dict[str, Any]) -> None:
        """Validate n8n workflow structure against schema and requirements.
        
        Args:
            workflow (Dict[str, Any]): Workflow to validate
            
        Raises:
            ValidationError: If workflow structure is invalid
        """
        if WORKFLOW_SCHEMA:
            try:
                jsonschema_validate(instance=workflow, schema=WORKFLOW_SCHEMA)
                return
            except ValidationError as e:
                logger.warning(f"Schema validation warning: {e.path} - {e.message}")
        
        for field in ['name', 'nodes', 'connections']:
            if field not in workflow:
                raise ValueError(f"Missing required field: {field}")
        if not isinstance(workflow.get('nodes'), list) or not workflow['nodes']:
            raise ValueError("Workflow must have at least one node")
        for i, node in enumerate(workflow['nodes']):
            for field in ['id', 'name', 'type', 'position']:
                if field not in node:
                    raise ValueError(f"Node {i} missing field: {field}")

    def smart_configure_nodes(self, workflow: Dict[str, Any], extracted_params: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently configure node parameters based on extracted data.
        
        Args:
            workflow (Dict[str, Any]): Workflow to configure
            extracted_params (Dict[str, Any]): Parameters extracted from instructions
            
        Returns:
            Dict[str, Any]: Workflow with configured nodes
        """
        nodes = workflow.get("nodes", [])
        for node in nodes:
            node_type = node.get("type", "")
            params = node.get("parameters", {})
            
            if "emailSend" in node_type:
                self._configure_email_node(params, extracted_params)
            elif "httpRequest" in node_type:
                self._configure_http_node(params, extracted_params)
            elif "scheduleTrigger" in node_type:
                self._configure_schedule_node(params, extracted_params)
            elif "slack" in node_type:
                self._configure_slack_node(params, extracted_params)
            elif "googleSheets" in node_type:
                self._configure_sheets_node(params, extracted_params)
            
            node["parameters"] = params
            
        workflow["nodes"] = nodes
        return workflow

    def _configure_email_node(self, params: Dict, extracted: Dict) -> None:
        """Configure email node parameters."""
        params.setdefault("fromEmail", extracted.get("from_email", "workflow@n8n.local"))
        params.setdefault("toEmail", extracted.get("to_email", "user@example.com"))
        params.setdefault("subject", "Workflow Result")
        if not params.get("text") and not params.get("html"):
            params["text"] = "={{$json}}"

    def _configure_http_node(self, params: Dict, extracted: Dict) -> None:
        """Configure HTTP request node parameters."""
        params.setdefault("url", extracted.get("api_url"))
        params.setdefault("method", "GET")
        if params.get("queryParameters"):
            params["sendQuery"] = True
        if extracted.get("api_key") and "authentication" not in params:
            params["authentication"] = "genericCredentialType"
            params.setdefault("genericAuthType", "httpHeaderAuth")
            params.setdefault("httpHeaderAuth", {
                "name": "Authorization",
                "value": f"Bearer {extracted['api_key']}"
            })

    def _configure_schedule_node(self, params: Dict, extracted: Dict) -> None:
        """Configure schedule trigger node parameters."""
        if "schedule_hour" in extracted:
            params["rule"] = {
                "interval": [{
                    "field": "cronExpression",
                    "cronExpression": f"0 {extracted['schedule_hour']} * * *"
                }]
            }

    def _configure_slack_node(self, params: Dict, extracted: Dict) -> None:
        """Configure Slack node parameters."""
        if "slack_channel" in extracted:
            channel = extracted["slack_channel"]
            if not channel.startswith("#"):
                channel = f"#{channel}"
            params["channel"] = channel

    def _configure_sheets_node(self, params: Dict, extracted: Dict) -> None:
        """Configure Google Sheets node parameters."""
        if "sheet_name" in extracted:
            params["sheetName"] = extracted["sheet_name"]
        params.setdefault("range", "A:Z")

    def auto_heal_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure workflow has proper structure and connections.
        
        Args:
            workflow (Dict[str, Any]): Workflow to heal
            
        Returns:
            Dict[str, Any]: Healed workflow with proper structure
        """
        nodes = workflow.get("nodes", [])
        connections = workflow.get("connections", {})
        
        for idx, node in enumerate(nodes):
            node.setdefault("id", f"node-{idx + 1}")
            node.setdefault("position", [250 + 250 * idx, 300])
            node.setdefault("name", f"Node {idx + 1}")
            node.setdefault("parameters", {})
        
        workflow["nodes"] = nodes
        workflow["connections"] = connections
        return workflow