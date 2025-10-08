# app/routes/api.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import os

from app.repositories import generate_workflow_from_instructions, workflow_generator
from app.config import INDEX_PATH

router = APIRouter()

class WorkflowRequest(BaseModel):
    instructions: str

@router.post("/generate-workflow", response_model=Dict[str, Any])
async def generate_workflow_endpoint(request: WorkflowRequest):
    """Generate fully-configured n8n workflow from natural language instructions"""
    instructions = request.instructions.strip()
    if not instructions:
        raise HTTPException(status_code=400, detail="Instructions cannot be empty")
    
    return generate_workflow_from_instructions(instructions)

@router.get("/")
async def serve_index():
    """Serve the main HTML interface"""
    if not os.path.exists(INDEX_PATH):
        return JSONResponse(
            {"message": "n8n Workflow Generator API", "version": "2.0.0"},
            status_code=200,
        )
    return FileResponse(INDEX_PATH)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gemini_initialized": getattr(workflow_generator, 'model', None) is not None,
        "model_in_use": getattr(workflow_generator, 'current_model_name', None),
    }