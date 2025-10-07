# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import api

from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

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

# Include all the routes from the api.py file
app.include_router(api.router)