# run.py

import uvicorn
from app.main import app

if __name__ == "__main__":
    print("\n🚀 Starting n8n Workflow Generator API...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)