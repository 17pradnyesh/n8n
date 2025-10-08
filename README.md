AI-Powered n8n Workflow Generator
Tech Stack

Backend: FastAPI, Python, Uvicorn
AI Model: Google Gemini API
Validation: Pydantic, JSON Schema
Frontend: HTML, CSS, JavaScript
Environment Management: dotenv, Virtual Environment


Working

1.The user enters a natural language instruction (e.g., “Send me a daily weather report at 8 AM”).
2.The backend sends this instruction to the Gemini API for processing.
3.The API generates a structured n8n workflow JSON based on the input.
4.Parameters such as emails, URLs, and time schedules are automatically extracted.
5.The generated workflow is validated against a JSON schema.
6.The final validated workflow is returned to the frontend, ready to import into n8n.


Features

AI-based workflow generation from plain English input
Schema validation for proper n8n structure
Zero manual configuration with smart defaults
Modular backend with clean architecture
Simple frontend for quick testing and generation


Implementation Steps
1. Clone the Repository
git clone <your-repository-url>
cd n8n-workflow-generator

2. Create and Activate Virtual Environment
python -m venv env

# On Windows
.\env\Scripts\activate

# On macOS/Linux
source env/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Add Gemini API Key

Create a .env file in the project root:

GEMINI_API_KEY="YOURAPIKEY"

5. Run the Application
python run.py

6. Access in Browser

Open:

http://127.0.0.1:8000


