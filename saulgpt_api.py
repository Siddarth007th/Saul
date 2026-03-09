from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
from legal_rag import search_law

app = FastAPI()

# Enable browser requests (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/generate"


# Request structure
class CaseRequest(BaseModel):
    case_type: str
    incident: str
    amount: str


@app.post("/generate")
def generate(data: CaseRequest):

    query = f"{data.case_type} {data.incident} {data.amount}"

    # Retrieve relevant law
    law_context = search_law(query)

    full_prompt = f"""
You are an expert Indian legal assistant.

Relevant law:
{law_context}

Case type: {data.case_type}
Incident: {data.incident}
Amount involved: {data.amount}

Write a formal legal complaint referencing the relevant IPC sections.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "mistral",
            "prompt": full_prompt,
            "stream": False
        }
    )

    result = response.json()["response"]

    return {"response": result}
