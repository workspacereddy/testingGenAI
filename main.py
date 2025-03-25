from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import Dict

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Google AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

class ChatMessage(BaseModel):
    message: str

class HealthData(BaseModel):
    bloodPressure: str
    bloodSugar: str
    cholesterol: str
    heartRate: str
    temperature: str

@app.get("/")
@app.head("/")
async def read_root():
    return {"message": "API is working!"}

# Handling OPTIONS request explicitly for CORS pre-flight
@app.options("/api/chat/")
async def handle_options_chat():
    return JSONResponse(content={}, status_code=200)

@app.options("/api/predict/")
async def handle_options_predict():
    return JSONResponse(content={}, status_code=200)

@app.post("/api/chat")
async def chat(message: ChatMessage) -> Dict[str, str]:
    prompt = f"""
    You are a medical AI assistant. Provide helpful but general health information.
    Always include a disclaimer that this is not professional medical advice.
    
    User question: {message.message}
    """
    
    try:
        response = model.generate_content(prompt)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in processing the request: {str(e)}")

@app.post("/api/predict")
async def predict(data: HealthData) -> Dict[str, str]:
    prompt = f"""
    Analyze the following health metrics and provide general health insights:
    - Blood Pressure: {data.bloodPressure} mmHg
    - Blood Sugar: {data.bloodSugar} mg/dL
    - Cholesterol: {data.cholesterol} mg/dL
    - Heart Rate: {data.heartRate} bpm
    - Temperature: {data.temperature} °F
    
    Provide a general health assessment and suggestions for maintaining or improving health.
    Include a disclaimer about consulting healthcare professionals.
    """
    
    try:
        response = model.generate_content(prompt)
        return {"prediction": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in processing the request: {str(e)}")
