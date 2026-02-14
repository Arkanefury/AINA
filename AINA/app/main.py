# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from .model import load_model, generate_text

print("Loading model...")
MODEL, TOKENIZER = load_model("weights/OpenAINAG", device="cpu")
print("Model loaded!")

app = FastAPI(title="AINA Chatbot", version="1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.6
    max_length: int = 100


@app.get("/")
async def serve_html():
    return FileResponse("static/index.html")


@app.post("/api/chat")

async def chat(request: ChatRequest):
    try:
        query = request.message
        query = f"### Instruction: {query} ### Response:"
        mn = len(query)
        
        response = generate_text(
            model=MODEL,
            tokenizer=TOKENIZER,
            prompt=query,
            temperature=request.temperature,
            max_length=request.max_length
        )
        
        ans = response
        cleaned_response = ans[mn:]
        return {
            "success": True,
            "response": cleaned_response,
            "original_message": request.message
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}