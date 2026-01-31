from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_engine import rag_engine
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI(title="AgriBot API", description="AI Assistant for Farmers")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    sources: list

@app.get("/")
def read_root():
    return {"status": "online", "message": "AgriBot Backend is running"}

@app.post("/api/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Get answer from RAG engine
    try:
        results = rag_engine.search(request.query)
        answer = rag_engine.generate_response(request.query)
        return {
            "response": answer,
            "sources": results
        }
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
