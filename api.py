from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from main import run_federated_query

load_dotenv()

app = FastAPI(
    title="Federated Medical RAG API",
    description="REST API for medical question answering with federated learning",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "federated-medical-rag"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Federated Medical RAG API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/query": "POST - Query medical questions",
            "/docs": "API documentation"
        }
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Process a medical query through the federated RAG system
    
    Args:
        request: QueryRequest containing the medical question
    
    Returns:
        QueryResponse with answer and sources
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = run_federated_query(request.query)
        return QueryResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

