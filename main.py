"""
RAG Application - FastAPI Server
Retrieval-Augmented Generation Pipeline with Ollama and Chroma DB
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from config import settings
from rag_pipeline import rag_pipeline
from vector_db import vector_db


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models
class QueryRequest(BaseModel):
    """Request model for user queries"""
    query: str
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    """Response model for queries"""
    query: str
    retrieved_documents: List[str]
    generated_response: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentInput(BaseModel):
    """Model for adding documents"""
    text: str
    metadata: Optional[Dict[str, Any]] = None


class BulkDocumentInput(BaseModel):
    """Model for adding multiple documents"""
    documents: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    app_name: str
    version: str


# Routes

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query through the RAG pipeline
    
    Args:
        request: QueryRequest with user query
        
    Returns:
        QueryResponse with retrieved documents and generated response
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Process query through RAG pipeline
        result = rag_pipeline.process_query(request.query, request.top_k)
        
        return {
            "query": result["query"],
            "retrieved_documents": result["retrieved_documents"],
            "generated_response": result["generated_response"],
            "metadata": {
                "num_documents_retrieved": len(result["retrieved_documents"]),
                "similarity_scores": result["similarity_scores"]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/documents/add")
async def add_document(document: DocumentInput):
    """
    Add a single document to the vector database
    
    Args:
        document: DocumentInput with text and optional metadata
        
    Returns:
        Document ID
    """
    try:
        if not document.text.strip():
            raise HTTPException(status_code=400, detail="Document text cannot be empty")
        
        doc_id = vector_db.add_documents(
            documents=[document.text],
            metadata=[document.metadata or {}]
        )[0]
        
        return {"document_id": doc_id, "status": "added"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")


@app.post("/documents/add-batch")
async def add_documents_batch(batch: BulkDocumentInput):
    """
    Add multiple documents to the vector database
    
    Args:
        batch: BulkDocumentInput with list of documents
        
    Returns:
        List of document IDs
    """
    try:
        if not batch.documents:
            raise HTTPException(status_code=400, detail="Documents list cannot be empty")
        
        doc_ids = vector_db.add_documents(
            documents=batch.documents,
            metadata=batch.metadatas or [{}] * len(batch.documents)
        )
        
        return {"document_ids": doc_ids, "count": len(doc_ids), "status": "added"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding documents: {str(e)}")


@app.get("/documents/stats")
async def get_database_stats():
    """Get vector database statistics"""
    try:
        stats = vector_db.get_collection_stats()
        return stats
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.delete("/documents/clear")
async def clear_database():
    """Clear all documents from the database"""
    try:
        vector_db.clear_collection()
        return {"status": "cleared", "message": "All documents removed"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "add_document": "/documents/add (POST)",
            "add_batch": "/documents/add-batch (POST)",
            "stats": "/documents/stats (GET)",
            "clear": "/documents/clear (DELETE)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
