"""RAG (Retrieval-Augmented Generation) Pipeline"""
from typing import List, Dict, Any
import requests
import json
from config import settings
from vector_db import vector_db


class RAGPipeline:
    """Custom RAG Pipeline with Ollama Integration"""
    
    def __init__(self):
        """Initialize RAG pipeline"""
        self.ollama_url = settings.ollama_base_url
        self.embedding_model = settings.embedding_model
        self.language_model = settings.language_model
    
    def get_embeddings(self, text: str) -> List[float]:
        """
        Get embeddings for text using Ollama
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            return response.json().get("embedding", [])
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return []
    
    def retrieve_context(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Retrieve relevant documents from vector database
        
        Args:
            query: User query
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        return vector_db.query_documents(query, top_k or settings.top_k_documents)
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate response using Ollama language model
        
        Args:
            query: User query
            context: Retrieved context from vector database
            
        Returns:
            Generated response
        """
        prompt = self._build_prompt(query, context)
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.language_model,
                    "prompt": prompt,
                    "temperature": settings.temperature,
                    "num_predict": settings.max_tokens,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json().get("response", "No response generated")
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for language model"""
        return f"""You are a helpful assistant. Use the provided context to answer the user's question.

Context:
{context}

User Question: {query}

Answer:"""
    
    def process_query(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Process a complete query through the RAG pipeline
        
        Args:
            query: User query
            
        Returns:
            Dictionary with retrieved documents and generated response
        """
        # Step 1: Retrieve relevant documents
        retrieval_results = self.retrieve_context(query, top_k)
        
        # Step 2: Build context from retrieved documents
        context = "\n\n".join(retrieval_results.get("documents", []))
        
        # Step 3: Generate response based on context and query
        response = self.generate_response(query, context)
        
        return {
            "query": query,
            "retrieved_documents": retrieval_results.get("documents", []),
            "retrieved_metadatas": retrieval_results.get("metadatas", []),
            "similarity_scores": retrieval_results.get("distances", []),
            "generated_response": response
        }


# Initialize RAG pipeline
rag_pipeline = RAGPipeline()
