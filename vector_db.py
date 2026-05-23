"""Vector Database Integration with Chroma DB"""
import chromadb
from typing import List, Dict, Any, Optional
from config import settings


class VectorDatabase:
    """Chroma DB Vector Database Manager"""
    
    def __init__(self):
        """Initialize Chroma DB client"""
        if settings.chroma_host and settings.chroma_port:
            # Remote Chroma DB
            self.client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port
            )
        else:
            # Local Chroma DB (persistent storage)
            self.client = chromadb.PersistentClient(path="./chroma_data")
        
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector database
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            ids: Optional document IDs
            
        Returns:
            List of document IDs
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        if metadata is None:
            metadata = [{} for _ in documents]
        
        self.collection.add(
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        return ids
    
    def query_documents(
        self,
        query_text: str,
        n_results: int = None
    ) -> Dict[str, Any]:
        """
        Query the vector database for similar documents
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            
        Returns:
            Dictionary with results
        """
        if n_results is None:
            n_results = settings.top_k_documents
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else []
        }
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from the database"""
        self.collection.delete(ids=ids)
    
    def clear_collection(self) -> None:
        """Clear all documents from collection"""
        self.collection.delete(where={})
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata
        }


# Initialize vector database
vector_db = VectorDatabase()
