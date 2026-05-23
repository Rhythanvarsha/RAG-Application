"""Configuration settings for RAC Application"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    app_name: str = "RAC Application"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Ollama Settings
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "llama2-embed"  # Using Llama 3.5 equivalent
    language_model: str = "llama2"  # Using Llama 3.7 equivalent
    
    # Vector Database Settings
    chroma_host: Optional[str] = None
    chroma_port: Optional[int] = None
    chroma_collection_name: str = "rac_documents"
    
    # RAG Settings
    top_k_documents: int = 5
    temperature: float = 0.7
    max_tokens: int = 512
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
