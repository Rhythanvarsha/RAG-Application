#!/usr/bin/env python
"""Test script to verify imports."""

try:
    from main import app
    from config import settings
    from rag_pipeline import rag_pipeline
    from vector_db import vector_db

    print("OK All imports successful")
    print(f"OK App: {settings.app_name} v{settings.app_version}")
    print("OK Project setup complete")
except Exception as e:
    print(f"ERROR Import failed: {e}")
    raise SystemExit(1)
