"""Shared test configuration."""
import sys
import os

# Add backend directory to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force mock mode for all tests
os.environ["USE_MOCK_GEMINI"] = "true"
os.environ["GEMINI_API_KEY"] = ""
os.environ["DATABASE_URL"] = "sqlite:///test.db"
