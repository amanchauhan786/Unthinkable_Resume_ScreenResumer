import os
from typing import Dict, Any

class Config:
    """Configuration settings for the Resume Screener"""
    
    # Model settings
    SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
    SPACY_MODEL = 'en_core_web_sm'
    
    # Database settings
    DATABASE_PATH = 'resume_screener.db'
    
    # Gemini settings
    GEMINI_MODEL = "gemini-2.0-flash"
    
    # File processing settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt'}
    
    # Text processing
    MAX_TEXT_LENGTH = 4000

config = Config()