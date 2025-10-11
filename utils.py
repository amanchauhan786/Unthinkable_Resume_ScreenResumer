import os
import tempfile
import streamlit as st
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def save_uploaded_file(uploaded_file) -> Tuple[Optional[str], str]:
    """Save uploaded file to temporary location"""
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name, uploaded_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        return None, ""

def validate_file(file) -> Tuple[bool, str]:
    """Validate uploaded file"""
    if file is None:
        return False, "No file uploaded"
    
    # Check file size (10MB limit)
    if file.size > 10 * 1024 * 1024:
        return False, "File size exceeds 10MB limit"
    
    # Check file extension
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    file_extension = os.path.splitext(file.name)[1].lower()
    if file_extension not in allowed_extensions:
        return False, f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
    
    return True, "File is valid"

def display_score_with_color(score: float) -> str:
    """Return color based on score"""
    if score >= 8:
        return "🟢"  # Green
    elif score >= 6:
        return "🟡"  # Yellow
    elif score >= 4:
        return "🟠"  # Orange
    else:
        return "🔴"  # Red

def format_justification(justification: str) -> str:
    """Format justification text for better display"""
    if not justification:
        return "No justification provided."
    
    # Simple formatting - can be enhanced
    sentences = justification.split('. ')
    formatted = "• " + ".\n• ".join(sentences)
    return formatted.rstrip('• ')