import pdfplumber
import docx
import os
import logging
from typing import Optional
import spacy

logger = logging.getLogger(__name__)

class FileProcessor:
    """Handle file processing for resumes and job descriptions"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            raise
    
    def read_text_file(self, file_path: str) -> str:
        """Read text from TXT file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            raise
    
    def process_file(self, file_path: str) -> str:
        """Process any supported file type and return text"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            return self.read_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def extract_skills(self, text: str) -> list:
        """Extract skills from text using spaCy"""
        try:
            doc = self.nlp(text)
            skills = []
            
            # Simple rule-based skill extraction
            skill_keywords = {
                'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
                'node.js', 'django', 'flask', 'fastapi', 'spring', 'express',
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis',
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
                'machine learning', 'deep learning', 'nlp', 'computer vision',
                'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
                'git', 'jenkins', 'ci/cd', 'agile', 'scrum'
            }
            
            # Extract skills using noun phrases and entities
            for token in doc:
                if token.text.lower() in skill_keywords and token.text.lower() not in skills:
                    skills.append(token.text.lower())
            
            # Also check noun phrases
            for chunk in doc.noun_chunks:
                if chunk.text.lower() in skill_keywords and chunk.text.lower() not in skills:
                    skills.append(chunk.text.lower())
            
            return sorted(list(set(skills)))
        except Exception as e:
            logger.error(f"Error extracting skills: {e}")
            return []
    
    def extract_experience(self, text: str) -> str:
        """Extract experience information"""
        try:
            # Simple experience extraction - can be enhanced
            doc = self.nlp(text)
            experience_keywords = ['experience', 'work', 'employment', 'career']
            
            sentences = []
            for sent in doc.sents:
                if any(keyword in sent.text.lower() for keyword in experience_keywords):
                    sentences.append(sent.text.strip())
            
            return " ".join(sentences[:3])  # Return first 3 relevant sentences
        except Exception as e:
            logger.error(f"Error extracting experience: {e}")
            return ""