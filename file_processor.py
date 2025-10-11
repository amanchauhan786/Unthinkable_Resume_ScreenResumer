import pdfplumber
import docx
import os
import logging
import re
from typing import Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download NLTK data if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class FileProcessor:
    """Handle file processing for resumes and job descriptions without spaCy"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
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
        """Extract skills from text using keyword matching"""
        try:
            # Comprehensive skill keywords
            skill_keywords = {
                # Programming Languages
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 
                'swift', 'kotlin', 'php', 'ruby', 'scala', 'r', 'matlab', 'perl',
                
                # Web Technologies
                'react', 'angular', 'vue', 'django', 'flask', 'fastapi', 'spring', 
                'express', 'node.js', 'nodejs', 'html', 'css', 'bootstrap', 'tailwind',
                'sass', 'less', 'jquery', 'webpack', 'babel',
                
                # Databases
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite',
                'cassandra', 'dynamodb', 'firebase',
                
                # Cloud & DevOps
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins',
                'ansible', 'puppet', 'chef', 'git', 'github', 'gitlab', 'ci/cd',
                
                # Data Science & AI
                'machine learning', 'deep learning', 'nlp', 'computer vision', 'ai',
                'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib',
                'seaborn', 'plotly', 'keras', 'opencv', 'nltk', 'spacy',
                
                # Mobile Development
                'react native', 'flutter', 'android', 'ios', 'xcode',
                
                # Tools & Methodologies
                'agile', 'scrum', 'kanban', 'jira', 'confluence', 'trello', 'slack',
                'microsoft office', 'excel', 'word', 'powerpoint', 'outlook',
                
                # Soft Skills
                'leadership', 'communication', 'teamwork', 'problem solving', 'analytical',
                'project management', 'time management', 'creativity', 'adaptability'
            }
            
            text_lower = text.lower()
            found_skills = []
            
            # Direct keyword matching
            for skill in skill_keywords:
                # Use word boundaries for better matching
                if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                    found_skills.append(skill)
            
            return sorted(list(set(found_skills)))
        except Exception as e:
            logger.error(f"Error extracting skills: {e}")
            return []
    
    def extract_experience(self, text: str) -> str:
        """Extract experience information using pattern matching"""
        try:
            # Look for experience-related sections
            experience_patterns = [
                r'experience.*?\n(.*?)(?:\n\n|\n[A-Z]|\n\s*$|$)',
                r'work.*?history.*?\n(.*?)(?:\n\n|\n[A-Z]|\n\s*$|$)',
                r'employment.*?\n(.*?)(?:\n\n|\n[A-Z]|\n\s*$|$)',
                r'professional.*?experience.*?\n(.*?)(?:\n\n|\n[A-Z]|\n\s*$|$)'
            ]
            
            for pattern in experience_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    experience_text = match.group(1).strip()
                    # Take first 200 characters
                    return experience_text[:200] + "..." if len(experience_text) > 200 else experience_text
            
            # If no specific section found, look for sentences with experience keywords
            sentences = sent_tokenize(text)
            experience_sentences = []
            experience_keywords = ['experience', 'worked', 'employed', 'position', 'role', 'job']
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in experience_keywords):
                    experience_sentences.append(sentence.strip())
                    if len(experience_sentences) >= 3:  # Limit to 3 sentences
                        break
            
            return " ".join(experience_sentences) if experience_sentences else "Experience information not found"
            
        except Exception as e:
            logger.error(f"Error extracting experience: {e}")
            return "Experience extraction failed"
    
    def extract_education(self, text: str) -> str:
        """Extract education information"""
        try:
            education_patterns = [
                r'education.*?\n(.*?)(?:\n\n|\n[A-Z]|\n\s*$|$)',
                r'qualifications.*?\n(.*?)(?:\n\n|\n[A-Z]|\n\s*$|$)',
                r'degree.*?\n(.*?)(?:\n\n|\n[A-Z]|\n\s*$|$)'
            ]
            
            for pattern in education_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    education_text = match.group(1).strip()
                    return education_text[:150] + "..." if len(education_text) > 150 else education_text
            
            return "Education information not found"
        except Exception as e:
            logger.error(f"Error extracting education: {e}")
            return "Education extraction failed"
