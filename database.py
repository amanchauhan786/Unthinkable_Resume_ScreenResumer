import sqlite3
import json
from typing import List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manage database operations for resume screening results"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        conn = self._get_connection()
        try:
            # Results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS screening_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_name TEXT NOT NULL,
                    resume_text TEXT,
                    jd_name TEXT NOT NULL,
                    jd_text TEXT,
                    local_score REAL,
                    gemini_score REAL,
                    final_score REAL,
                    justification TEXT,
                    skills_extracted TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Candidates table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_name TEXT,
                    resume_path TEXT UNIQUE,
                    resume_text TEXT,
                    skills TEXT,
                    experience TEXT,
                    education TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
        finally:
            conn.close()
    
    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def save_screening_result(self, result_data: Dict[str, Any]) -> int:
        """Save screening result to database"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO screening_results 
                (resume_name, resume_text, jd_name, jd_text, local_score, gemini_score, final_score, justification, skills_extracted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result_data['resume_name'],
                result_data.get('resume_text', ''),
                result_data['jd_name'],
                result_data.get('jd_text', ''),
                result_data['local_score'],
                result_data['gemini_score'],
                result_data['final_score'],
                result_data['justification'],
                json.dumps(result_data.get('skills_extracted', []))
            ))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving screening result: {e}")
            raise
        finally:
            conn.close()
    
    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all screening results"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM screening_results 
                ORDER BY created_at DESC
            """)
            columns = [col[0] for col in cursor.description]
            results = []
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                # Parse skills if exists
                if result.get('skills_extracted'):
                    result['skills_extracted'] = json.loads(result['skills_extracted'])
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Error fetching results: {e}")
            return []
        finally:
            conn.close()
    
    def save_candidate(self, candidate_data: Dict[str, Any]) -> int:
        """Save candidate information"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO candidates 
                (candidate_name, resume_path, resume_text, skills, experience, education)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                candidate_data.get('candidate_name', ''),
                candidate_data['resume_path'],
                candidate_data.get('resume_text', ''),
                json.dumps(candidate_data.get('skills', [])),
                candidate_data.get('experience', ''),
                candidate_data.get('education', '')
            ))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving candidate: {e}")
            raise
        finally:
            conn.close()
    
    def get_candidates(self) -> List[Dict[str, Any]]:
        """Get all candidates"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM candidates ORDER BY created_at DESC")
            columns = [col[0] for col in cursor.description]
            candidates = []
            for row in cursor.fetchall():
                candidate = dict(zip(columns, row))
                if candidate.get('skills'):
                    candidate['skills'] = json.loads(candidate['skills'])
                candidates.append(candidate)
            return candidates
        except Exception as e:
            logger.error(f"Error fetching candidates: {e}")
            return []
        finally:
            conn.close()