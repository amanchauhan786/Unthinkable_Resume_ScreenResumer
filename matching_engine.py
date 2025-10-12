import google.generativeai as genai
from fuzzywuzzy import fuzz
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(_name_)

class MatchingEngine:
    """Handle resume-job description matching with minimal dependencies"""
    
    def _init_(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    
    def compute_local_similarity(self, resume_text: str, jd_text: str) -> Dict[str, float]:
        """Compute similarity scores using only fuzzy matching and keyword analysis"""
        try:
            # Limit text length
            resume_text_limited = resume_text[:50000]
            jd_text_limited = jd_text[:50000]
            
            # Fuzzy matching scores
            fuzzy_score_ratio = fuzz.ratio(resume_text_limited.lower(), jd_text_limited.lower()) / 100
            fuzzy_score_partial = fuzz.partial_ratio(resume_text_limited.lower(), jd_text_limited.lower()) / 100
            fuzzy_score_token = fuzz.token_sort_ratio(resume_text_limited.lower(), jd_text_limited.lower()) / 100
            
            # Average fuzzy scores
            fuzzy_score = (fuzzy_score_ratio + fuzzy_score_partial + fuzzy_score_token) / 3
            
            # Keyword overlap analysis
            resume_words = set(resume_text_limited.lower().split())
            jd_words = set(jd_text_limited.lower().split())
            common_words = resume_words.intersection(jd_words)
            keyword_overlap = len(common_words) / max(len(jd_words), 1)
            
            # Length-based score (shorter texts might match better by chance)
            length_penalty = min(len(resume_text_limited), len(jd_text_limited)) / max(len(resume_text_limited), len(jd_text_limited), 1)
            
            # Combine scores
            final_score = (
                fuzzy_score * 0.6 + 
                keyword_overlap * 0.3 +
                length_penalty * 0.1
            )
            
            return {
                "fuzzy_ratio": round(fuzzy_score_ratio * 100, 2),
                "fuzzy_token": round(fuzzy_score_token * 100, 2),
                "keyword_overlap": round(keyword_overlap * 100, 2),
                "final_local_score": round(final_score * 10, 2)
            }
        except Exception as e:
            logger.error(f"Error computing local similarity: {e}")
            return {
                "fuzzy_ratio": 0.0,
                "fuzzy_token": 0.0,
                "keyword_overlap": 0.0,
                "final_local_score": 0.0
            }
    
    def gemini_resume_match(self, resume_text: str, jd_text: str, skills: list = None) -> Dict[str, Any]:
        """Use Gemini LLM for advanced matching and justification"""
        try:
            skills_text = ", ".join(skills) if skills else "Not specified"
            
            prompt = f"""
            You are an expert recruiter analyzing a candidate's fit for a job role.

            RESUME:
            {resume_text[:2500]}

            JOB DESCRIPTION:
            {jd_text[:2500]}

            EXTRACTED SKILLS: {skills_text}

            Provide a JSON response with this exact structure:
            {{
                "fit_score": 7,
                "strengths": ["Skill match", "Experience relevance"],
                "gaps": ["Missing specific technology", "Experience level"],
                "justification": "Detailed explanation here...",
                "recommendation": "Recommend"
            }}

            Consider skills match, experience relevance, and overall fit.
            """
            
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text
            
            # Try to extract JSON
            try:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except:
                # Fallback parsing
                result = {
                    "fit_score": 5,
                    "strengths": ["AI analysis completed"],
                    "gaps": ["Formatting issue in response"],
                    "justification": response_text[:300] if response_text else "Analysis completed",
                    "recommendation": "Consider"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error with Gemini matching: {e}")
            return {
                "fit_score": 0,
                "strengths": ["Analysis failed"],
                "gaps": ["Technical error"],
                "justification": f"Please check your API key and try again. Error: {str(e)}",
                "recommendation": "Failed"
            }
    
    def comprehensive_match(self, resume_text: str, jd_text: str, skills: list = None) -> Dict[str, Any]:
        """Perform comprehensive matching"""
        local_results = self.compute_local_similarity(resume_text, jd_text)
        gemini_results = self.gemini_resume_match(resume_text, jd_text, skills)
        
        # Use only Gemini score if local scoring fails
        if local_results["final_local_score"] == 0:
            final_score = gemini_results["fit_score"]
        else:
            final_score = round(
                (local_results["final_local_score"] * 0.3) + 
                (gemini_results["fit_score"] * 0.7), 2
            )
        
        return {
            "local_scores": local_results,
            "gemini_scores": gemini_results,
            "final_score": final_score,
            "skills_matched": skills
        }
