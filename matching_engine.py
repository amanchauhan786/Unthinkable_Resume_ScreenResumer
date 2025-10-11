import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import spacy
from fuzzywuzzy import fuzz
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MatchingEngine:
    """Handle resume-job description matching using multiple techniques"""
    
    def __init__(self, gemini_api_key: str, spacy_model: str = 'en_core_web_sm'):
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=gemini_api_key)
        
        # Initialize models
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load(spacy_model)
        self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    
    def compute_local_similarity(self, resume_text: str, jd_text: str) -> Dict[str, float]:
        """Compute similarity scores using local NLP models"""
        try:
            # SpaCy similarity
            doc_resume = self.nlp(resume_text[:1000000])  # Limit text length
            doc_jd = self.nlp(jd_text[:1000000])
            spacy_similarity = doc_resume.similarity(doc_jd)
            
            # Sentence transformer similarity
            resume_embedding = self.embed_model.encode(resume_text, convert_to_tensor=True)
            jd_embedding = self.embed_model.encode(jd_text, convert_to_tensor=True)
            transformer_similarity = util.pytorch_cos_sim(resume_embedding, jd_embedding).item()
            
            # Fuzzy matching
            fuzzy_score = fuzz.token_sort_ratio(resume_text.lower(), jd_text.lower()) / 100
            
            # Combine scores
            final_score = (spacy_similarity + transformer_similarity + fuzzy_score) / 3
            
            return {
                "spacy_similarity": round(spacy_similarity, 3),
                "transformer_similarity": round(transformer_similarity, 3),
                "fuzzy_match": round(fuzzy_score * 100, 2),
                "final_local_score": round(final_score * 10, 2)
            }
        except Exception as e:
            logger.error(f"Error computing local similarity: {e}")
            return {
                "spacy_similarity": 0.0,
                "transformer_similarity": 0.0,
                "fuzzy_match": 0.0,
                "final_local_score": 0.0
            }
    
    def gemini_resume_match(self, resume_text: str, jd_text: str, skills: list = None) -> Dict[str, Any]:
        """Use Gemini LLM for advanced matching and justification"""
        try:
            skills_text = ", ".join(skills) if skills else "Not specified"
            
            prompt = f"""
            You are an expert recruiter and HR analyst. Analyze the candidate's resume against the job description and provide a comprehensive evaluation.

            RESUME TEXT:
            {resume_text[:3000]}

            JOB DESCRIPTION:
            {jd_text[:3000]}

            CANDIDATE'S SKILLS (extracted): {skills_text}

            Please provide a JSON response with the following structure:
            {{
                "fit_score": <integer from 1-10>,
                "strengths": [<list of 3-5 key strengths>],
                "gaps": [<list of 2-3 key gaps or concerns>],
                "justification": "<detailed paragraph explaining the score and key matching factors>",
                "recommendation": "<short recommendation: 'Strong Recommend', 'Recommend', 'Consider', 'Not Suitable'>"
            }}

            Consider:
            - Skills match and relevance
            - Experience level alignment
            - Industry/domain fit
            - Potential for growth
            - Overall suitability

            Be objective and provide specific reasons for your evaluation.
            """
            
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text
            
            # Extract JSON from response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_str = response_text[start:end]
            
            result = json.loads(json_str)
            return result
            
        except Exception as e:
            logger.error(f"Error with Gemini matching: {e}")
            return {
                "fit_score": 0,
                "strengths": ["Error in analysis"],
                "gaps": ["Technical issue"],
                "justification": f"Error during analysis: {str(e)}",
                "recommendation": "Analysis Failed"
            }
    
    def comprehensive_match(self, resume_text: str, jd_text: str, skills: list = None) -> Dict[str, Any]:
        """Perform comprehensive matching using both local and LLM approaches"""
        # Local matching
        local_results = self.compute_local_similarity(resume_text, jd_text)
        
        # Gemini matching
        gemini_results = self.gemini_resume_match(resume_text, jd_text, skills)
        
        # Combined score (weighted average)
        local_weight = 0.3
        gemini_weight = 0.7
        combined_score = round(
            (local_results["final_local_score"] * local_weight) + 
            (gemini_results["fit_score"] * gemini_weight), 2
        )
        
        return {
            "local_scores": local_results,
            "gemini_scores": gemini_results,
            "final_score": combined_score,
            "skills_matched": skills
        }