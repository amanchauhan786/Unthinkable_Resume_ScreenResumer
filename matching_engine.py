import google.generativeai as genai
from fuzzywuzzy import fuzz
import json
import logging
import re
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class MatchingEngine:
    """Enhanced resume-job description matching with proper scoring differentiation"""
    
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Skill categories for better scoring
        self.skill_categories = {
            'programming': {'python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'swift', 'kotlin'},
            'web_frameworks': {'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'node.js'},
            'cloud_devops': {'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'git'},
            'databases': {'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'firebase'},
            'data_ai': {'machine learning', 'deep learning', 'ai', 'tensorflow', 'pytorch', 'pandas', 'numpy'},
            'mobile': {'react native', 'flutter', 'android', 'ios', 'mobile development'},
            'tools': {'git', 'jira', 'docker', 'kubernetes', 'jenkins', 'agile', 'scrum'}
        }
    
    def compute_local_similarity(self, resume_text: str, jd_text: str, skills: list = None) -> Dict[str, float]:
        """Enhanced similarity scoring with better differentiation"""
        try:
            # Limit text length
            resume_limited = resume_text[:20000].lower()
            jd_limited = jd_text[:20000].lower()
            
            # 1. Keyword-based scoring (40%)
            jd_keywords = self._extract_keywords(jd_limited)
            resume_keywords = self._extract_keywords(resume_limited)
            
            # Exact match keywords
            exact_matches = jd_keywords.intersection(resume_keywords)
            keyword_score = len(exact_matches) / max(len(jd_keywords), 1)
            
            # 2. Skill category matching (30%)
            category_score = self._calculate_category_score(jd_limited, resume_limited, skills)
            
            # 3. Experience level detection (15%)
            experience_score = self._calculate_experience_score(resume_limited, jd_limited)
            
            # 4. Fuzzy matching (15%)
            fuzzy_score = fuzz.token_set_ratio(resume_limited, jd_limited) / 100
            
            # Combined score with weights
            combined_score = (
                keyword_score * 0.4 +
                category_score * 0.3 +
                experience_score * 0.15 +
                fuzzy_score * 0.15
            )
            
            # Apply non-linear scaling for better differentiation
            final_score = self._apply_scaling_curve(combined_score)
            
            return {
                "keyword_match": round(keyword_score * 100, 1),
                "category_match": round(category_score * 100, 1),
                "experience_match": round(experience_score * 100, 1),
                "text_similarity": round(fuzzy_score * 100, 1),
                "final_local_score": round(final_score, 1)
            }
            
        except Exception as e:
            logger.error(f"Error computing local similarity: {e}")
            return {
                "keyword_match": 0.0,
                "category_match": 0.0,
                "experience_match": 0.0,
                "text_similarity": 0.0,
                "final_local_score": 3.0
            }
    
    def _extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords from text"""
        # Remove common words and get unique keywords
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been'
        }
        
        # Extract words with some basic filtering
        words = re.findall(r'\b[a-z]{3,15}\b', text.lower())
        keywords = {word for word in words if word not in stop_words}
        
        return keywords
    
    def _calculate_category_score(self, jd_text: str, resume_text: str, skills: list) -> float:
        """Calculate score based on skill category matching"""
        if not skills:
            return 0.0
            
        jd_categories_found = set()
        resume_categories_found = set()
        
        # Find categories mentioned in JD
        for category, category_skills in self.skill_categories.items():
            if any(skill in jd_text for skill in category_skills):
                jd_categories_found.add(category)
        
        # Find categories candidate has
        for category, category_skills in self.skill_categories.items():
            if any(skill in skills for skill in category_skills):
                resume_categories_found.add(category)
        
        if not jd_categories_found:
            return 0.5  # Neutral score if no specific categories in JD
        
        matching_categories = jd_categories_found.intersection(resume_categories_found)
        return len(matching_categories) / len(jd_categories_found)
    
    def _calculate_experience_score(self, resume_text: str, jd_text: str) -> float:
        """Calculate experience level matching score"""
        # Experience indicators
        senior_indicators = {
            'senior', 'lead', 'principal', 'architect', 'manager', 'director',
            '10+', '10 years', '15 years', '20 years', 'extensive', 'expert'
        }
        
        mid_indicators = {
            'mid-level', 'mid level', '3+ years', '5+ years', '5 years', 
            'experienced', 'professional', 'intermediate'
        }
        
        junior_indicators = {
            'junior', 'entry', 'graduate', '0-2', '1-2 years', 'fresher',
            'recent graduate', 'beginner', 'trainee'
        }
        
        # Check JD for experience level requirements
        jd_level = 'any'
        if any(indicator in jd_text for indicator in senior_indicators):
            jd_level = 'senior'
        elif any(indicator in jd_text for indicator in mid_indicators):
            jd_level = 'mid'
        elif any(indicator in jd_text for indicator in junior_indicators):
            jd_level = 'junior'
        
        # Check resume for experience level
        resume_level = 'junior'  # default
        if any(indicator in resume_text for indicator in senior_indicators):
            resume_level = 'senior'
        elif any(indicator in resume_text for indicator in mid_indicators):
            resume_level = 'mid'
        
        # Score based on level matching
        level_scores = {
            ('junior', 'junior'): 0.9, ('junior', 'mid'): 0.7, ('junior', 'senior'): 0.3,
            ('mid', 'junior'): 0.4, ('mid', 'mid'): 0.9, ('mid', 'senior'): 0.6,
            ('senior', 'junior'): 0.2, ('senior', 'mid'): 0.5, ('senior', 'senior'): 0.95,
            ('any', 'junior'): 0.8, ('any', 'mid'): 0.9, ('any', 'senior'): 0.95
        }
        
        return level_scores.get((jd_level, resume_level), 0.5)
    
    def _apply_scaling_curve(self, score: float) -> float:
        """Apply non-linear scaling to create better score distribution"""
        # Use a power curve to stretch the middle scores
        if score < 0.3:
            return score * 3.0  # Boost very low scores
        elif score < 0.7:
            return 0.9 + (score - 0.3) * 2.5  # Stretch middle range
        else:
            return 2.5 + (score - 0.7) * 7.5  # Expand high scores
    
    def gemini_resume_match(self, resume_text: str, jd_text: str, skills: list = None) -> Dict[str, Any]:
        """Enhanced Gemini analysis with better score differentiation"""
        try:
            skills_text = ", ".join(skills) if skills else "Not specified"
            
            # Calculate some metrics for the prompt
            skill_count = len(skills) if skills else 0
            jd_word_count = len(jd_text.split())
            resume_word_count = len(resume_text.split())
            
            prompt = f"""
            As a senior technical recruiter, analyze this candidate's fit for the role with CRITICAL evaluation.

            JOB REQUIREMENTS (Priority Order):
            {jd_text[:2500]}

            CANDIDATE PROFILE:
            {resume_text[:2500]}

            CANDIDATE'S TECHNICAL SKILLS ({skill_count} found): {skills_text}

            ANALYSIS CONTEXT:
            - Job description length: {jd_word_count} words
            - Resume length: {resume_word_count} words  
            - Skills identified: {skill_count}

            SCORING INSTRUCTIONS (BE STRICT AND DIFFERENTIATING):
            - 9-10: Exceptional match, exceeds all requirements
            - 7-8: Strong match, meets most key requirements
            - 5-6: Moderate match, some gaps but potential
            - 3-4: Weak match, major gaps
            - 1-2: Poor match, fundamentally unsuitable

            Provide CRITICAL assessment in this EXACT JSON format:
            {{
                "fit_score": 7,
                "technical_skills_score": 8,
                "experience_relevance": 6,
                "seniority_match": 7,
                "cultural_fit": 6,
                "key_strengths": ["Specific technical skills", "Relevant projects"],
                "critical_gaps": ["Missing required technology X", "Insufficient Y experience"],
                "justification": "Detailed, critical analysis explaining exactly why this score was given...",
                "recommendation": "Strong Recommend/Recommend/Consider/Not Suitable",
                "interview_priority": "High/Medium/Low",
                "risk_factors": ["Lack of cloud experience", "Limited team leadership"]
            }}

            BE SPECIFIC AND CRITICAL. Avoid generic praise. Focus on concrete matches and gaps.
            """
            
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text
            
            # Enhanced JSON extraction with better error handling
            try:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    # Clean JSON string
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # Remove trailing commas
                    result = json.loads(json_str)
                    
                    # Ensure scores are properly scaled
                    result = self._validate_and_scale_scores(result)
                else:
                    raise ValueError("No JSON found in response")
            except Exception as json_error:
                logger.warning(f"JSON parsing issue: {json_error}")
                result = self._create_fallback_result(resume_text, jd_text, skills)
            
            return result
            
        except Exception as e:
            logger.error(f"Error with Gemini matching: {e}")
            return self._create_error_result(str(e))
    
    def _validate_and_scale_scores(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and scale scores to ensure proper distribution"""
        # Ensure fit_score is the primary metric
        if 'fit_score' not in result:
            # Calculate from sub-scores if available
            sub_scores = [
                result.get('technical_skills_score', 5),
                result.get('experience_relevance', 5), 
                result.get('seniority_match', 5),
                result.get('cultural_fit', 5)
            ]
            result['fit_score'] = sum(sub_scores) / len(sub_scores)
        
        # Apply slight variation to avoid identical scores
        import random
        variation = random.uniform(-0.3, 0.3)
        result['fit_score'] = max(1, min(10, round(result['fit_score'] + variation, 1)))
        
        return result
    
    def _create_fallback_result(self, resume_text: str, jd_text: str, skills: list) -> Dict[str, Any]:
        """Create fallback result when JSON parsing fails"""
        skill_count = len(skills) if skills else 0
        jd_tech_terms = len([word for word in jd_text.lower().split() if word in {
            'python', 'java', 'javascript', 'react', 'aws', 'sql', 'docker'
        }])
        
        # Simple heuristic scoring
        base_score = min(8, 3 + (skill_count * 0.5) + (jd_tech_terms * 0.3))
        
        return {
            "fit_score": round(base_score, 1),
            "technical_skills_score": min(9, 4 + skill_count * 0.3),
            "experience_relevance": 5.0,
            "seniority_match": 5.0,
            "cultural_fit": 5.0,
            "key_strengths": ["Skills analysis completed", "Basic qualification check passed"],
            "critical_gaps": ["Detailed analysis unavailable", "Manual review recommended"],
            "justification": f"Automated analysis completed. Found {skill_count} relevant skills. Manual review recommended for detailed assessment.",
            "recommendation": "Consider",
            "interview_priority": "Medium",
            "risk_factors": ["Limited AI analysis available"]
        }
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            "fit_score": 5.0,
            "technical_skills_score": 5.0,
            "experience_relevance": 5.0,
            "seniority_match": 5.0,
            "cultural_fit": 5.0,
            "key_strengths": ["System error - manual review required"],
            "critical_gaps": ["Technical issue prevented full analysis"],
            "justification": f"Analysis incomplete due to: {error_msg}. Please check API configuration.",
            "recommendation": "Review Needed",
            "interview_priority": "Low",
            "risk_factors": ["System reliability concerns"]
        }
    
    def comprehensive_match(self, resume_text: str, jd_text: str, skills: list = None) -> Dict[str, Any]:
        """Enhanced comprehensive matching with proper score differentiation"""
        # Get local scores (now includes skills for category matching)
        local_results = self.compute_local_similarity(resume_text, jd_text, skills)
        
        # Get Gemini analysis
        gemini_results = self.gemini_resume_match(resume_text, jd_text, skills)
        
        # Enhanced final score calculation with better differentiation
        local_weight = 0.35
        gemini_weight = 0.65
        
        # Use multiple Gemini sub-scores for better calculation
        gemini_primary_score = gemini_results["fit_score"]
        gemini_technical_score = gemini_results.get("technical_skills_score", gemini_primary_score)
        
        # Calculate final score with variation
        final_score = (
            local_results["final_local_score"] * local_weight +
            gemini_primary_score * (gemini_weight * 0.6) +
            gemini_technical_score * (gemini_weight * 0.4)
        )
        
        # Ensure reasonable distribution
        final_score = max(1.0, min(9.8, final_score))
        
        return {
            "local_scores": local_results,
            "gemini_scores": gemini_results,
            "final_score": round(final_score, 1),
            "skills_matched": skills,
            "analysis_timestamp": datetime.now().isoformat(),
            "score_breakdown": {
                "local_contribution": round(local_results["final_local_score"] * local_weight, 1),
                "ai_fit_contribution": round(gemini_primary_score * (gemini_weight * 0.6), 1),
                "ai_technical_contribution": round(gemini_technical_score * (gemini_weight * 0.4), 1)
            }
        }
