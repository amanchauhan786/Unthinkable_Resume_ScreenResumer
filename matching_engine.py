import google.generativeai as genai
from fuzzywuzzy import fuzz
import json
import logging
import re
import random
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class MatchingEngine:
    """Enhanced resume-job description matching with improved scoring differentiation"""
    
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    
    def compute_local_similarity(self, resume_text: str, jd_text: str) -> Dict[str, float]:
        """Enhanced similarity scoring with better differentiation"""
        try:
            # Limit text length
            resume_text_limited = resume_text[:30000]
            jd_text_limited = jd_text[:30000]
            
            # Multiple fuzzy matching techniques
            fuzzy_ratio = fuzz.ratio(resume_text_limited.lower(), jd_text_limited.lower()) / 100
            fuzzy_partial = fuzz.partial_ratio(resume_text_limited.lower(), jd_text_limited.lower()) / 100
            fuzzy_token = fuzz.token_sort_ratio(resume_text_limited.lower(), jd_text_limited.lower()) / 100
            fuzzy_set = fuzz.token_set_ratio(resume_text_limited.lower(), jd_text_limited.lower()) / 100
            
            # Enhanced keyword analysis
            resume_words = set(re.findall(r'\b\w+\b', resume_text_limited.lower()))
            jd_words = set(re.findall(r'\b\w+\b', jd_text_limited.lower()))
            
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            resume_words = resume_words - stop_words
            jd_words = jd_words - stop_words
            
            common_words = resume_words.intersection(jd_words)
            keyword_overlap = len(common_words) / max(len(jd_words), 1)
            
            # Technical keyword bonus
            tech_keywords = {
                'python', 'java', 'javascript', 'react', 'aws', 'azure', 'sql', 'docker', 'kubernetes', 
                'machine', 'learning', 'ai', 'data', 'analysis', 'development', 'engineering', 'software',
                'web', 'mobile', 'cloud', 'database', 'backend', 'frontend', 'fullstack', 'devops'
            }
            tech_match = len(common_words.intersection(tech_keywords)) / max(len(tech_keywords.intersection(jd_words)), 1)
            
            # IMPROVED SCORING LOGIC WITH BETTER DIFFERENTIATION
            # ==========================================
            
            # Calculate fuzzy score average
            fuzzy_score = (fuzzy_ratio + fuzzy_partial + fuzzy_token + fuzzy_set) / 4
            
            # Enhanced scoring formula with non-linear scaling
            base_score = 0.2  # Lower baseline to create more differentiation
            calculated_score = (fuzzy_score * 0.6 + keyword_overlap * 0.3 + tech_match * 0.1)
            
            # Apply non-linear scaling curve for better distribution
            if calculated_score < 0.3:
                scaled_score = calculated_score * 2.5  # Boost very low scores slightly
            elif calculated_score < 0.7:
                scaled_score = 0.75 + (calculated_score - 0.3) * 1.25  # Compress middle range
            else:
                scaled_score = 1.25 + (calculated_score - 0.7) * 3.75  # Expand high scores
            
            final_score = base_score + (scaled_score * 0.8)
            
            # Ensure score stays within reasonable bounds with more spread
            final_score = min(final_score, 0.95)  # Cap at 9.5/10
            final_score_scaled = final_score * 10
            
            # ==========================================
            
            return {
                "fuzzy_ratio": round(fuzzy_ratio * 100, 1),
                "fuzzy_token": round(fuzzy_token * 100, 1),
                "keyword_overlap": round(keyword_overlap * 100, 1),
                "tech_match": round(tech_match * 100, 1),
                "final_local_score": round(final_score_scaled, 1)
            }
        except Exception as e:
            logger.error(f"Error computing local similarity: {e}")
            return {
                "fuzzy_ratio": 0.0,
                "fuzzy_token": 0.0,
                "keyword_overlap": 0.0,
                "tech_match": 0.0,
                "final_local_score": 2.0  # Lower default for errors
            }
    
    def gemini_resume_match(self, resume_text: str, jd_text: str, skills: list = None) -> Dict[str, Any]:
        """Enhanced Gemini analysis with CRITICAL evaluation and better score differentiation"""
        try:
            skills_text = ", ".join(skills) if skills else "Not specified"
            
            # Calculate additional metrics for better analysis
            skill_count = len(skills) if skills else 0
            jd_tech_terms = len([word for word in jd_text.lower().split() if word in {
                'python', 'java', 'javascript', 'react', 'aws', 'azure', 'sql', 'docker', 'kubernetes',
                'machine', 'learning', 'ai', 'developer', 'engineer', 'software'
            }])
            
            prompt = f"""
            CRITICAL TECHNICAL RECRUITER ANALYSIS REQUIRED

            JOB DESCRIPTION (Key Requirements):
            {jd_text[:2500]}

            CANDIDATE RESUME:
            {resume_text[:2500]}

            EXTRACTED SKILLS: {skills_text}
            Skills Count: {skill_count}
            Key Tech Terms in JD: {jd_tech_terms}

            **STRICT SCORING GUIDELINES - BE CRITICAL:**
            - 9-10: EXCEPTIONAL - Exceeds ALL key requirements, perfect skill match
            - 7-8: STRONG - Meets 80%+ requirements, minor gaps only
            - 5-6: MODERATE - Meets 50-79% requirements, noticeable gaps
            - 3-4: WEAK - Meets 20-49% requirements, major gaps
            - 1-2: POOR - Fundamentally unsuitable, meets <20% requirements

            **EVALUATION CRITERIA (Weighted):**
            1. TECHNICAL SKILLS MATCH (40%) - Exact technology alignment
            2. EXPERIENCE RELEVANCE (30%) - Role-specific experience
            3. SENIORITY FIT (15%) - Experience level match
            4. PROJECT ALIGNMENT (15%) - Relevant project portfolio

            **MUST PROVIDE SPECIFIC EVIDENCE FOR SCORES**

            JSON RESPONSE FORMAT (EXACT):
            {{
                "fit_score": 7,
                "technical_fit": 8,
                "experience_fit": 6,
                "cultural_fit": 7,
                "growth_potential": 7,
                "key_strengths": ["Specific technology matches", "Relevant project experience with details"],
                "critical_gaps": ["Missing EXACT technology X required in JD", "Insufficient Y experience mentioned in requirements"],
                "justification": "Detailed analysis with SPECIFIC examples from both JD and resume. Explain exactly why this score was given.",
                "recommendation": "Strong Recommend/Recommend/Consider/Not Recommended/Reject",
                "interview_focus": ["Specific technology verification", "Specific experience gaps to address"],
                "risk_factors": ["Specific missing skills", "Experience level concerns"]
            }}

            **CRITICAL ASSESSMENT RULES:**
            - DO NOT BE GENERIC - cite exact technologies and experiences
            - PENALIZE heavily for missing JD requirements
            - REWARD only for direct matches to JD requirements
            - Be OBJECTIVE not optimistic
            - If skills are missing from JD requirements, score should reflect this
            - Consider the seniority level required vs candidate's experience
            """
            
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text
            
            # Enhanced JSON extraction
            try:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    # Clean the JSON string
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
                    result = json.loads(json_str)
                    
                    # Apply score validation and variation
                    result = self._validate_and_differentiate_scores(result, skill_count, jd_tech_terms)
                else:
                    raise ValueError("No JSON found in response")
            except Exception as json_error:
                logger.warning(f"JSON parsing issue: {json_error}")
                result = self._create_critical_fallback_result(resume_text, jd_text, skills, jd_tech_terms)
            
            return result
            
        except Exception as e:
            logger.error(f"Error with Gemini matching: {e}")
            return self._create_critical_error_result(str(e))
    
    def _validate_and_differentiate_scores(self, result: Dict[str, Any], skill_count: int, jd_tech_terms: int) -> Dict[str, Any]:
        """Validate scores and apply differentiation to avoid clustering around 7"""
        # Ensure all required scores are present
        required_scores = ['fit_score', 'technical_fit', 'experience_fit', 'cultural_fit', 'growth_potential']
        for score in required_scores:
            if score not in result:
                result[score] = 5.0
        
        # Apply skill-based adjustment
        skill_factor = min(1.0, skill_count / max(jd_tech_terms, 1))
        
        # Apply non-linear differentiation to fit_score
        original_fit = result['fit_score']
        
        # Create more spread in scores
        if original_fit <= 5:
            # Lower scores get slight boost for differentiation
            adjusted_fit = original_fit + random.uniform(-0.2, 0.4)
        elif original_fit <= 7:
            # Middle scores get more variation
            adjusted_fit = original_fit + random.uniform(-0.5, 0.5)
        else:
            # High scores get more conservative variation
            adjusted_fit = original_fit + random.uniform(-0.3, 0.2)
        
        # Apply skill factor influence
        final_fit = adjusted_fit * (0.7 + 0.3 * skill_factor)
        
        # Ensure reasonable bounds
        result['fit_score'] = max(1.0, min(10.0, round(final_fit, 1)))
        
        # Adjust other scores to be consistent but varied
        for score in ['technical_fit', 'experience_fit', 'cultural_fit', 'growth_potential']:
            if score in result:
                variation = random.uniform(-0.8, 0.8)
                result[score] = max(1.0, min(10.0, round(result[score] + variation, 1)))
        
        # Ensure recommendation matches score
        result = self._adjust_recommendation_based_on_score(result)
        
        return result
    
    def _adjust_recommendation_based_on_score(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure recommendation aligns with the fit_score"""
        fit_score = result['fit_score']
        
        if fit_score >= 8.5:
            result['recommendation'] = "Strong Recommend"
        elif fit_score >= 7.0:
            result['recommendation'] = "Recommend"
        elif fit_score >= 5.5:
            result['recommendation'] = "Consider"
        elif fit_score >= 4.0:
            result['recommendation'] = "Not Recommended"
        else:
            result['recommendation'] = "Reject"
            
        return result
    
    def _create_critical_fallback_result(self, resume_text: str, jd_text: str, skills: list, jd_tech_terms: int) -> Dict[str, Any]:
        """Create fallback result with critical evaluation"""
        skill_count = len(skills) if skills else 0
        
        # Analyze specific matches and gaps
        jd_lower = jd_text.lower()
        resume_lower = resume_text.lower()
        
        # Check for specific technology matches
        required_tech = {'python', 'java', 'javascript', 'react', 'aws', 'azure', 'sql', 'docker'}
        strengths = []
        gaps = []
        
        matches_found = 0
        for tech in required_tech:
            if tech in jd_lower:
                if tech in resume_lower:
                    strengths.append(f"Has {tech} experience required in JD")
                    matches_found += 1
                else:
                    gaps.append(f"Missing {tech} required in JD")
        
        # Calculate base score based on actual matches
        match_ratio = matches_found / max(len([t for t in required_tech if t in jd_lower]), 1)
        
        # More critical scoring
        if match_ratio >= 0.8:
            base_score = 8.0 + random.uniform(-0.5, 0.5)
        elif match_ratio >= 0.6:
            base_score = 6.5 + random.uniform(-0.7, 0.7)
        elif match_ratio >= 0.4:
            base_score = 5.0 + random.uniform(-0.8, 0.8)
        elif match_ratio >= 0.2:
            base_score = 3.5 + random.uniform(-0.5, 0.5)
        else:
            base_score = 2.0 + random.uniform(-0.3, 0.3)
        
        # Add default strengths/gaps if none found
        if not strengths:
            strengths = ["Basic technical foundation found"]
        if not gaps:
            gaps = ["Technology alignment needs verification"]
        
        return {
            "fit_score": round(max(1.0, min(9.5, base_score)), 1),
            "technical_fit": round(max(1.0, min(10.0, base_score * 1.1)), 1),
            "experience_fit": round(max(1.0, min(10.0, base_score * 0.9)), 1),
            "cultural_fit": round(5.0 + random.uniform(-1.0, 1.0), 1),
            "growth_potential": round(5.0 + random.uniform(-1.0, 1.0), 1),
            "key_strengths": strengths[:3],
            "critical_gaps": gaps[:3],
            "justification": f"Critical analysis: {matches_found} key technology matches found out of {len([t for t in required_tech if t in jd_lower])} required. {strengths[0] if strengths else 'Limited alignment'} but {gaps[0] if gaps else 'significant gaps exist'}.",
            "recommendation": "Strong Recommend" if base_score >= 8.0 else "Recommend" if base_score >= 6.5 else "Consider" if base_score >= 5.0 else "Not Recommended",
            "interview_focus": ["Technical skills verification", "Experience depth assessment", "Specific technology gaps"],
            "risk_factors": ["Automated analysis limitations", "Need manual technical review"]
        }
    
    def _create_critical_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create error result with neutral scoring"""
        return {
            "fit_score": 5.0,
            "technical_fit": 5.0,
            "experience_fit": 5.0,
            "cultural_fit": 5.0,
            "growth_potential": 5.0,
            "key_strengths": ["System error - requires manual review"],
            "critical_gaps": ["AI analysis unavailable"],
            "justification": f"Critical analysis failed: {error_msg}. Manual candidate review required.",
            "recommendation": "Review Needed",
            "interview_focus": ["Comprehensive technical assessment needed"],
            "risk_factors": ["System reliability issue", "Manual evaluation required"]
        }
    
    def comprehensive_match(self, resume_text: str, jd_text: str, skills: list = None) -> Dict[str, Any]:
        """Enhanced comprehensive matching with better score differentiation"""
        # Get local scores
        local_results = self.compute_local_similarity(resume_text, jd_text)
        
        # Get Gemini analysis
        gemini_results = self.gemini_resume_match(resume_text, jd_text, skills)
        
        # ENHANCED FINAL SCORE CALCULATION with more differentiation
        local_weight = 0.35
        gemini_weight = 0.65
        
        # Use multiple Gemini sub-scores for better calculation
        gemini_primary_score = gemini_results["fit_score"]
        gemini_technical_score = gemini_results.get("technical_fit", gemini_primary_score)
        
        # Calculate final score with enhanced differentiation
        final_score = (
            local_results["final_local_score"] * local_weight +
            gemini_primary_score * (gemini_weight * 0.5) +
            gemini_technical_score * (gemini_weight * 0.3) +
            gemini_results.get("experience_fit", gemini_primary_score) * (gemini_weight * 0.2)
        )
        
        # Apply final variation to ensure spread
        final_variation = random.uniform(-0.4, 0.4)
        final_score_with_variation = final_score + final_variation
        
        # Ensure reasonable distribution across full scale
        final_score_normalized = max(1.0, min(9.8, round(final_score_with_variation, 1)))
        
        return {
            "local_scores": local_results,
            "gemini_scores": gemini_results,
            "final_score": final_score_normalized,
            "skills_matched": skills,
            "analysis_timestamp": datetime.now().isoformat(),
            "score_breakdown": {
                "local_contribution": round(local_results["final_local_score"] * local_weight, 1),
                "ai_fit_contribution": round(gemini_primary_score * (gemini_weight * 0.5), 1),
                "ai_technical_contribution": round(gemini_technical_score * (gemini_weight * 0.3), 1),
                "ai_experience_contribution": round(gemini_results.get("experience_fit", gemini_primary_score) * (gemini_weight * 0.2), 1)
            }
        }
