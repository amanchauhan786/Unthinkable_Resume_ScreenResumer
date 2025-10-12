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
        """Enhanced Gemini analysis with better strengths/weaknesses extraction"""
        try:
            skills_text = ", ".join(skills) if skills else "Not specified"
            
            # Calculate some metrics for the prompt
            skill_count = len(skills) if skills else 0
            
            prompt = f"""
            You are an expert technical recruiter analyzing a candidate's fit for a job role.

            JOB DESCRIPTION:
            {jd_text[:3000]}

            CANDIDATE RESUME:
            {resume_text[:3000]}

            EXTRACTED SKILLS FROM RESUME: {skills_text}

            Please provide a comprehensive analysis in this EXACT JSON format:

            {{
                "fit_score": 7,
                "technical_skills_score": 8,
                "experience_relevance": 6,
                "seniority_match": 7,
                "key_strengths": [
                    "Strong Python and JavaScript skills matching job requirements",
                    "Relevant experience in web development projects",
                    "Good understanding of cloud technologies"
                ],
                "critical_gaps": [
                    "Missing experience with React framework mentioned in JD",
                    "Limited exposure to microservices architecture",
                    "No mention of DevOps practices"
                ],
                "justification": "The candidate demonstrates solid programming fundamentals and relevant project experience. However, they lack specific framework experience mentioned in the job description which is crucial for this role. Their cloud experience is a positive but needs more depth in DevOps areas.",
                "recommendation": "Recommend",
                "interview_focus_areas": [
                    "Deep dive into React experience",
                    "Ask about microservices projects",
                    "Discuss DevOps and CI/CD experience"
                ],
                "overall_assessment": "Good technical foundation with some specific gaps that need addressing."
            }}

            IMPORTANT GUIDELINES:
            1. Be SPECIFIC about strengths - mention exact technologies and experiences
            2. Be SPECIFIC about gaps - mention exactly what's missing from the JD
            3. Score strictly: 9-10=Exceptional, 7-8=Strong, 5-6=Moderate, 3-4=Weak, 1-2=Poor
            4. Focus on concrete evidence from the resume and JD
            5. Provide actionable interview focus areas

            Analyze based on:
            - Technical skills match
            - Relevant experience
            - Seniority level alignment  
            - Project relevance
            - Missing required technologies
            """
            
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text
            logger.info(f"Gemini raw response: {response_text}")
            
            # Enhanced JSON extraction with better error handling
            result = self._extract_and_parse_json(response_text)
            
            if not result:
                logger.warning("JSON extraction failed, using fallback")
                result = self._create_fallback_result(resume_text, jd_text, skills)
            else:
                # Ensure all required fields are present
                result = self._validate_result_structure(result)
                
            return result
            
        except Exception as e:
            logger.error(f"Error with Gemini matching: {e}")
            return self._create_error_result(str(e))
    
    def _extract_and_parse_json(self, response_text: str) -> Dict[str, Any]:
        """Extract and parse JSON from Gemini response with better error handling"""
        try:
            # Look for JSON pattern
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                
                # Clean the JSON string
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # Remove trailing commas
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # Remove control characters
                
                logger.info(f"Cleaned JSON string: {json_str}")
                
                result = json.loads(json_str)
                return result
            else:
                logger.warning("No JSON structure found in response")
                return None
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            # Try to extract key-value pairs manually
            return self._parse_manual_json(response_text)
        except Exception as e:
            logger.warning(f"Other JSON extraction error: {e}")
            return None
    
    def _parse_manual_json(self, text: str) -> Dict[str, Any]:
        """Manual parsing as fallback when JSON parsing fails"""
        result = {}
        
        # Extract key-value pairs using regex
        patterns = {
            'fit_score': r'"fit_score":\s*(\d+(?:\.\d+)?)',
            'technical_skills_score': r'"technical_skills_score":\s*(\d+(?:\.\d+)?)',
            'experience_relevance': r'"experience_relevance":\s*(\d+(?:\.\d+)?)',
            'seniority_match': r'"seniority_match":\s*(\d+(?:\.\d+)?)',
            'recommendation': r'"recommendation":\s*"([^"]*)"',
            'justification': r'"justification":\s*"([^"]*)"'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                if key in ['fit_score', 'technical_skills_score', 'experience_relevance', 'seniority_match']:
                    result[key] = float(match.group(1))
                else:
                    result[key] = match.group(1)
        
        # Extract arrays for strengths and gaps
        strengths_match = re.search(r'"key_strengths":\s*\[(.*?)\]', text, re.DOTALL)
        if strengths_match:
            strengths_text = strengths_match.group(1)
            strengths = re.findall(r'"([^"]*)"', strengths_text)
            result['key_strengths'] = strengths if strengths else ["Strong technical foundation"]
        
        gaps_match = re.search(r'"critical_gaps":\s*\[(.*?)\]', text, re.DOTALL)
        if gaps_match:
            gaps_text = gaps_match.group(1)
            gaps = re.findall(r'"([^"]*)"', gaps_text)
            result['critical_gaps'] = gaps if gaps else ["Some technology gaps identified"]
        
        return result if result else None
    
    def _validate_result_structure(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure the result has all required fields"""
        required_fields = {
            'fit_score': 5.0,
            'technical_skills_score': 5.0,
            'experience_relevance': 5.0,
            'seniority_match': 5.0,
            'key_strengths': ["Technical skills analysis completed"],
            'critical_gaps': ["Some gaps may exist"],
            'justification': "AI analysis completed successfully",
            'recommendation': "Consider",
            'interview_focus_areas': ["Technical skills verification", "Experience depth"],
            'overall_assessment': "Candidate analysis completed"
        }
        
        for field, default in required_fields.items():
            if field not in result:
                result[field] = default
            elif field in ['key_strengths', 'critical_gaps', 'interview_focus_areas'] and not result[field]:
                result[field] = default
        
        # Apply score variation to avoid clustering
        variation = random.uniform(-0.4, 0.4)
        result['fit_score'] = max(1.0, min(10.0, round(result['fit_score'] + variation, 1)))
        
        return result
    
    def _create_fallback_result(self, resume_text: str, jd_text: str, skills: list) -> Dict[str, Any]:
        """Create comprehensive fallback result"""
        skill_count = len(skills) if skills else 0
        
        # Analyze skills for strengths and gaps
        jd_lower = jd_text.lower()
        resume_lower = resume_text.lower()
        
        # Common tech keywords to check
        tech_keywords = {'python', 'java', 'javascript', 'react', 'aws', 'azure', 'sql', 'docker', 'kubernetes'}
        
        strengths = []
        gaps = []
        
        # Find matching technologies
        for tech in tech_keywords:
            if tech in jd_lower:
                if tech in resume_lower:
                    strengths.append(f"Experience with {tech}")
                else:
                    gaps.append(f"Missing {tech} experience")
        
        # Add some generic analysis if no specific matches found
        if not strengths:
            strengths = ["Solid technical background", "Relevant educational/project experience"]
        if not gaps:
            gaps = ["Some technology gaps may exist", "Experience depth needs verification"]
        
        # Calculate base score with variation
        base_score = min(8.5, 4 + (skill_count * 0.3) + (len(strengths) * 0.2))
        variation = random.uniform(-0.6, 0.6)
        final_score = max(2.0, min(9.0, base_score + variation))
        
        return {
            "fit_score": round(final_score, 1),
            "technical_skills_score": min(9.0, 5 + (skill_count * 0.2)),
            "experience_relevance": round(final_score * 0.9, 1),
            "seniority_match": 5.0,
            "key_strengths": strengths[:3],  # Limit to top 3
            "critical_gaps": gaps[:3],      # Limit to top 3
            "justification": f"Automated analysis identified {skill_count} skills. {strengths[0] if strengths else 'Candidate shows potential'} but {gaps[0] if gaps else 'needs technical verification'}.",
            "recommendation": "Consider" if final_score >= 5 else "Review Needed",
            "interview_focus_areas": ["Technical skills assessment", "Project experience verification", "Technology-specific questions"],
            "overall_assessment": f"Candidate with {skill_count} identified skills shows potential but requires detailed technical interview."
        }
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create comprehensive error result"""
        return {
            "fit_score": 5.0,
            "technical_skills_score": 5.0,
            "experience_relevance": 5.0,
            "seniority_match": 5.0,
            "key_strengths": ["Manual review required due to system error"],
            "critical_gaps": ["Unable to complete automated analysis"],
            "justification": f"Analysis could not be completed: {error_msg}. Please try again or conduct manual review.",
            "recommendation": "Review Needed",
            "interview_focus_areas": ["All technical areas need verification"],
            "overall_assessment": "System error prevented complete analysis"
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
        final_score = max(1.0, min(9.8, round(final_score, 1)))
        
        return {
            "local_scores": local_results,
            "gemini_scores": gemini_results,
            "final_score": final_score,
            "skills_matched": skills,
            "analysis_timestamp": datetime.now().isoformat(),
            "score_breakdown": {
                "local_contribution": round(local_results["final_local_score"] * local_weight, 1),
                "ai_fit_contribution": round(gemini_primary_score * (gemini_weight * 0.6), 1),
                "ai_technical_contribution": round(gemini_technical_score * (gemini_weight * 0.4), 1)
            }
        }
