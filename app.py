import streamlit as st
import os
import tempfile
from datetime import datetime
import pandas as pd

from config import config
from database import DatabaseManager
from file_processor import FileProcessor
from matching_engine import MatchingEngine
from utils import save_uploaded_file, validate_file, display_score_with_color, format_justification

# Page configuration
st.set_page_config(
    page_title="Smart Resume Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .strength-item {
        color: #00aa00;
        margin: 0.5rem 0;
    }
    .gap-item {
        color: #ff4b4b;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class ResumeScreenerApp:
    def __init__(self):
        self.db = DatabaseManager(config.DATABASE_PATH)
        self.file_processor = FileProcessor()
        self.gemini_api_key = None
        self.matching_engine = None
        
    def initialize_gemini(self):
        """Initialize Gemini API"""
        if not self.gemini_api_key:
            if 'GEMINI_API_KEY' in st.secrets:
                self.gemini_api_key = st.secrets['GEMINI_API_KEY']
            else:
                self.gemini_api_key = st.sidebar.text_input("🔑 Enter Gemini API Key:", type="password")
        
        if self.gemini_api_key:
            try:
                self.matching_engine = MatchingEngine(self.gemini_api_key)
                return True
            except Exception as e:
                st.error(f"Error initializing Gemini: {e}")
                return False
        return False
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("🎯 Navigation")
        app_mode = st.sidebar.selectbox(
            "Choose Mode",
            ["🏠 Home", "📊 Analyze Resume", "👥 View Candidates", "📈 Results History"]
        )
        return app_mode
    
    def render_home(self):
        """Render home page"""
        st.markdown('<div class="main-header">🚀 Smart Resume Screener</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)
            
            st.markdown("""
            ### Welcome to Smart Resume Screener!
            
            This AI-powered tool helps recruiters and hiring managers:
            
            - **📄 Parse resumes** in multiple formats (PDF, DOCX, TXT)
            - **🔍 Extract skills and experience** automatically
            - **🎯 Match candidates** with job descriptions using AI
            - **📊 Generate comprehensive scores** with detailed justifications
            - **💾 Store and manage** candidate data
            
            ### How to Use:
            1. Navigate to **"Analyze Resume"** in the sidebar
            2. Upload a resume and job description
            3. Enter your Gemini API key
            4. Get instant AI-powered analysis
            
            ### Supported Features:
            - Multi-format file support
            - Local NLP + Gemini AI matching
            - Skills extraction
            - Candidate database
            - Results history
            """)
    
    def render_analyze_resume(self):
        """Render resume analysis page"""
        st.header("📊 Resume Analysis")
        
        if not self.initialize_gemini():
            st.warning("Please enter your Gemini API key in the sidebar to enable AI analysis.")
            return
        
        # File upload section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Upload Resume")
            resume_file = st.file_uploader(
                "Choose resume file",
                type=['pdf', 'docx', 'txt'],
                key="resume"
            )
            
            if resume_file:
                is_valid, message = validate_file(resume_file)
                if is_valid:
                    st.success(f"✅ {resume_file.name} - {message}")
                else:
                    st.error(f"❌ {message}")
        
        with col2:
            st.subheader("💼 Upload Job Description")
            jd_file = st.file_uploader(
                "Choose job description file",
                type=['pdf', 'txt'],
                key="jd"
            )
            
            if jd_file:
                is_valid, message = validate_file(jd_file)
                if is_valid:
                    st.success(f"✅ {jd_file.name} - {message}")
                else:
                    st.error(f"❌ {message}")
        
        # Text input alternative for JD
        st.subheader("📝 Or Enter Job Description Text")
        jd_text_input = st.text_area(
            "Paste job description here:",
            height=150,
            placeholder="Enter the job description text directly..."
        )
        
        # Analyze button
        if st.button("🚀 Analyze Match", type="primary", use_container_width=True):
            if not resume_file:
                st.error("Please upload a resume file.")
                return
            
            if not jd_file and not jd_text_input:
                st.error("Please upload a job description file or enter the text.")
                return
            
            with st.spinner("🤖 Analyzing resume and job description..."):
                try:
                    # Process resume
                    resume_path, resume_filename = save_uploaded_file(resume_file)
                    resume_text = self.file_processor.process_file(resume_path)
                    
                    # Process job description
                    if jd_file:
                        jd_path, jd_filename = save_uploaded_file(jd_file)
                        jd_text = self.file_processor.process_file(jd_path)
                    else:
                        jd_text = jd_text_input
                        jd_filename = "manual_input.txt"
                    
                    # Extract skills
                    skills = self.file_processor.extract_skills(resume_text)
                    
                    # Perform matching
                    results = self.matching_engine.comprehensive_match(resume_text, jd_text, skills)
                    
                    # Save candidate
                    candidate_data = {
                        'candidate_name': resume_filename,
                        'resume_path': resume_path,
                        'resume_text': resume_text[:1000],  # Store first 1000 chars
                        'skills': skills,
                        'experience': self.file_processor.extract_experience(resume_text),
                        'education': ""  # Can be enhanced
                    }
                    self.db.save_candidate(candidate_data)
                    
                    # Save results
                    result_data = {
                        'resume_name': resume_filename,
                        'resume_text': resume_text[:1000],
                        'jd_name': jd_filename,
                        'jd_text': jd_text[:1000],
                        'local_score': results['local_scores']['final_local_score'],
                        'gemini_score': results['gemini_scores']['fit_score'],
                        'final_score': results['final_score'],
                        'justification': results['gemini_scores']['justification'],
                        'skills_extracted': skills
                    }
                    self.db.save_screening_result(result_data)
                    
                    # Display results
                    self.display_results(results, skills)
                    
                    # Cleanup temp files
                    if resume_path and os.path.exists(resume_path):
                        os.unlink(resume_path)
                    if 'jd_path' in locals() and jd_path and os.path.exists(jd_path):
                        os.unlink(jd_path)
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
    
    def display_results(self, results: dict, skills: list):
        """Display analysis results"""
        st.header("🎯 Analysis Results")
        
        # Score cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            local_score = results['local_scores']['final_local_score']
            color_icon = display_score_with_color(local_score)
            st.metric(
                label="🧩 Local NLP Score",
                value=f"{local_score}/10",
                delta=f"{color_icon} Basic Match"
            )
        
        with col2:
            gemini_score = results['gemini_scores']['fit_score']
            color_icon = display_score_with_color(gemini_score)
            st.metric(
                label="🤖 Gemini AI Score",
                value=f"{gemini_score}/10",
                delta=f"{color_icon} AI Analysis"
            )
        
        with col3:
            final_score = results['final_score']
            color_icon = display_score_with_color(final_score)
            st.metric(
                label="🎯 Final Score",
                value=f"{final_score}/10",
                delta=f"{color_icon} Combined"
            )
        
        with col4:
            recommendation = results['gemini_scores']['recommendation']
            st.metric(
                label="📋 Recommendation",
                value=recommendation,
                delta="AI Suggestion"
            )
        
        # Detailed scores
        with st.expander("📊 Detailed Scores", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Local NLP Scores")
                local_scores = results['local_scores']
                st.write(f"**SpaCy Similarity:** {local_scores['spacy_similarity']:.3f}")
                st.write(f"**Transformer Similarity:** {local_scores['transformer_similarity']:.3f}")
                st.write(f"**Fuzzy Match:** {local_scores['fuzzy_match']}%")
            
            with col2:
                st.subheader("Extracted Skills")
                if skills:
                    st.write(", ".join(skills))
                else:
                    st.write("No skills detected")
        
        # Strengths and Gaps
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Strengths")
            strengths = results['gemini_scores'].get('strengths', [])
            for strength in strengths:
                st.markdown(f'<div class="strength-item">✓ {strength}</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("⚠️ Gaps & Concerns")
            gaps = results['gemini_scores'].get('gaps', [])
            for gap in gaps:
                st.markdown(f'<div class="gap-item">✗ {gap}</div>', unsafe_allow_html=True)
        
        # Justification
        st.subheader("📋 Detailed Analysis")
        justification = results['gemini_scores']['justification']
        st.info(format_justification(justification))
    
    def render_candidates(self):
        """Render candidates database page"""
        st.header("👥 Candidate Database")
        
        candidates = self.db.get_candidates()
        
        if not candidates:
            st.info("No candidates in database yet. Upload resumes to see them here.")
            return
        
        # Display candidates in a table
        candidate_data = []
        for candidate in candidates:
            candidate_data.append({
                'Name': candidate['candidate_name'],
                'Skills': ', '.join(candidate.get('skills', [])[:3]) if candidate.get('skills') else 'Not extracted',
                'Experience': candidate.get('experience', '')[:100] + '...' if candidate.get('experience') else 'Not extracted',
                'Added': candidate['created_at'][:10] if candidate['created_at'] else 'Unknown'
            })
        
        df = pd.DataFrame(candidate_data)
        st.dataframe(df, use_container_width=True)
        
        # Candidate details
        st.subheader("Candidate Details")
        candidate_names = [c['candidate_name'] for c in candidates]
        selected_candidate = st.selectbox("Select candidate to view details:", candidate_names)
        
        if selected_candidate:
            candidate = next((c for c in candidates if c['candidate_name'] == selected_candidate), None)
            if candidate:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Skills:**")
                    if candidate.get('skills'):
                        for skill in candidate['skills']:
                            st.write(f"- {skill}")
                    else:
                        st.write("No skills extracted")
                
                with col2:
                    st.write("**Experience Summary:**")
                    st.write(candidate.get('experience', 'Not available'))
    
    def render_results_history(self):
        """Render results history page"""
        st.header("📈 Screening History")
        
        results = self.db.get_all_results()
        
        if not results:
            st.info("No screening results yet. Analyze some resumes to see history here.")
            return
        
        # Summary statistics
        total_analyses = len(results)
        avg_score = sum(r['final_score'] for r in results) / total_analyses if total_analyses > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Analyses", total_analyses)
        col2.metric("Average Score", f"{avg_score:.2f}/10")
        col3.metric("Last Analysis", results[0]['created_at'][:10] if results else "N/A")
        
        # Results table
        st.subheader("Recent Analyses")
        results_data = []
        for result in results[:10]:  # Show last 10
            results_data.append({
                'Resume': result['resume_name'],
                'Job Description': result['jd_name'],
                'Final Score': result['final_score'],
                'Local Score': result['local_score'],
                'Gemini Score': result['gemini_score'],
                'Date': result['created_at'][:10]
            })
        
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True)
        
        # Detailed view
        st.subheader("Detailed View")
        analysis_ids = [f"{r['resume_name']} vs {r['jd_name']} ({r['created_at'][:10]})" for r in results]
        selected_analysis = st.selectbox("Select analysis to view details:", analysis_ids)
        
        if selected_analysis:
            selected_index = analysis_ids.index(selected_analysis)
            result = results[selected_index]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Scores:**")
                st.write(f"Final Score: **{result['final_score']}/10**")
                st.write(f"Local NLP Score: {result['local_score']}/10")
                st.write(f"Gemini AI Score: {result['gemini_score']}/10")
            
            with col2:
                st.write("**Extracted Skills:**")
                skills = result.get('skills_extracted', [])
                if skills:
                    for skill in skills[:5]:
                        st.write(f"- {skill}")
                    if len(skills) > 5:
                        st.write(f"... and {len(skills) - 5} more")
                else:
                    st.write("No skills recorded")
            
            st.write("**Justification:**")
            st.info(result['justification'])
    
    def run(self):
        """Main application runner"""
        app_mode = self.render_sidebar()
        
        if app_mode == "🏠 Home":
            self.render_home()
        elif app_mode == "📊 Analyze Resume":
            self.render_analyze_resume()
        elif app_mode == "👥 View Candidates":
            self.render_candidates()
        elif app_mode == "📈 Results History":
            self.render_results_history()

# Run the application
if __name__ == "__main__":
    app = ResumeScreenerApp()
    app.run()