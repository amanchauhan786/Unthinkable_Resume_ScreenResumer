import streamlit as st
import os
import tempfile
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config import config
from database import DatabaseManager
from file_processor import FileProcessor
from matching_engine import MatchingEngine
from utils import save_uploaded_file, validate_file, display_score_with_color, format_justification

# Page configuration
st.set_page_config(
    page_title="TalentScreener Pro | AI Resume Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with modern design
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        padding: 1rem;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Modern cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0,0,0,0.12);
    }
    
    .score-high { border-left: 4px solid #28a745; }
    .score-medium { border-left: 4px solid #ffc107; }
    .score-low { border-left: 4px solid #dc3545; }
    
    /* Section cards */
    .section-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 1.5rem 0;
        border: 1px solid #e9ecef;
    }
    
    /* Skill pills */
    .skill-pill {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 1rem;
        margin: 0.3rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* List items */
    .strength-item {
        color: #28a745;
        margin: 0.8rem 0;
        padding: 0.8rem;
        background: #f8fff9;
        border-left: 4px solid #28a745;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .gap-item {
        color: #dc3545;
        margin: 0.8rem 0;
        padding: 0.8rem;
        background: #fff5f5;
        border-left: 4px solid #dc3545;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Recommendation boxes */
    .recommendation-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        font-weight: 700;
        text-align: center;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .recommend-strong { 
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 2px solid #28a745;
    }
    
    .recommend-good { 
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        color: #0c5460;
        border: 2px solid #17a2b8;
    }
    
    .recommend-consider { 
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        border: 2px solid #ffc107;
    }
    
    .recommend-no { 
        background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%);
        color: #721c24;
        border: 2px solid #dc3545;
    }
    
    /* Upload areas */
    .upload-area {
        border: 2px dashed #6c757d;
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: #f0f2ff;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Badges */
    .analysis-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
        background: #e9ecef;
        color: #495057;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

class ProfessionalResumeScreener:
    def __init__(self):
        self.db = DatabaseManager(config.DATABASE_PATH)
        self.file_processor = FileProcessor()
        self.gemini_api_key = None
        self.matching_engine = None
        
    def initialize_gemini(self):
        """Initialize Gemini API with professional styling"""
        if not self.gemini_api_key:
            if 'GEMINI_API_KEY' in st.secrets:
                self.gemini_api_key = st.secrets['GEMINI_API_KEY']
            else:
                with st.sidebar:
                    st.markdown("---")
                    st.subheader("🔐 API Configuration")
                    self.gemini_api_key = st.text_input(
                        "Enter Gemini API Key:",
                        type="password",
                        help="Get your API key from Google AI Studio",
                        placeholder="Enter your Gemini API key here..."
                    )
                    if self.gemini_api_key:
                        st.success("✅ API Key Configured")
        
        if self.gemini_api_key:
            try:
                self.matching_engine = MatchingEngine(self.gemini_api_key)
                return True
            except Exception as e:
                st.error(f"❌ Error initializing Gemini: {e}")
                return False
        return False
    
    def render_sidebar(self):
        """Professional sidebar navigation"""
        with st.sidebar:
            # Logo and Title
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
            
            st.markdown("<h2 style='text-align: center; color: #667eea; margin-bottom: 0;'>TalentScreener Pro</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #6c757d; font-size: 0.9rem;'>AI-Powered Recruitment</p>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Navigation
            st.subheader("🎯 Navigation")
            app_mode = st.selectbox(
                "Choose Analysis Mode",
                ["🏠 Dashboard", "📊 Analyze Resume", "👥 Candidate Database", "📈 Analytics & History"],
                label_visibility="collapsed"
            )

            
        return app_mode
    
    def render_dashboard(self):
        """Professional dashboard view"""
        st.markdown('<div class="main-header">TalentScreener Pro</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">AI-Powered Resume Screening Platform</div>', unsafe_allow_html=True)
        
        # Feature highlights in cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h3 style='color: #667eea;'>🚀 Fast Analysis</h3>
            <p style='color: #6c757d;'>Process resumes in seconds with AI-powered screening and real-time insights</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h3 style='color: #667eea;'>🎯 Accurate Matching</h3>
            <p style='color: #6c757d;'>Advanced algorithms for precise candidate-job fit assessment</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-card">
            <h3 style='color: #667eea;'>📊 Data-Driven</h3>
            <p style='color: #6c757d;'>Comprehensive analytics and insights for hiring teams</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick start guide
        st.markdown("""
        <div class="section-card">
        <h3 style='color: #495057; border-bottom: 2px solid #667eea; padding-bottom: 0.5rem;'>🚀 Get Started</h3>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-top: 1rem;'>
            <div>
                <h4 style='color: #667eea;'>1. Configure API</h4>
                <p>Enter your Gemini API key in the sidebar</p>
            </div>
            <div>
                <h4 style='color: #667eea;'>2. Upload Documents</h4>
                <p>Add resume and job description files</p>
            </div>
            <div>
                <h4 style='color: #667eea;'>3. Analyze</h4>
                <p>Get instant AI-powered candidate evaluation</p>
            </div>
            <div>
                <h4 style='color: #667eea;'>4. Review</h4>
                <p>Explore detailed scores and recommendations</p>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Recent activity chart
        try:
            results = self.db.get_all_results()[:10]  # Last 10 analyses
            if results:
                st.markdown("""
                <div class="section-card">
                <h3 style='color: #495057; border-bottom: 2px solid #667eea; padding-bottom: 0.5rem;'>📈 Recent Activity</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a simple bar chart
                df_activity = pd.DataFrame({
                    'Analysis': [f"Analysis {i+1}" for i in range(len(results))],
                    'Score': [r['final_score'] for r in results],
                    'Date': [r['created_at'][:10] for r in results]
                })
                
                fig = px.bar(
                    df_activity, 
                    x='Analysis', 
                    y='Score',
                    title="Recent Analysis Scores",
                    color='Score',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
        except:
            pass
    
    def render_analyze_resume(self):
        """Professional resume analysis interface"""
        st.markdown('<div class="main-header">Candidate Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">AI-Powered Resume & Job Description Matching</div>', unsafe_allow_html=True)
        
        if not self.initialize_gemini():
            st.warning("🔑 Please configure your Gemini API key in the sidebar to begin analysis.")
            return
        
        # File upload section with professional styling
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📄 Candidate Resume")
            st.markdown('<div class="upload-area">', unsafe_allow_html=True)
            resume_file = st.file_uploader(
                "Upload Resume File",
                type=['pdf', 'docx', 'txt'],
                key="resume",
                label_visibility="collapsed",
                help="Supported formats: PDF, DOCX, TXT"
            )
            st.markdown("""
            <p style='color: #6c757d; text-align: center; margin: 0;'>
            📁 Drag & drop or click to upload<br>
            <small>Max file size: 10MB</small>
            </p>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if resume_file:
                is_valid, message = validate_file(resume_file)
                if is_valid:
                    st.success(f"✅ **{resume_file.name}** - Ready for analysis")
                else:
                    st.error(f"❌ {message}")
        
        with col2:
            st.markdown("### 💼 Job Description")
            st.markdown('<div class="upload-area">', unsafe_allow_html=True)
            jd_file = st.file_uploader(
                "Upload Job Description",
                type=['pdf', 'txt'],
                key="jd",
                label_visibility="collapsed",
                help="Supported formats: PDF, TXT"
            )
            st.markdown("""
            <p style='color: #6c757d; text-align: center; margin: 0;'>
            📁 Drag & drop or click to upload<br>
            <small>Max file size: 10MB</small>
            </p>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if jd_file:
                is_valid, message = validate_file(jd_file)
                if is_valid:
                    st.success(f"✅ **{jd_file.name}** - Ready for analysis")
                else:
                    st.error(f"❌ {message}")
        
        # Text input alternative
        with st.expander("📝 Or Enter Job Description Manually", expanded=False):
            jd_text_input = st.text_area(
                "Paste job description text:",
                height=150,
                placeholder="Enter the complete job description here...",
                help="You can paste the job description directly instead of uploading a file"
            )
        
        # Analysis button - centered and prominent
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_clicked = st.button(
                "🚀 Start AI Analysis", 
                type="primary", 
                use_container_width=True,
                help="Click to begin AI-powered candidate analysis"
            )
        
        if analyze_clicked:
            self._perform_analysis(resume_file, jd_file, jd_text_input)
    
    def _perform_analysis(self, resume_file, jd_file, jd_text_input):
        """Perform analysis with professional progress indicators"""
        if not resume_file:
            st.error("📄 Please upload a resume file to continue.")
            return
        
        if not jd_file and not jd_text_input:
            st.error("💼 Please upload a job description or enter the text.")
            return
        
        # Progress container
        progress_container = st.container()
        
        with progress_container:
            st.markdown("### 🔍 Analysis Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Process resume
                status_text.text("📄 Processing resume file...")
                resume_path, resume_filename = save_uploaded_file(resume_file)
                resume_text = self.file_processor.process_file(resume_path)
                progress_bar.progress(25)
                
                # Process job description
                status_text.text("💼 Processing job description...")
                if jd_file:
                    jd_path, jd_filename = save_uploaded_file(jd_file)
                    jd_text = self.file_processor.process_file(jd_path)
                else:
                    jd_text = jd_text_input
                    jd_filename = "Manual_Input"
                progress_bar.progress(50)
                
                # Extract skills
                status_text.text("🔧 Extracting skills and experience...")
                skills = self.file_processor.extract_skills(resume_text)
                progress_bar.progress(75)
                
                # Perform matching
                status_text.text("🤖 AI analyzing candidate fit...")
                results = self.matching_engine.comprehensive_match(resume_text, jd_text, skills)
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Save results
                self._save_analysis_results(resume_filename, resume_text, jd_filename, jd_text, results, skills)
                
                # Display results
                st.markdown("<br>", unsafe_allow_html=True)
                self._display_professional_results(results, skills, resume_filename, jd_filename)
                
                # Cleanup
                if resume_path and os.path.exists(resume_path):
                    os.unlink(resume_path)
                if 'jd_path' in locals() and jd_path and os.path.exists(jd_path):
                    os.unlink(jd_path)
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
    
    def _save_analysis_results(self, resume_filename, resume_text, jd_filename, jd_text, results, skills):
        """Save analysis results to database"""
        candidate_data = {
            'candidate_name': resume_filename,
            'resume_path': resume_filename,
            'resume_text': resume_text[:1000],
            'skills': skills,
            'experience': self.file_processor.extract_experience(resume_text),
            'education': self.file_processor.extract_education(resume_text)
        }
        self.db.save_candidate(candidate_data)
        
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
    
    def _display_professional_results(self, results: dict, skills: list, resume_name: str, jd_name: str):
        """Display professional results with enhanced UI"""
        st.markdown('<div class="main-header">Analysis Results</div>', unsafe_allow_html=True)
        
        # Header with analysis info
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Candidate:** {resume_name}")
            st.markdown(f"**Job Description:** {jd_name}")
        with col2:
            st.markdown(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Score cards with enhanced styling
        st.markdown("### 📊 Match Scores")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            local_score = results['local_scores']['final_local_score']
            score_class = "score-high" if local_score >= 7 else "score-medium" if local_score >= 5 else "score-low"
            st.markdown(f"""
            <div class="metric-card {score_class}">
                <h3 style='margin: 0; color: #495057;'>🧩 Text Match</h3>
                <h1 style='margin: 0.5rem 0; color: #667eea;'>{local_score}/10</h1>
                <p style='margin: 0; color: #6c757d; font-size: 0.9rem;'>Document Similarity</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            gemini_score = results['gemini_scores']['fit_score']
            score_class = "score-high" if gemini_score >= 7 else "score-medium" if gemini_score >= 5 else "score-low"
            st.markdown(f"""
            <div class="metric-card {score_class}">
                <h3 style='margin: 0; color: #495057;'>🤖 AI Assessment</h3>
                <h1 style='margin: 0.5rem 0; color: #667eea;'>{gemini_score}/10</h1>
                <p style='margin: 0; color: #6c757d; font-size: 0.9rem;'>Overall Fit Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            final_score = results['final_score']
            score_class = "score-high" if final_score >= 7 else "score-medium" if final_score >= 5 else "score-low"
            st.markdown(f"""
            <div class="metric-card {score_class}">
                <h3 style='margin: 0; color: #495057;'>🎯 Final Score</h3>
                <h1 style='margin: 0.5rem 0; color: #667eea;'>{final_score}/10</h1>
                <p style='margin: 0; color: #6c757d; font-size: 0.9rem;'>Combined Assessment</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            recommendation = results['gemini_scores']['recommendation']
            rec_class = "recommend-strong" if "strong" in recommendation.lower() else \
                       "recommend-good" if "recommend" in recommendation.lower() else \
                       "recommend-consider" if "consider" in recommendation.lower() else "recommend-no"
            st.markdown(f"""
            <div class="{rec_class}">
                <h3 style='margin: 0;'>📋 Recommendation</h3>
                <p style='margin: 0.5rem 0; font-size: 1.3rem;'>{recommendation}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed Analysis Section
        st.markdown("### 📈 Detailed Analysis")
        
        # Create a radar chart for skill categories
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Skills visualization
            if skills:
                st.markdown("#### 🔧 Technical Skills")
                # Display skills as pills
                skill_html = "".join([f'<span class="skill-pill">{skill}</span>' for skill in skills])
                st.markdown(f'<div style="margin: 1rem 0;">{skill_html}</div>', unsafe_allow_html=True)
                st.markdown(f"**Total Skills Identified:** {len(skills)}")
        
        with col2:
            # Quick stats
            st.markdown("#### 📊 Quick Stats")
            local_scores = results['local_scores']
            gemini_scores = results['gemini_scores']
            
            if 'fuzzy_ratio' in local_scores:
                st.metric("Text Similarity", f"{local_scores['fuzzy_ratio']}%")
            if 'keyword_overlap' in local_scores:
                st.metric("Keyword Match", f"{local_scores['keyword_overlap']}%")
            if 'technical_fit' in gemini_scores:
                st.metric("Technical Fit", f"{gemini_scores['technical_fit']}/10")
        
        # Strengths and Gaps in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Key Strengths")
            strengths = results['gemini_scores'].get('strengths', [])
            if strengths:
                for strength in strengths:
                    st.markdown(f'<div class="strength-item">✓ {strength}</div>', unsafe_allow_html=True)
            else:
                st.info("No specific strengths identified in this analysis.")
        
        with col2:
            st.markdown("#### ⚠️ Areas for Development")
            gaps = results['gemini_scores'].get('gaps', [])
            if gaps:
                for gap in gaps:
                    st.markdown(f'<div class="gap-item">✗ {gap}</div>', unsafe_allow_html=True)
            else:
                st.info("No major gaps identified in this analysis.")
        
        # Detailed Justification
        st.markdown("#### 📋 Comprehensive Analysis")
        justification = results['gemini_scores']['justification']
        st.markdown(f"""
        <div class="section-card">
            <p style='line-height: 1.6; color: #495057;'>{justification}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Interview focus areas
        interview_focus = results['gemini_scores'].get('interview_focus', [])
        if interview_focus:
            st.markdown("#### 🎯 Suggested Interview Focus")
            focus_html = "".join([f'<span class="analysis-badge">{focus}</span>' for focus in interview_focus])
            st.markdown(f'<div style="margin: 1rem 0;">{focus_html}</div>', unsafe_allow_html=True)
    
    def render_candidates(self):
        """Enhanced candidate database view"""
        st.markdown('<div class="main-header">Candidate Database</div>', unsafe_allow_html=True)
        
        candidates = self.db.get_candidates()
        
        if not candidates:
            st.info("👤 No candidates in database yet. Upload resumes to see them here.")
            return
        
        # Summary stats
        total_candidates = len(candidates)
        avg_skills = sum(len(c.get('skills', [])) for c in candidates) / total_candidates
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Candidates", total_candidates)
        col2.metric("Average Skills", f"{avg_skills:.1f}")
        col3.metric("Database Size", f"{total_candidates * 2}KB")  # Approximate
        
        # Enhanced candidate table
        st.markdown("### 👥 Candidate Overview")
        candidate_data = []
        for candidate in candidates:
            candidate_data.append({
                'Name': candidate['candidate_name'],
                'Skills Count': len(candidate.get('skills', [])),
                'Top Skills': ', '.join(candidate.get('skills', [])[:3]),
                'Experience': candidate.get('experience', '')[:80] + '...' if candidate.get('experience') else 'Not extracted',
                'Added': candidate['created_at'][:10] if candidate['created_at'] else 'Unknown'
            })
        
        df = pd.DataFrame(candidate_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        # Candidate details with enhanced UI
        st.markdown("### 🔍 Candidate Details")
        candidate_names = [c['candidate_name'] for c in candidates]
        selected_candidate = st.selectbox("Select candidate to view details:", candidate_names)
        
        if selected_candidate:
            candidate = next((c for c in candidates if c['candidate_name'] == selected_candidate), None)
            if candidate:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 💼 Skills Profile")
                    if candidate.get('skills'):
                        skill_html = "".join([f'<span class="skill-pill">{skill}</span>' for skill in candidate['skills']])
                        st.markdown(f'<div style="margin: 1rem 0;">{skill_html}</div>', unsafe_allow_html=True)
                        st.metric("Total Skills", len(candidate['skills']))
                    else:
                        st.info("No skills extracted for this candidate.")
                
                with col2:
                    st.markdown("#### 📈 Experience & Education")
                    st.markdown(f"**Experience:** {candidate.get('experience', 'Not available')}")
                    st.markdown(f"**Education:** {candidate.get('education', 'Not available')}")
    
    def render_results_history(self):
        """Enhanced results history with analytics"""
        st.markdown('<div class="main-header">Analytics & History</div>', unsafe_allow_html=True)
        
        results = self.db.get_all_results()
        
        if not results:
            st.info("📊 No screening results yet. Analyze some resumes to see analytics here.")
            return
        
        # Enhanced statistics
        total_analyses = len(results)
        avg_score = sum(r['final_score'] for r in results) / total_analyses
        avg_local = sum(r['local_score'] for r in results) / total_analyses
        avg_gemini = sum(r['gemini_score'] for r in results) / total_analyses
        
        # Statistics cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Analyses", total_analyses)
        col2.metric("Average Final Score", f"{avg_score:.1f}/10")
        col3.metric("Average Local Score", f"{avg_local:.1f}/10")
        col4.metric("Average AI Score", f"{avg_gemini:.1f}/10")
        
        # Score distribution chart
        st.markdown("### 📈 Score Distribution")
        score_data = pd.DataFrame({
            'Score': [r['final_score'] for r in results],
            'Type': 'Final Score'
        })
        
        fig = px.histogram(
            score_data, 
            x='Score',
            nbins=10,
            title="Distribution of Final Scores",
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent analyses table
        st.markdown("### 📋 Recent Analyses")
        results_data = []
        for result in results[:10]:
            results_data.append({
                'Resume': result['resume_name'],
                'Job Description': result['jd_name'],
                'Final Score': result['final_score'],
                'Local Score': result['local_score'],
                'Gemini Score': result['gemini_score'],
                'Date': result['created_at'][:10]
            })
        
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        # Detailed view
        st.markdown("### 🔍 Analysis Details")
        analysis_ids = [f"{r['resume_name']} vs {r['jd_name']} ({r['created_at'][:10]})" for r in results]
        selected_analysis = st.selectbox("Select analysis to view details:", analysis_ids)
        
        if selected_analysis:
            selected_index = analysis_ids.index(selected_analysis)
            result = results[selected_index]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Scores Breakdown")
                st.metric("Final Score", f"{result['final_score']}/10")
                st.metric("Local Match Score", f"{result['local_score']}/10")
                st.metric("Gemini AI Score", f"{result['gemini_score']}/10")
            
            with col2:
                st.markdown("#### 🔧 Extracted Skills")
                skills = result.get('skills_extracted', [])
                if skills:
                    skill_html = "".join([f'<span class="skill-pill">{skill}</span>' for skill in skills[:8]])
                    st.markdown(f'<div style="margin: 1rem 0;">{skill_html}</div>', unsafe_allow_html=True)
                    if len(skills) > 8:
                        st.write(f"... and {len(skills) - 8} more skills")
                else:
                    st.info("No skills recorded for this analysis")
            
            st.markdown("#### 📋 AI Analysis")
            st.markdown(f"""
            <div class="section-card">
                <p style='line-height: 1.6; color: #495057;'>{result['justification']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def run(self):
        """Main application runner"""
        app_mode = self.render_sidebar()
        
        if app_mode == "🏠 Dashboard":
            self.render_dashboard()
        elif app_mode == "📊 Analyze Resume":
            self.render_analyze_resume()
        elif app_mode == "👥 Candidate Database":
            self.render_candidates()
        elif app_mode == "📈 Analytics & History":
            self.render_results_history()

# Run the application
if __name__ == "__main__":
    app = ProfessionalResumeScreener()
    app.run()
