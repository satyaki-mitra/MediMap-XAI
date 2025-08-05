
# DEPENDENCIES
import streamlit as st
from datetime import datetime


class AppLayout:
    def __init__(self):
        self.app_name    = "🩺 Explainable Medical Search"
        self.app_version = "v1.0"
    

    def apply_custom_css(self):
        """
        Apply comprehensive custom CSS styling for Streamlit 1.47.1
        """
        # Use st.html() for better CSS handling in newer Streamlit versions
        st.html("""
        <style>
            /* Import Google Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            /* Global Styles */
            .main .block-container {
                font-family: 'Inter', sans-serif;
                max-width: 1200px;
                padding-top: 2rem;
            }
            
            /* Hide default Streamlit elements */
            #MainMenu {visibility: hidden;}
            .stDeployButton {display: none;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Header Styles */
            .main-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            
            .main-header h1 {
                font-size: 3.5rem;
                font-weight: 700;
                margin: 0;
                text-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            
            .main-header p {
                font-size: 1.2rem;
                margin: 0.5rem 0 0 0;
                opacity: 0.9;
            }
            
            /* Search Container */
            .search-container {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 2rem;
                border-radius: 15px;
                margin-bottom: 2rem;
                border: 1px solid #e0e6ed;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            }
            
            /* Enhanced Metric Cards */
            .metric-card-enhanced {
                background: white;
                padding: 1.5rem;
                border-radius: 12px;
                border: 1px solid #e1e5e9;
                margin: 0.5rem 0;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .metric-card-enhanced:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            }
            
            .metric-card-enhanced::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, #667eea, #764ba2);
            }
            
            .metric-header {
                display: flex;
                align-items: center;
                margin-bottom: 0.5rem;
            }
            
            .metric-icon {
                font-size: 1.5rem;
                margin-right: 0.5rem;
            }
            
            .metric-title {
                font-weight: 500;
                color: #495057;
                font-size: 0.9rem;
            }
            
            .metric-value {
                font-size: 2rem;
                font-weight: 700;
                margin: 0.5rem 0;
                color: #1f77b4;
            }
            
            .metric-change {
                font-size: 0.8rem;
                color: #28a745;
                font-weight: 500;
            }
            
            /* Enhanced Result Cards */
            .main-result-container {
                background: white;
                border-radius: 15px;
                padding: 0;
                margin: 1.5rem 0;
                box-shadow: 0 8px 30px rgba(0,0,0,0.08);
                border: 1px solid #e9ecef;
                overflow: hidden;
            }
            
            .result-card-enhanced {
                padding: 2rem;
                background: white;
            }
            
            .result-header {
                margin-bottom: 1.5rem;
            }
            
            .result-title {
                color: #1f77b4;
                margin: 0 0 1rem 0;
                font-weight: 600;
                font-size: 1.5rem;
            }
            
            .result-meta {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-bottom: 1rem;
            }
            
            .result-content {
                margin: 1.5rem 0;
                line-height: 1.6;
                color: #495057;
                font-size: 1rem;
            }
            
            .result-footer {
                border-top: 1px solid #e9ecef;
                padding-top: 1rem;
                margin-top: 1.5rem;
            }
            
            /* Enhanced Tags */
            .tag {
                display: inline-block;
                padding: 0.4rem 1rem;
                margin: 0.2rem;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 500;
                border: 1px solid transparent;
                transition: all 0.2s ease;
            }
            
            .tag:hover {
                transform: scale(1.05);
            }
            
            .tag-source { 
                background: linear-gradient(135deg, #6c5ce7, #a29bfe);
                color: white;
            }
            
            .tag-drug { 
                background: linear-gradient(135deg, #fd79a8, #e84393);
                color: white;
            }
            
            .tag-condition { 
                background: linear-gradient(135deg, #00b894, #00cec9);
                color: white;
            }
            
            .tag-specialty { 
                background: linear-gradient(135deg, #0984e3, #74b9ff);
                color: white;
            }
            
            /* Confidence Panel */
            .confidence-panel {
                padding: 1.5rem;
                border-radius: 12px;
                margin: 1rem 0;
                border-left: 4px solid;
                background: white;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            }
            
            .confidence-excellent {
                border-left-color: #28a745;
                background: linear-gradient(135deg, #d4edda, #c3e6cb);
            }
            
            .confidence-good {
                border-left-color: #17a2b8;
                background: linear-gradient(135deg, #d1ecf1, #bee5eb);
            }
            
            .confidence-moderate {
                border-left-color: #ffc107;
                background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            }
            
            .confidence-weak {
                border-left-color: #fd7e14;
                background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            }
            
            .confidence-poor {
                border-left-color: #dc3545;
                background: linear-gradient(135deg, #f8d7da, #f5c6cb);
            }
            
            .confidence-text {
                font-size: 1rem;
                margin-bottom: 0.5rem;
                font-weight: 500;
            }
            
            .confidence-details {
                font-size: 0.9rem;
                opacity: 0.8;
            }
            
            /* App Footer */
            .app-footer {
                background: linear-gradient(135deg, #2d3436, #636e72);
                color: white;
                padding: 2rem;
                border-radius: 15px;
                margin-top: 3rem;
                text-align: center;
            }
            
            .footer-content {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                flex-wrap: wrap;
                gap: 2rem;
                margin-bottom: 2rem;
            }
            
            .footer-section {
                flex: 1;
                min-width: 200px;
                text-align: left;
            }
            
            .footer-title {
                font-weight: 600;
                margin-bottom: 1rem;
                color: #fff;
                font-size: 1.1rem;
            }
            
            .footer-links {
                list-style: none;
                padding: 0;
                margin: 0;
            }
            
            .footer-links li {
                margin: 0.5rem 0;
                color: #bbb;
                font-size: 0.9rem;
            }
            
            .footer-links li:hover {
                color: #fff;
                cursor: pointer;
            }
            
            .footer-divider {
                border: none;
                height: 1px;
                background: #636e72;
                margin: 1.5rem 0;
            }
            
            .footer-bottom {
                text-align: center;
                color: #bbb;
                font-size: 0.9rem;
                line-height: 1.6;
            }
            
            .footer-bottom strong {
                color: #fff;
            }
            
            /* Streamlit widget styling */
            .stTextInput > div > div > input {
                border-radius: 10px;
                border: 2px solid #e1e5e9;
                padding: 0.75rem;
                font-size: 1rem;
            }
            
            .stTextInput > div > div > input:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
            }
            
            .stTextArea > div > div > textarea {
                border-radius: 10px;
                border: 2px solid #e1e5e9;
                padding: 0.75rem;
                font-size: 1rem;
                min-height: 120px;
            }
            
            .stTextArea > div > div > textarea:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
            }
            
            .stButton > button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 0.75rem 2rem;
                font-weight: 600;
                font-size: 1rem;
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            }
            
            /* Responsive Design */
            @media (max-width: 768px) {
                .main-header h1 {
                    font-size: 2.5rem;
                }
                
                .main-header p {
                    font-size: 1rem;
                }
                
                .search-container {
                    padding: 1.5rem;
                }
                
                .result-card-enhanced {
                    padding: 1.5rem;
                }
                
                .footer-content {
                    flex-direction: column;
                    text-align: center;
                }
                
                .footer-section {
                    text-align: center;
                }
            }
        </style>
        """)
    

    def add_compact_css(self):
        """
        Add compact CSS styles to the existing CSS
        """
        st.html("""
        <style>
            /* Compact Header */
            .compact-header {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 1rem 1.5rem;
                border-radius: 10px;
                margin-bottom: 1.5rem;
                border: 1px solid #dee2e6;
            }
            
            .header-content {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .header-toggle {
                background: #667eea;
                color: white;
                padding: 0.3rem 0.8rem;
                border-radius: 15px;
                font-size: 0.8rem;
            }
            
            /* Compact Result Container */
            .compact-result-container {
                background: white;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }
            
            .compact-result-header {
                margin-bottom: 0.75rem;
            }
            
            .result-title-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.5rem;
            }
            
            .compact-title {
                margin: 0;
                font-size: 1.1rem;
                color: #1f77b4;
                font-weight: 600;
            }
            
            .compact-score {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: 12px;
                font-size: 0.85rem;
                font-weight: 600;
            }
            
            .compact-meta {
                display: flex;
                gap: 0.5rem;
                flex-wrap: wrap;
            }
            
            .compact-tag {
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                padding: 0.2rem 0.5rem;
                border-radius: 8px;
                font-size: 0.8rem;
                color: #495057;
            }
            
            .compact-tag.tag-drug { border-color: #fd79a8; color: #fd79a8; }
            .compact-tag.tag-condition { border-color: #00b894; color: #00b894; }
            .compact-tag.tag-specialty { border-color: #0984e3; color: #0984e3; }
            
            .compact-content {
                color: #495057;
                line-height: 1.4;
                margin: 0.75rem 0;
                font-size: 0.9rem;
            }
            
            .compact-actions {
                display: flex;
                gap: 0.5rem;
                margin-top: 0.75rem;
            }
            
            .compact-btn {
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                padding: 0.4rem 0.8rem;
                border-radius: 6px;
                font-size: 0.8rem;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            
            .compact-btn:hover {
                background: #e9ecef;
                border-color: #adb5bd;
            }
            
            /* Compact Confidence */
            .compact-confidence {
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }
            
            .confidence-badge {
                padding: 0.3rem 0.8rem;
                border-radius: 15px;
                font-size: 0.8rem;
                font-weight: 600;
                text-align: center;
                color: white;
                width: fit-content;
            }
            
            .confidence-excellent { background: #28a745; }
            .confidence-good { background: #17a2b8; }
            .confidence-moderate { background: #ffc107; color: #000; }
            .confidence-weak { background: #fd7e14; }
            .confidence-poor { background: #dc3545; }
            
            .confidence-brief {
                font-size: 0.85rem;
                color: #6c757d;
                line-height: 1.3;
            }
            
            /* Compact Token Display */
            .compact-token {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.3rem;
                border-bottom: 1px solid #f1f3f4;
            }
            
            .token-word {
                font-weight: 500;
                font-size: 0.9rem;
            }
            
            .token-score {
                font-size: 0.8rem;
                font-weight: 600;
            }
            
            /* Compact spacing adjustments */
            .element-container {
                margin-bottom: 0.5rem !important;
            }
            
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.5rem;
            }
            
            .stTabs [data-baseweb="tab"] {
                padding: 0.5rem 1rem;
                font-size: 0.9rem;
            }
            
            .stExpander {
                margin: 0.5rem 0;
            }
            
            .stExpander > div > div > div > div {
                padding: 0.75rem;
            }
            
            /* Process Step Styling */
            .process-step {
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 0.75rem;
                margin: 0.5rem 0;
            }
            
            .step-header {
                display: flex;
                align-items: center;
                margin-bottom: 0.25rem;
            }
            
            .step-icon {
                font-size: 1.2rem;
                margin-right: 0.5rem;
            }
            
            .step-title {
                font-weight: 600;
                color: #495057;
            }
            
            .step-description {
                font-size: 0.85rem;
                color: #6c757d;
                margin-left: 1.7rem;
            }
            
            .step-connector {
                text-align: center;
                color: #6c757d;
                margin: 0.25rem 0;
            }
            
            /* Context Cards */
            .context-card {
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 0.75rem;
                margin: 0.5rem 0;
                text-align: center;
            }
            
            .context-icon {
                font-size: 1.5rem;
                margin-bottom: 0.25rem;
            }
            
            .context-label {
                font-size: 0.8rem;
                color: #6c757d;
                margin-bottom: 0.25rem;
            }
            
            .context-value {
                font-weight: 600;
                color: #495057;
                font-size: 0.9rem;
            }
            
            /* Token Analysis */
            .token-analysis-item {
                background: #f8f9fa;
                border-radius: 6px;
                padding: 0.5rem;
                margin: 0.25rem 0;
                font-size: 0.85rem;
            }
            
            /* Confidence Metric */
            .confidence-metric {
                background: #e9ecef;
                border-radius: 4px;
                height: 8px;
                margin: 0.25rem 0;
                overflow: hidden;
            }
            
            .metric-bar {
                height: 100%;
                border-radius: 4px;
                transition: width 0.5s ease;
            }
            
            /* Confidence Reason */
            .confidence-reason {
                display: flex;
                align-items: center;
                padding: 0.5rem;
                border-radius: 6px;
                margin: 0.25rem 0;
                background: #f8f9fa;
                border: 1px solid #e9ecef;
            }
            
            .reason-icon {
                margin-right: 0.5rem;
            }
            
            .reason-text {
                font-size: 0.85rem;
                color: #495057;
            }
            
            .reason-positive {
                border-color: #28a745;
                background: #d4edda;
            }
            
            .reason-negative {
                border-color: #dc3545;
                background: #f8d7da;
            }
            
            .reason-neutral {
                border-color: #6c757d;
                background: #f8f9fa;
            }
            
            /* No Results and Warnings */
            .no-results-container,
            .warning-container {
                text-align: center;
                padding: 2rem;
                background: #f8f9fa;
                border-radius: 12px;
                border: 1px solid #e9ecef;
                margin: 2rem 0;
            }
            
            .no-results-icon,
            .warning-icon {
                font-size: 4rem;
                margin-bottom: 1rem;
            }
            
            .no-results-title,
            .warning-title {
                color: #495057;
                margin: 0 0 0.5rem 0;
            }
            
            .no-results-subtitle,
            .warning-subtitle {
                color: #6c757d;
                margin: 0;
            }
            
            .setup-instructions {
                background: white;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 1.5rem;
                margin: 1rem 0;
                text-align: left;
            }
            
            .setup-instructions ol {
                margin: 0.5rem 0;
                padding-left: 1.5rem;
            }
            
            .setup-instructions li {
                margin: 0.5rem 0;
                color: #495057;
            }
            
            .setup-instructions code {
                background: #f8f9fa;
                padding: 0.2rem 0.4rem;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                color: #e83e8c;
            }
        </style>
        """)


    def render_header(self):
        """
        Render the enhanced application header using st.html()
        """
        current_time = datetime.now().strftime("%B %d, %Y")
        
        st.html(f"""
        <div class="main-header">
            <h1>{self.app_name}</h1>
            <p>AI-powered medical information retrieval with transparent explanations • {current_time}</p>
        </div>
        """)
    

    def render_compact_header(self):
        """
        Render a more compact version of the header
        """
        st.html(f"""
        <div class="compact-header">
            <div class="header-content">
                <div class="header-left">
                    <h2 style="margin: 0; color: #667eea;">{self.app_name}</h2>
                    <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">AI-powered medical information retrieval</p>
                </div>
                <div class="header-right">
                    <div class="header-toggle">
                        <span style="font-size: 0.8rem; color: white;">Compact View</span>
                    </div>
                </div>
            </div>
        </div>
        """)
    

    def render_footer(self):
        """
        Render enhanced application footer using st.html()
        """
        st.html("""
        <div class="app-footer">
            <div class="footer-content">
                <div class="footer-section">
                    <div class="footer-title">🩺 Medical Search AI</div>
                    <p style="margin: 0; color: #bbb; font-size: 0.9rem;">
                        Empowering healthcare with explainable AI
                    </p>
                </div>
                
                <div class="footer-section">
                    <div class="footer-title">📋 Features</div>
                    <ul class="footer-links">
                        <li>Semantic Search</li>
                        <li>AI Explanations</li>
                        <li>Result Analytics</li>
                        <li>Confidence Scoring</li>
                    </ul>
                </div>
                
                <div class="footer-section">
                    <div class="footer-title">🔗 Resources</div>
                    <ul class="footer-links">
                        <li>Documentation</li>
                        <li>API Reference</li>
                        <li>Support</li>
                        <li>GitHub</li>
                    </ul>
                </div>
                
                <div class="footer-section">
                    <div class="footer-title">⚖️ Legal</div>
                    <ul class="footer-links">
                        <li>Privacy Policy</li>
                        <li>Terms of Use</li>
                        <li>Medical Disclaimer</li>
                    </ul>
                </div>
            </div>
            
            <hr class="footer-divider">
            
            <div class="footer-bottom">
                <strong>⚠️ Medical Disclaimer:</strong> This tool is for informational purposes only. 
                Always consult with healthcare professionals for medical advice.
            </div>
        </div>
        """)
    

    def create_metric_card(self, title: str, value: str, icon: str = "📊", change: str = ""):
        """
        Create a metric card with enhanced styling using st.html()
        """
        change_html = f'<div class="metric-change">{change}</div>' if change else ''
        
        st.html(f"""
        <div class="metric-card-enhanced">
            <div class="metric-header">
                <span class="metric-icon">{icon}</span>
                <span class="metric-title">{title}</span>
            </div>
            <div class="metric-value">{value}</div>
            {change_html}
        </div>
        """)


    def create_info_card(self, title: str, content: str, icon: str = "ℹ️", color: str = "#17a2b8"):
        """
        Create an informational card using st.html()
        """
        st.html(f"""
        <div style="
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid {color};
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin: 1rem 0;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                <h4 style="margin: 0; color: {color};">{title}</h4>
            </div>
            <p style="margin: 0; color: #495057; line-height: 1.5;">{content}</p>
        </div>
        """)
    

    def create_progress_bar(self, value: float, label: str = "", color: str = "#667eea"):
        """
        Create an animated progress bar using st.html()
        """
        percentage = int(value * 100)
        st.html(f"""
        <div style="margin: 1rem 0;">
            {f'<label style="font-weight: 500; color: #495057; margin-bottom: 0.5rem; display: block;">{label}</label>' if label else ''}
            <div style="
                background: #e9ecef;
                border-radius: 10px;
                overflow: hidden;
                height: 20px;
            ">
                <div style="
                    background: linear-gradient(90deg, {color}, {color}aa);
                    height: 100%;
                    width: {percentage}%;
                    border-radius: 10px;
                    transition: width 0.8s ease;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 0.8rem;
                    font-weight: 500;
                ">
                    {percentage}%
                </div>
            </div>
        </div>
        """)


    def create_search_container(self):
        """
        Create a styled search container
        """
        st.html('<div class="search-container">')
        st.markdown("### 🔍 **Search Medical Database**")
        st.markdown("Enter your medical query or upload a document to search through our comprehensive medical database.")
        
    def end_search_container(self):
        """
        End the search container
        """
        st.html('</div>')


    def create_loading_state(self, message: str = "Loading..."):
        """
        Create a loading state with spinner using st.html()
        """
        st.html(f"""
        <div style="text-align: center; padding: 2rem;">
            <div style="
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            "></div>
            <p style="margin-top: 1rem; color: #6c757d;">{message}</p>
        </div>
        <style>
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
        """)


    def render_sidebar(self):
        """
        Render enhanced sidebar with navigation and settings
        """
        with st.sidebar:
            st.html(f"""
            <div style="text-align: center; padding: 1rem;">
                <h3 style="color: #667eea;">{self.app_name}</h3>
                <p style="color: #6c757d; font-size: 0.9rem;">{self.app_version}</p>
            </div>
            """)
            
            st.markdown("---")
            
            # Navigation
            st.markdown("### 🧭 **Navigation**")
            
            nav_options = {"🔍 Search"    : "search",
                           "📊 Analytics" : "analytics", 
                           "⚙️ Settings"   : "settings",
                           "❓ Help"      : "help",
                          }
            
            selected_nav = st.selectbox(label   = "Choose a section:",
                                        options = list(nav_options.keys()),
                                        index   = 0,
                                       )
            
            st.markdown("---")
            
            # Quick Actions
            st.markdown("### ⚡ **Quick Actions**")
            
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()
            
            if st.button("📈 View Statistics", use_container_width=True):
                st.session_state.show_stats = True
            
            if st.button("🆘 Get Help", use_container_width=True):
                st.session_state.show_help = True
            
            st.markdown("---")
            
            # System Status
            st.markdown("### 🔧 **System Status**")
            
            # Mock system status
            status_items = [("Database", "🟢 Online", "green"),
                            ("AI Model", "🟢 Ready", "green"),
                            ("Search Engine", "🟢 Active", "green"),
                            ("Explanations", "🟢 Available", "green"),
                           ]
            
            for item, status, color in status_items:
                st.markdown(f"**{item}:** {status}")
            
            st.markdown("---")
            
            # About
            st.markdown("### ℹ️ **About**")
            st.markdown("""
            This application uses advanced AI to help you search medical information 
            with transparent explanations of how results are found.
            
            **Features:**
            - 🔍 Semantic search
            - 🧠 AI explanations
            - 📊 Result analytics
            - 🎯 Confidence scoring
            """)