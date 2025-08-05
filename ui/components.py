# DEPENDENCIES
import time
import numpy as np
import pandas as pd
import streamlit as st
from typing import List
from typing import Dict
from typing import Optional
import plotly.graph_objects as go
from ui.visualizations import VisualizationManager
from core.confidence_analyzer import ConfidenceAnalyzer


class UIComponents:
    def __init__(self):
        self.confidence_analyzer = ConfidenceAnalyzer()
    

    def render_search_interface(self) -> tuple[str, Optional[any]]:
        """
        Render the search interface
        """
        st.markdown(body              = '<div class="search-container">', 
                    unsafe_allow_html = True,
                   )
        
        # Search tabs for different input methods
        tab1, tab2    = st.tabs(tabs = ["💬 **Text Search**", "📄 **File Upload**"])
        
        query_text    = ""
        uploaded_file = None
        
        with tab1:
            col1, col2 = st.columns(spec = [4, 1])
            
            with col1:
                query_text = st.text_area(label       = "🔍 **Describe your symptoms or medical questions**",
                                          placeholder = "e.g., I have been experiencing shortness of breath and fatigue for the past week. What could be causing this?",
                                          height      = 100,
                                          help        = "Be as specific as possible. Include symptoms, duration, severity, and any other relevant details.",
                                         )
            
            
            with col2:
                st.markdown(body = "#### 💡 **Quick Examples**")
                if st.button(label = "🫀 Chest Pain", use_container_width = True):
                    st.session_state.example_query = "sharp chest pain when breathing deeply"
                
                if st.button(label = "🤒 Fever & Fatigue", use_container_width = True):
                    st.session_state.example_query = "high fever with extreme fatigue and body aches"
                
                if st.button(label = "💊 Drug Information", use_container_width = True):
                    st.session_state.example_query = "side effects of metformin for diabetes"
                
                # Apply example query if selected
                if ('example_query' in st.session_state):
                    query_text = st.session_state.example_query
                    del st.session_state.example_query
        
        with tab2:
            col1, col2 = st.columns(spec = [3, 1])
            with col1:
                uploaded_file = st.file_uploader(label = "📁 **Upload Medical Document**",
                                                 type  = ["txt", "pdf", "docx"],
                                                 help  = "Upload medical reports, lab results, or any medical text document",
                                                )
            
            with col2:
                st.markdown(body = "#### 📋 **Supported Formats**")
                st.markdown(body = "- 📄 PDF files")
                st.markdown(body = "- 📝 Text files (.txt)")
                st.markdown(body = "- 📄 Word docs (.docx)")
                st.markdown(body = "- 🏥 Medical reports")
        

        st.markdown(body              = '</div>', 
                    unsafe_allow_html = True,
                   )

        return query_text, uploaded_file
    

    def display_stats_cards(self, data_status: Dict, total_documents: int):
        """
        Display statistics cards
        """
        col1, col2, col3, col4 = st.columns(spec = 4)
        
        cards_data             = [("Medical Reports", data_status.get('reports', 0), "#1f77b4", "🏥"),
                                  ("Q&A Pairs", data_status.get('queries', 0), "#ff7f0e", "❓"),
                                  ("Drug Reviews", data_status.get('drug_reviews', 0), "#2ca02c", "💊"),
                                  ("Total Documents", total_documents, "#d62728", "📚")
                                 ]
        
        for index, (title, count, color, icon) in enumerate(cards_data):
            col = [col1, col2, col3, col4][index]
            
            with col:
                self._render_metric_card(title, count, color, icon)
    

    def _render_metric_card(self, title: str, value: int, color: str, icon: str):
        """
        Render an individual metric card with animation
        """
        st.markdown(body                     = f"""
                                                    <div class="metric-card-enhanced" style="border-left: 4px solid {color};">
                                                            <div class="metric-header">
                                                                <span class="metric-icon">{icon}</span>
                                                                <span class="metric-title">{title}</span>
                                                            </div>
                                                            <div class="metric-value" style="color: {color};">{value:,}</div>
                                                            <div class="metric-change">📈 Active</div>
                                                    </div>
                                                """, 
                           unsafe_allow_html = True,
                          )


    def display_compact_main_result_card(self, result: Dict, explanation: Dict, mongo):
        """
        Display the main search result with compact UI design
        """
        document_id       = result.get("id")
        source_collection = result.get("source_collection")
        document          = mongo.find_one(source_collection, {"_id": document_id}) or {}
        
        # Extract result information
        title             = self._extract_title(document, source_collection)
        content           = self._extract_content(document, max_length=150)  # Reduced length
        similarity_score  = explanation.get("base_similarity", result.get("cosine_score", 0.0))
        
        # Compact result container
        st.markdown(body=f"""
            <div class="compact-result-container">
                <div class="compact-result-header">
                    <div class="result-title-row">
                        <h4 class="compact-title">📋 {title}</h4>
                        <div class="compact-score">{similarity_score:.1%}</div>
                    </div>
                    <div class="compact-meta">
                        <span class="compact-tag">📁 {source_collection.title()}</span>
                        {self._generate_compact_tags(document)}
                    </div>
                </div>
                <div class="compact-content">{content}</div>
            </div>
        """, unsafe_allow_html=True)


    def display_main_result_card(self, result: Dict, explanation: Dict, mongo):
        """
        Display the main search result with enhanced UI
        """
        document_id       = result.get("id")
        source_collection = result.get("source_collection")
        document          = mongo.find_one(source_collection, {"_id": document_id}) or {}
        
        # Extract result information
        title             = self._extract_title(document, source_collection)
        content           = self._extract_content(document)
        similarity_score  = explanation.get("base_similarity", result.get("cosine_score", 0.0))
        
        # Main result container
        st.markdown(body              = '<div class="main-result-container">', 
                    unsafe_allow_html = True,
                   )
        
        col1, col2        = st.columns(spec = [3, 1])
        
        with col1:
            # Result card with custom styling
            st.markdown(body              = f"""
                                                <div class="result-card-enhanced">
                                                    <div class="result-header">
                                                        <h3 class="result-title">📋 {title}</h3>
                                                        <div class="result-meta">
                                                            <span class="tag tag-source">📁 {source_collection.title()}</span>
                                                            {self._generate_additional_tags(document)}
                                                        </div>
                                                    </div>
                                                    <div class="result-content">
                                                        <p>{content}</p>
                                                    </div>
                                                </div>
                                                """, 
                        unsafe_allow_html = True,
                       )
        
        with col2:
            # Enhanced confidence panel
            self._display_confidence_panel(similarity_score = similarity_score, 
                                           explanation      = explanation,
                                          )
        
        st.markdown(body              = '</div>', 
                    unsafe_allow_html = True,
                   )

    
    def _display_confidence_panel(self, similarity_score: float, explanation: Dict):
        """
        Display enhanced confidence analysis panel
        """
        st.markdown(body = "#### 🎯 **Match Analysis**")
        
        # Confidence gauge
        viz_manager      = VisualizationManager()
        
        fig_gauge        = viz_manager.create_similarity_gauge(similarity_score, "Relevance Score")
        
        st.plotly_chart(figure_or_data      = fig_gauge,
                        use_container_width = True,
                       )
        
        # Confidence explanation with styling
        confidence_text  = self.confidence_analyzer.get_confidence_explanation(similarity_score)
        confidence_level = self.confidence_analyzer.get_confidence_level(similarity_score)
        
        st.markdown(body             = f"""
                                            <div class="confidence-panel confidence-{confidence_level.lower()}">
                                                <div class="confidence-text">{confidence_text}</div>
                                                <div class="confidence-details">
                                                    <strong>Similarity Score:</strong> {similarity_score:.1%}<br>
                                                    <strong>Confidence Level:</strong> {confidence_level}
                                                </div>
                                            </div>
                                        """, 
                    unsafe_allow_html = True,
                   )
        
        # Quick metrics
        token_impacts = explanation.get("token_importance", [])
        
        if token_impacts:
            positive_words = len([t for t in token_impacts if t.get('importance', 0) > 0.02])
            st.metric("🟢 Strong Matches", positive_words)
            st.metric("📝 Words Analyzed", len(token_impacts))


    def display_compact_explanations(self, explanation: Dict, top_result: Dict, all_results: List[Dict], mongo):
        """
        Display compact XAI explanations using expanders and condensed layout
        """
        st.markdown("---")
        
        # Use expander for the entire explanation section
        with st.expander(body = "🧠 **AI Explanations & Analysis**", expanded=False):
            
            # Create 2x2 grid layout for tabs content
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence Analysis (Compact)
                st.markdown(body = "#### 🎯 Confidence")
                confidence_level = self.confidence_analyzer.get_confidence_level(explanation.get("base_similarity", 0))
                confidence_text  = self.confidence_analyzer.get_confidence_explanation(explanation.get("base_similarity", 0))
                
                # Compact confidence display
                st.markdown(f"""
                    <div class="compact-confidence">
                        <div class="confidence-badge confidence-{confidence_level.lower()}">
                            {confidence_level}
                        </div>
                        <div class="confidence-brief">{confidence_text[:1000]}...</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Quick metrics in compact format
                token_impacts = explanation.get("token_importance", [])
                if token_impacts:
                    positive_words = len([t for t in token_impacts if t.get('importance', 0) > 0.02])
                    st.metric(label = "Strong Matches", 
                              value = positive_words, 
                              delta = None,
                             )
            
            with col2:
                # Token Analysis (Top 5 only)
                st.markdown(body = "#### 🔤 Key Words")
                if token_impacts:
                    top_tokens = sorted(token_impacts, 
                                        key     = lambda x: abs(x.get('importance', 0)), 
                                        reverse = True)[:5]
                    
                    for token in top_tokens:
                        importance = token.get('importance', 0)
                        word = token['token']
                        color = "#51cf66" if importance > 0 else "#ff6b6b"
                        
                        st.markdown(f"""
                            <div class="compact-token">
                                <span class="token-word">{word}</span>
                                <span class="token-score" style="color: {color}">
                                    {importance:.3f}
                                </span>
                            </div>
                        """, unsafe_allow_html=True)
            
            # Additional analysis in collapsible sections
            with st.expander("📊 Detailed Analysis", expanded=False):
                self._display_detailed_analysis_compact(explanation, top_result, all_results, mongo)


    def _display_detailed_analysis_compact(self, explanation: Dict, result: Dict, all_results: List[Dict], mongo):
        """
        Compact detailed analysis with minimal space usage
        """
        tab1, tab2, tab3 = st.tabs(["Decision Process", "Medical Context", "Result Stats"])
        
        with tab1:
            # Simplified process steps
            process_steps = [
                ("Query Processing", "Text analyzed"),
                ("Semantic Search", "Database queried"), 
                ("Relevance Scoring", "Results ranked")
            ]
            
            for i, (title, desc) in enumerate(process_steps):
                st.markdown(f"**{i+1}. {title}:** {desc}")
        
        with tab2:
            # Context in simple list format
            document = mongo.find_one(result['source_collection'], {"_id": result['id']}) or {}
            
            context_info = []
            if document.get('medical_specialty'):
                context_info.append(f"**Field:** {document['medical_specialty']}")
            if document.get('drug_name'):
                context_info.append(f"**Drug:** {document['drug_name']}")
            if document.get('condition'):
                context_info.append(f"**Condition:** {document['condition']}")
            
            for info in context_info:
                st.markdown(info)
        
        with tab3:
            # Quick result statistics
            scores = [r.get('cosine_score', 0) for r in all_results]
            if scores:
                st.metric("Avg Score", f"{np.mean(scores):.2%}")
                st.metric("Best Score", f"{max(scores):.2%}")
                st.metric("Results", len(all_results))


    def display_advanced_explanations(self, explanation: Dict, top_result: Dict, all_results: List[Dict], mongo):
        """
        Display advanced XAI explanations with enhanced UI
        """
        st.markdown(body = "---")
        st.markdown(body              = """
                                            <div class="section-header">
                                                <h3>🧠 Advanced AI Explanations</h3>
                                                <p class="section-subtitle">Understanding how the AI made its decision</p>
                                            </div>
                                        """, 
                    unsafe_allow_html = True,
                   )
        
        # Enhanced tabs with icons
        tab1, tab2, tab3, tab4 = st.tabs(tabs = ["🎯 Confidence Analysis", 
                                                 "🔍 Decision Process", 
                                                 "🏥 Medical Context",
                                                 "📊 Result Analytics",
                                                ],
                                        )
        
        with tab1:
            self._display_confidence_analysis_tab(explanation, 
                                                  top_result, 
                                                  mongo,
                                                 )
        
        with tab2:
            self._display_decision_process_tab(explanation)
        
        with tab3:
            self._display_medical_context_tab(top_result,
                                              mongo,
                                             )
        
        with tab4:
            self._display_result_analytics_tab(all_results, 
                                               mongo,
                                              )

    
    def _display_confidence_analysis_tab(self, explanation: Dict, result: Dict, mongo):
        """
        Confidence analysis tab
        """
        col1, col2 = st.columns(spec = [1, 1])
        
        with col1:
            st.markdown(body = "#### 📊 **Confidence Breakdown**")
            
            # Calculate confidence metrics
            document           = mongo.find_one(result['source_collection'], {"_id": result['id']}) or {}
            confidence_metrics = self.confidence_analyzer.calculate_detailed_confidence(explanation, 
                                                                                        document, 
                                                                                        result['source_collection'],
                                                                                       )
            
            # Display metrics with styling
            for metric, value in confidence_metrics.items():
                self._display_confidence_metric(metric, value)
        
        with col2:
            st.markdown(body = "#### 🤔 **Confidence Factors**")
            
            # Formatting reasoning display
            reasons = self.confidence_analyzer.generate_confidence_reasons(explanation, 
                                                                           document, 
                                                                           result['source_collection'],
                                                                          )
            
            for reason in reasons:
                self._display_confidence_reason(reason)
    

    def _display_decision_process_tab(self, explanation: Dict):
        """
        Decision process visualization
        """
        col1, col2 = st.columns(spec = [1, 1])
        
        with col1:
            st.markdown(body = "#### 🔄 **AI Processing Pipeline**")
            
            # Animated processing steps
            processing_steps = [("📥", "Query Processing", "Breaking down your text into meaningful components"),
                                ("🧮", "Embedding Generation", "Converting text to AI-readable numerical format"),
                                ("🔍", "Semantic Search", "Finding similar content in medical database"),
                                ("📊", "Relevance Scoring", "Calculating match quality scores"),
                                ("🎯", "Result Ranking", "Ordering results by relevance and quality"),
                                ("🧠", "Explanation Generation", "Creating transparent explanations"),
                               ]
            
            for i, (icon, title, description) in enumerate(processing_steps):
                with st.container():
                    st.markdown(body              = f"""
                                                        <div class="process-step">
                                                           <div class="step-header">
                                                               <span class="step-icon">{icon}</span>
                                                               <span class="step-title">{title}</span>
                                                           </div>
                                                           <div class="step-description">{description}</div>
                                                        </div>
                                                   """, 
                                unsafe_allow_html = True,
                               )
                    
                    if (i < len(processing_steps) - 1):
                        st.markdown(body              = '<div class="step-connector">⬇️</div>', 
                                    unsafe_allow_html = True,
                                   )
        
        with col2:
            # Token analysis
            self._display_token_analysis(explanation)
    

    def _display_token_analysis(self, explanation: Dict):
        """
        Display enhanced word-by-word analysis
        """
        st.markdown(body = "#### 🔤 **Word Impact Analysis**")
        
        token_impacts = explanation.get("token_importance", [])
        
        if not token_impacts:
            st.info("💡 Token analysis not available for this result")
            return
        
        # Group tokens by impact
        high_impact   = [t for t in token_impacts if abs(t.get('importance', 0)) > 0.05]
        medium_impact = [t for t in token_impacts if 0.02 < abs(t.get('importance', 0)) <= 0.05]
        low_impact    = [t for t in token_impacts if abs(t.get('importance', 0)) <= 0.02]
        
        # Display categorized tokens
        categories    = [("🔥 High Impact", high_impact, "#ff4757"),
                         ("⚡ Medium Impact", medium_impact, "#ffa502"),
                         ("💫 Low Impact", low_impact, "#747d8c"),
                        ]
        
        for category_name, tokens, color in categories:
            if tokens:
                st.markdown(f"**{category_name}**")
                for token in tokens[:5]:  # Limit display
                    importance  = token.get('importance', 0)
                    word        = token['token']
                    
                    impact_type = "Positive" if importance > 0 else "Negative"
                    st.markdown(body              = f"""
                                                        <div class="token-analysis-item" style="border-left: 3px solid {color};">
                                                            <strong>"{word}"</strong> - {impact_type} impact ({importance:.3f})
                                                        </div>
                                                     """, 
                                unsafe_allow_html = True,
                               )


    def display_compact_additional_results(self, results: List[Dict], mongo):
        """
        Display additional results in a compact table format
        """
        if len(results) <= 1:
            return
            
        st.markdown("#### 📚 More Results")
        
        # Create compact results table
        result_data = []
        for i, result in enumerate(results[1:6], 1):  # Show max 5 additional
            doc_id = result.get("id")
            collection = result.get("source_collection")
            doc = mongo.find_one(collection, {"_id": doc_id}) or {}
            
            title = self._extract_title(doc, collection)
            score = result.get("cosine_score", 0.0)
            
            result_data.append({
                "Rank": i + 1,
                "Title": title[:50] + "..." if len(title) > 50 else title,
                "Score": f"{score:.1%}",
                "Source": collection.replace('_', ' ').title()
            })
        
        # Display as dataframe for compact view
        if result_data:
            df = pd.DataFrame(result_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

    
    def display_additional_results(self, results: List[Dict], mongo):
        """
        Display additional results with enhanced styling
        """
        st.markdown(body = "#### 📚 **Additional Relevant Results**")
        
        # Limit to 5 additional results
        for i, result in enumerate(results[:5], 1):  
            doc_id     = result.get("id")
            collection = result.get("source_collection")
            doc        = mongo.find_one(collection, {"_id": doc_id}) or {}
            
            title      = self._extract_title(doc, collection)
            content    = self._extract_content(doc, max_length=200)
            score      = result.get("cosine_score", 0.0)
            
            # Expandable result
            with st.expander(label    = f"**{i}. {title}** {'🟢' if score > 0.7 else '🟡' if score > 0.4 else '🔴'} {score:.1%}", 
                             expanded = False,
                            ):

                col1, col2 = st.columns(spec = [3, 1])
                
                with col1:
                    st.markdown(body              = f"<div class='additional-result-content'>{content}</div>", 
                                unsafe_allow_html = True,
                               )
                
                with col2:
                    # Result metrics
                    st.metric(label = "Relevance", 
                              value = f"{score:.1%}",
                             )

                    st.metric(label = "Source", 
                              value = collection.title(),
                             )
                    
                    # Additional metadata
                    if doc.get("drug_name"):
                        st.metric(label = "💊 Drug", 
                                  value = doc["drug_name"])
                    
                    elif doc.get("condition"):
                        st.metric(label = "🏥 Condition", 
                                  value = doc["condition"])
                    
                    # Quick action buttons
                    if st.button(label = f"📋 View Details {i}", key = f"details_{i}"):
                        st.session_state[f"show_details_{i}"] = True
    

    def show_no_results_message(self):
        """
        Display no results message
        """
        st.markdown(body              = """
                                            <div class="no-results-container">
                                                <div class="no-results-icon">🔍</div>
                                                    <h3 class="no-results-title">No Matching Results Found</h3>
                                                    <p class="no-results-subtitle">We couldn't find any medical information matching your query</p>
                                            </div>
                                        """,       
                    unsafe_allow_html = True,
                  )
        
        # Suggestions
        col1, col2, col3 = st.columns(spec = 3)
        
        with col1:
            st.markdown("""
            #### 💡 **Try These Tips:**
            - Be more specific about symptoms
            - Use medical terminology
            - Include symptom duration/severity
            """)
        
        with col2:
            st.markdown("""
            #### 🔄 **Rephrasing Examples:**
            - "headache" → "severe headache for 3 days"
            - "tired" → "chronic fatigue with weakness"
            - "pain" → "sharp chest pain when breathing"
            """)
        
        with col3:
            st.markdown("""
            #### 📋 **Alternative Approaches:**
            - Upload a medical document
            - Ask specific questions
            - Use different synonyms
            """)
    

    def show_no_data_warning(self):
        """
        Display enhanced no data warning
        """
        st.markdown(body              = """
                                            <div class="warning-container">
                                                <div class="warning-icon">⚠️</div>
                                                <h2 class="warning-title">Medical Database Not Found</h2>
                                                <p class="warning-subtitle">The medical database appears to be empty or not properly loaded</p>
                                            </div>
                                        """, 
                    unsafe_allow_html = True,
                   )
        
        st.markdown(body              = """
                                            <div class="setup-instructions">
                                                <h4>🚀 **Quick Setup Guide:**</h4>
                                                <ol>
                                                    <li>📁 Place your medical CSV files in <code>data/raw_data/</code></li>
                                                    <li>🔄 Run the data ingestion: <code>python data_ingestion.py</code></li>
                                                    <li>⚡ Train the AI model: <code>python run_som_training.py</code></li>
                                                    <li>🎉 Restart this application</li>
                                                </ol>
                                            </div>
                                        """, 
                    unsafe_allow_html = True,
                   )
    

    def show_quick_start_guide(self):
        """
        Display quick start guide
        """
        st.markdown(body = "### 🚀 **Quick Start Guide**")
        
        col1, col2 = st.columns(spec = 2)
        
        with col1:
            st.markdown("""
            #### 💬 **How to Search Effectively**
            
            **For Symptoms:**
            - Describe location, intensity, duration
            - Include associated symptoms
            - Mention what makes it better/worse
            
            **For Medical Questions:**
            - Be specific about the condition
            - Ask about treatments, causes, or symptoms
            - Include relevant medical history
            """)
        
        with col2:
            st.markdown("""
            #### 📄 **File Upload Tips**
            
            **Supported Files:**
            - Medical reports (PDF, Word, Text)
            - Lab results and test reports
            - Discharge summaries
            - Clinical notes
            
            **Best Results:**
            - Clear, legible text
            - Complete medical information
            - Recent reports preferred
            """)
    

    # Helper methods
    def _extract_title(self, document: Dict, source_collection: str) -> str:
        """
        Extract appropriate title from document
        """
        title = document.get("medical_specialty") or document.get("condition") or document.get("drug_name")
        return title if title else f"{source_collection.replace('_', ' ').title()} Document"
    

    def _extract_content(self, document: Dict, max_length: int = 300) -> str:
        """
        Extract and truncate content from document
        """
        content = document.get("clean_text") or document.get("abstract") or document.get("question", "") + " " + document.get("answer", "")
        if (len(content) > max_length):
            content = content[:max_length] + "..."
        
        return content or "No content available"
    

    def _generate_additional_tags(self, document: Dict) -> str:
        """
        Generate additional tags for document metadata
        """
        tags = list()
        
        if document.get("drug_name"):
            tags.append(f'<span class="tag tag-drug">💊 {document["drug_name"]}</span>')
        
        if document.get("condition"):
            tags.append(f'<span class="tag tag-condition">🏥 {document["condition"]}</span>')
        
        if document.get("medical_specialty"):
            tags.append(f'<span class="tag tag-specialty">👨‍⚕️ {document["medical_specialty"]}</span>')
        
        return " ".join(tags)


    def _generate_compact_tags(self, document: Dict) -> str:
        """
        Generate compact tags for document metadata
        """
        tags = []
        
        if document.get("drug_name"):
            tags.append(f'<span class="compact-tag tag-drug">💊</span>')
        if document.get("condition"):
            tags.append(f'<span class="compact-tag tag-condition">🏥</span>')
        if document.get("medical_specialty"):
            tags.append(f'<span class="compact-tag tag-specialty">👨‍⚕️</span>')
        
        return " ".join(tags)
    

    def _display_confidence_metric(self, metric: str, value: float):
        """
        Display individual confidence metric with progress bar
        """
        st.metric(metric, f"{value:.1%}")
        
        color         = "#4CAF50" if value > 0.7 else "#FF9800" if value > 0.4 else "#f44336"
        progress_html = f"""
        <div class="confidence-metric">
            <div class="metric-bar" style="width: {value*100}%; background-color: {color};"></div>
        </div>
        """
        st.markdown(body              = progress_html, 
                    unsafe_allow_html = True,
                   )

        st.markdown("")
    

    def _display_confidence_reason(self, reason: Dict):
        """
        Display individual confidence reason
        """
        icon         = reason.get('icon', '💡')
        text         = reason.get('text', '')
        impact       = reason.get('impact', 'neutral')
        
        impact_class = f"reason-{impact}"
        
        st.markdown(body              = f"""
                                            <div class="confidence-reason {impact_class}">
                                                <span class="reason-icon">{icon}</span>
                                                <span class="reason-text">{text}</span>
                                            </div>
                                         """, 
                    unsafe_allow_html = True,
                   )

    
    def _display_medical_context_tab(self, result: Dict, mongo):
        """
        Display medical context analysis
        """
        document = mongo.find_one(result['source_collection'], {"_id": result['id']}) or {}
        
        st.markdown(body = "#### 🏥 **Medical Context Analysis**")
        
        # Context information in cards
        context_items = list()
        
        if document.get('medical_specialty'):
            context_items.append(("Medical Field", document['medical_specialty'], "🏥"))
        
        if document.get('drug_name'):
            context_items.append(("Medication", document['drug_name'], "💊"))
        
        if document.get('condition'):
            context_items.append(("Condition", document['condition'], "🩺"))
        
        context_items.append(("Source Type", result['source_collection'].replace('_', ' ').title(), "📁"))
        
        # Display in grid
        cols = st.columns(min(len(context_items), 2))
        for i, (label, value, icon) in enumerate(context_items):
            col = cols[i % len(cols)]
            
            with col:
                st.markdown(body              = f"""
                                                    <div class="context-card">
                                                        <div class="context-icon">{icon}</div>
                                                        <div class="context-label">{label}</div>
                                                        <div class="context-value">{value}</div>
                                                    </div>
                                                 """, 
                            unsafe_allow_html = True,
                           )
    

    def _display_result_analytics_tab(self, results: List[Dict], mongo):
        """
        Display result analytics and patterns
        """
        st.markdown(body = "#### 📊 **Search Result Analytics**")
        
        # Analyze result patterns
        visualization_manager = VisualizationManager()
        
        # Medical specialty distribution
        fig_specialty         = visualization_manager.create_medical_specialty_distribution(results = results, 
                                                                                            mongo   = mongo,
                                                                                           )
        
        if fig_specialty and fig_specialty.data:
            st.plotly_chart(figure_or_data      = fig_specialty, 
                            use_container_width = True,
                           )
        
        # Score distribution
        scores                = [result.get('cosine_score', 0) for result in results]
        
        if scores:
            # Create histogram of similarity scores
            fig_scores = go.Figure(data = [go.Histogram(x = scores, nbinsx = 20)])
            
            fig_scores.update_layout(title       = "Similarity Score Distribution",
                                     xaxis_title = "Cosine Similarity",
                                     yaxis_title = "Count",
                                     height      = 300,
                                     margin      = dict(l = 20, 
                                                        r = 20, 
                                                        t = 40, 
                                                        b = 20,
                                                       ),
                                    )

            st.plotly_chart(figure_or_data      = fig_scores, 
                            use_container_width = True,
                           )