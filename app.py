# DEPENDENCIES
import logging
import traceback
import streamlit as st
from typing import Optional
from ui.layouts import AppLayout
from ui.components import UIComponents
from core.file_handler import FileHandler
from core.search_engine import SearchEngine
from core.system_manager import SystemManager
from ui.visualizations import VisualizationManager


# Configure logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

class MedicalSearchApp:
    def __init__(self):
        self.ui             = UIComponents()
        self.layout         = AppLayout()
        self.viz_manager    = VisualizationManager()
        self.file_handler   = FileHandler()
        self.system_manager = SystemManager()
        self.search_engine  = None
        

    def initialize_app(self):
        """
        Initialize the Streamlit app configuration and load system components
        """
        # Page configuration
        st.set_page_config(page_title            = "Explainable Medical Search",
                           page_icon             = "🩺",
                           layout                = "wide",
                           initial_sidebar_state = "collapsed",
                           menu_items            = {'Get Help'     : 'https://github.com/satyaki-mitra/MediMap-XAI',
                                                    'Report a bug' : 'https://github.com/satyaki-mitra/MediMap-XAI/issues',
                                                    'About'        : "Explainable Medical Search App - AI-powered medical information retrieval with transparent explanations"
                                                   },
                          )
        

        # Apply custom CSS
        self.layout.apply_custom_css()
        
        # Initialize system components
        try:
            with st.spinner(text = "🔧 Initializing AI systems..."):
                self.search_engine = self.system_manager.initialize_system()
                
            if (not self.search_engine):
                st.error("Failed to initialize search system")
                st.stop()
                
        except Exception as e:
            st.error(f"System initialization failed: {str(e)}")
            logger.error(f"System initialization error: {traceback.format_exc()}")
            st.stop()
    

    def check_data_status(self):
        """
        Check and display data availability
        """
        data_status     = self.system_manager.check_data_availability()
        total_documents = sum(data_status.values())
        
        if (total_documents == 0):
            self.ui.show_no_data_warning()
            st.stop()
            
        return data_status, total_documents
    

    def handle_search_input(self) -> tuple[str, bool]:
        """
        Handle search input from text or file upload
        """
        query_text, uploaded_file = self.ui.render_search_interface()
        
        user_query                = ""
        file_processed            = False
        
        if (uploaded_file is not None):
            extracted_text = self.file_handler.extract_text_from_upload(uploaded_file)
            
            if (extracted_text and extracted_text.strip()):
                user_query     = extracted_text
                file_processed = True
                st.success(f"File processed successfully! Extracted {len(extracted_text):,} characters.")
            
            else:
                st.error("Failed to extract text from uploaded file")
                user_query = query_text
        
        else:
            user_query = query_text
            
        return user_query.strip(), file_processed
    

    def display_results(self, results, query, data_status):
        """
        Display search results with explanations and visualizations
        """
        if not results:
            self.ui.show_no_results_message()
            return
            
        # Show knowledge map if available and has data
        if (self.search_engine.visualizer and self.search_engine.visualizer.som is not None):
            self.viz_manager.display_knowledge_map(visualizer  = self.search_engine.visualizer, 
                                                   collections = ["reports", "queries", "drug_reviews"],
                                                  )
        
        st.markdown(body = "### 🔍 **Search Results**")
        
        # Main result
        top_result  = results[0]
        explanation = self.search_engine.get_explanation(query, 
                                                         top_result
                                                        )
        
        # Display main result card
        self.ui.display_main_result_card(result      = top_result, 
                                         explanation = explanation, 
                                         mongo       = self.search_engine.mongo,
                                        )
        
        # Display explanations and analysis
        if (explanation and (explanation.get('base_similarity', 0) > 0)):
            self.ui.display_advanced_explanations(explanation = explanation, 
                                                  top_result  = top_result, 
                                                  all_results = results, 
                                                  mongo       = self.search_engine.mongo,
                                                 )
        
        # Display additional results
        if (len(results) > 1):
            self.ui.display_additional_results(results[1:], self.search_engine.mongo)
    

    def display_dashboard(self, data_status, total_documents):
        """
        Display the main dashboard when no search is performe
        """
        st.markdown(body = "### 📊 **Medical Database Overview**")
        
        # Stats cards
        self.ui.display_stats_cards(data_status     = data_status, 
                                    total_documents = total_documents,
                                   )
        
        # Recent activity or sample data visualization
        self.viz_manager.display_database_overview(data_status = data_status)
        
        # Quick start guide
        self.ui.show_quick_start_guide()
    

    def run(self):
        """
        Main application entry point
        """
        try:
            # Initialize app
            self.initialize_app()
            
            # Render header
            self.layout.render_header()
            
            # Check data availability
            data_status, total_documents = self.check_data_status()
            
            # Handle search input
            user_query, file_processed    = self.handle_search_input()
            search_clicked                = st.button(label               = "🔍 **Search Medical Database**", 
                                                      type                = "primary", 
                                                      use_container_width = True,
                                                     )
            
            # Process search or show dashboard
            if (search_clicked and user_query):
                with st.spinner(text = "🔍 Analyzing your query and searching medical database..."):
                    try:
                        results = self.search_engine.search(user_query)
                        self.display_results(results     = results, 
                                             query       = user_query, 
                                             data_status = data_status,
                                            )
                        
                        # Log search for analytics
                        logger.info(f"Search performed: query_length = {len(user_query)}, results_count = {len(results)}")
                        
                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")
                        logger.error(f"Search error: {traceback.format_exc()}")
            
            else:
                # Show dashboard
                self.display_dashboard(data_status     = data_status, 
                                       total_documents = total_documents,
                                      )
            
            # Render footer
            self.layout.render_footer()
            
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            logger.error(f"Application error: {traceback.format_exc()}")


def main():
    """
    Application entry point
    """
    app = MedicalSearchApp()
    app.run()

# EXECUTEE
if __name__ == "__main__":
    main()