# DEPENDENCIES
import numpy as np
import pandas as pd
import streamlit as st
from typing import List
from typing import Dict
import plotly.express as px
import plotly.graph_objects as go
from src.som_visualizer import SOMVisualizer


class VisualizationManager:
    def create_similarity_gauge(self, score: float, title: str) -> go.Figure:
        """
        Create a gauge chart with finer-grained similarity thresholds
        """
        fig = go.Figure(go.Indicator(mode   = "gauge+number+delta",
                                     value  = score * 100,
                                     domain = {'x': [0, 1], 'y': [0, 1]},
                                     title  = {'text': title, 'font': {'size': 16}},
                                     gauge  = {'axis'        : {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                                               'bar'         : {'color': "#1f77b4"},
                                               'bgcolor'     : "white",
                                               'borderwidth' : 2,
                                               'bordercolor' : "gray",
                                               'steps'       : [{'range': [0, 30], 'color': '#ffcdd2'},
                                                                {'range': [30, 50], 'color': '#ffe0b2'},
                                                                {'range': [50, 65], 'color': '#fff9c4'},
                                                                {'range': [65, 80], 'color': '#dcedc8'},
                                                                {'range': [80, 100], 'color': '#c8e6c9'},
                                                               ],
                                               'threshold'   : {'line'      : {'color': "red", 'width': 4},
                                                                'thickness' : 0.75,
                                                                'value'     : 85,
                                                               }
                                              }
                                    ),
                       )

        fig.update_layout(height = 220, 
                          margin = dict(l = 20, 
                                        r = 20, 
                                        t = 40, 
                                        b = 20,
                                       ),
                         )
        return fig


    def create_token_importance_chart(self, token_impacts: List[Dict]) -> go.Figure:
        """
        Create horizontal bar chart for token importance
        """
        if not token_impacts:
            return go.Figure()
        
        tokens      = [item['token'] for item in token_impacts]
        importances = [item['importance'] for item in token_impacts]
        colors      = ['#ff6b6b' if imp < 0 else '#51cf66' for imp in importances]
        
        fig         = go.Figure(go.Bar(y            = tokens,
                                       x            = importances,
                                       orientation  = 'h',
                                       marker_color = colors,
                                       text         = [f"{imp:.3f}" for imp in importances],
                                       textposition = 'auto',
                                      ),
                               )
        
        fig.update_layout(title       = "How Each Word Influences the Match",
                          xaxis_title = "Influence Score",
                          yaxis_title = "Words in Your Query",
                          height      = max(200, len(tokens) * 40),
                          margin      = dict(l = 20, 
                                             r = 20, 
                                             t = 40, 
                                             b = 20,
                                            ),
                          showlegend  = False,
                         )
        
        return fig


    def create_medical_specialty_distribution(self, results: List[Dict], mongo) -> go.Figure:
        """
        Create pie chart showing distribution of medical specialties in results
        """
        specialties = list()
        
        for result in results:
            doc = mongo.find_one(result['source_collection'], {"_id": result['id']})
            
            if doc:
                specialty = doc.get('medical_specialty', doc.get('condition', 'General'))
                
                if specialty:
                    specialties.append(specialty)
        
        if not specialties:
            return go.Figure()
        
        specialty_counts = pd.Series(specialties).value_counts()
        
        fig = go.Figure(data = [go.Pie(labels       = specialty_counts.index,
                                       values       = specialty_counts.values,
                                       hole         = 0.3,
                                       textinfo     = 'label+percent',
                                       textposition = 'outside'
                                      ),
                               ],
                       )
        
        fig.update_layout(title  = "Medical Areas in Your Results",
                          height = 300,
                          margin = dict(l = 20, 
                                        r = 20, 
                                        t = 40,
                                        b = 20,
                                       ),
                         )
        
        return fig


    def display_knowledge_map(self, visualizer: SOMVisualizer, collections: List[str]):
        """
        Display SOM knowledge map visualization
        """
        try:
            st.markdown(body = "### 🗺️ **Medical Knowledge Map**")
            map_x, map_y = visualizer._get_som_dimensions()
            
            # Build occupancy map
            occupancy    = np.zeros((map_x, map_y))
            
            for collection in collections:
                try:
                    _, _, clusters, _ = visualizer._fetch_collection_data(collection)
                    
                    for x, y in clusters:    
                        if ((0 <= x < map_x) and (0 <= y < map_y)):
                            occupancy[x, y] += 1

                except Exception:
                    pass

            if (occupancy.max() > 0):
                fig = px.imshow(occupancy.T,
                                origin                 = "lower",
                                color_continuous_scale = "Turbo",
                                aspect                 = "auto",
                                title                  = "Distribution of Medical Knowledge in AI Memory",
                                labels                 = {"x": "Knowledge Area X", "y": "Knowledge Area Y", "color": "Document Density"},
                               )
                
                fig.update_layout(height  = 400,
                                  margin  = dict(l = 20, 
                                                 r = 20, 
                                                 t = 40, 
                                                 b = 20,
                                                ),
                                  title_x = 0.5,
                                 )
                
                st.plotly_chart(figure_or_data      = fig, 
                                use_container_width = True,
                               )

                st.caption("🧠 This map shows how medical knowledge is organized in the AI's memory. Warmer colors indicate areas with more medical documents.")
        
        except Exception as e:
            st.warning(f"Visualization failed: {str(e)}")


    def display_database_overview(self, data_status: Dict):
        """
        Display database statistics visualization
        """
        try:
            data = {"Category" : ["Medical Reports", "Q&A Pairs", "Drug Reviews", "Total"],
                    "Count"    : [data_status.get('reports', 0),
                                  data_status.get('queries', 0),
                                  data_status.get('drug_reviews', 0),
                                  sum(data_status.values())
                                 ],
                    "Color"    : ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
                   }
            
            fig  = px.bar(data_frame          = data,
                          x                   = "Category",
                          y                   = "Count",
                          color               = "Color",
                          color_discrete_map  = "identity",
                          title               = "Medical Database Overview",
                         )
            
            fig.update_layout(showlegend = False,
                              height     = 400,
                              margin     = dict(l = 20, 
                                                r = 20, 
                                                t = 40, 
                                                b = 20,
                                               ),
                             )
            
            st.plotly_chart(figure_or_data      = fig, 
                            use_container_width = True,
                           )

        except Exception as e:
            st.warning(f"Database visualization failed: {repr(e)}")


    def display_ontology_matches(self, extracted_terms: List[str], ontology_lookup: Dict[str, List[str]]):
        """
        Display MeSH / RxNorm / UMLS matches for key terms
        """
        if not extracted_terms:
            st.info("No clinical terms extracted for ontology mapping")
            return

        st.markdown(body = "### 🧬 Ontology-Based Term Mapping")
        for term in extracted_terms:
            st.markdown(body = f"**🔹 {term}**")
            
            matches = ontology_lookup.get(term, [])

            if matches:
                for concept in matches:
                    st.markdown(body = f"- {concept}")
            
            else:
                st.markdown(body = "_No ontology matches found._")


    
