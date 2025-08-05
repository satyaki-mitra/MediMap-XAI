# DEPENDENCIES
import logging
import numpy as np
from typing import List
from typing import Dict
from typing import Optional
from src.db import MongoHandler
from src.search import Retriever
from src.embedder import Embedder
from src.explainer import Explainer

# INITIALIZE LOGGING
logger = logging.getLogger(__name__)


class SearchEngine:
    def __init__(self, mongo: MongoHandler, embedder: Embedder, explainer: Explainer):
        """
        Initialize the SearchEngine with MongoDB handler, embedder, and explainer

        Arguments:
        ----------
            mongo   { MongoHandler } : MongoDB handler for database operations

            embedder  { Embedder }   : Embedder for generating text embeddings

            explainer { Explainer }  : Explainer for generating explanations of search results
        """
        self.mongo      = mongo
        self.embedder   = embedder
        self.retriever  = Retriever(mongo = mongo)
        self.explainer  = explainer
        self.visualizer = None 


    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform semantic search across medical collections
        """
        try:
            # Generate embedding
            query_embedding       = self.embedder.embed([query])[0]
            query_embedding_arr   = np.array(query_embedding, dtype = float)

            # Search across collections
            collections_to_search = ["reports", "queries", "drug_reviews"]
            all_candidates        = list()

            for collection in collections_to_search:
                try:
                    # Retrieve top_k results from each collection
                    results = self.retriever.retrieve_similar(query_embedding = query_embedding_arr.tolist(), 
                                                              collection      = collection, 
                                                              top_k           = top_k//2,
                                                             )
                    
                    # Add source collection to each result
                    for result in results:
                        result["source_collection"] = collection
                        all_candidates.append(result)
                
                except Exception as e:
                    logger.warning(f"Search failed for {collection}: {e}")

            # Deduplicate and rank
            seen     = set()
            combined = list()
            
            # Sort candidates by cosine similarity score 
            for item in sorted(all_candidates, key = lambda x: x.get("cosine_score", 0.0), reverse = True):
                if item.get("id") not in seen:
                    seen.add(item.get("id"))
                    combined.append(item)

            return combined[:top_k]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []


    def get_explanation(self, query: str, result: Dict) -> Dict:
        """
        Generate explanation for a search result
        
        Arguments:
        ----------
            query   { str } : Search query string

            result { Dict } : Result dictionary containing:
                              - 'id': Document ID
                              - 'source_collection': Collection name where the document is stored
                              - 'cosine_score': Similarity score of the result
                              - 'context': Optional context snippet from the document

        Returns:
        --------
                { Dict }    : Explanation dictionary containing:
                              - 'base_similarity': Base similarity score of the result
                              - 'explanation': Explain the result with respect to the query
                              - 'context': Optional context snippet from the document
                              - 'explanation_details': Additional details about the explanation
        """
        try:
            explanation = self.explainer.explain_query_result(query_text        = query, 
                                                              result_id         = result['id'], 
                                                              result_collection = result['source_collection']
                                                             )
            return explanation

        except Exception as e:
            logger.warning(f"Explanation failed: {repr(e)}")
            return {"base_similarity": result.get("cosine_score", 0.0)}