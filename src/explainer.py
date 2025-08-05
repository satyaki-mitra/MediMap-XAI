# DEPENDENCIES
import logging
import numpy as np
from typing import Any
from typing import List
from typing import Dict 
from typing import Tuple
from typing import Optional
from .db import MongoHandler
from .utils import clean_text
from .embedder import Embedder
from sklearn.metrics.pairwise import cosine_similarity

# SETUP LOGGING
logger = logging.getLogger(__name__)


class Explainer:
    """
    Enhanced explanation system with multiple explanation methods
    """
    def __init__(self, mongo: MongoHandler, embedder: Embedder):
        self.mongo    = mongo
        self.embedder = embedder
    
    
    def _categorize_impact(self, impact: float) -> str:
        """
        Categorize token impact
        """
        if (impact > 0.05):
            return "strongly_positive"

        elif (impact > 0.02):
            return "positive"

        elif (impact > -0.02):
            return "neutral"

        elif (impact > -0.05):
            return "negative"

        else:
            return "strongly_negative"


    def _analyze_token_importance(self, query_text: str, result_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """
        Analyze importance of individual tokens using leave-one-out

        Arguments:
        ----------
            query_text          { str }     : Query text to analyze

            result_embedding { np.ndarray } : Embedding of the result document to compare against

            top_k                { int }    : Number of top tokens to return based on importance

        Returns:
        --------
                { List[Dict[str, Any]] }    : List of dictionaries containing:
                                              - token,
                                              - importance score,
                                              - impact category
        """
        # Split query into tokens
        tokens = query_text.split()

        # Validate tokens
        if (len(tokens) <= 1):
            return [{"token"      : query_text, 
                     "importance" : 1.0, 
                     "impact"     : "high",
                   }]
        
        # Get baseline similarity
        baseline_embedding  = self.embedder.embed([query_text])[0]
        baseline_similarity = float(cosine_similarity([baseline_embedding], [result_embedding])[0][0])
        
        token_impacts       = list()

        # Iterate over tokens and calculate impact
        for i, token in enumerate(tokens):
            try:
                # Create modified query without this token
                modified_tokens     = tokens[:i] + tokens[i+1:]
                modified_query      = " ".join(modified_tokens)
                
                # Skip empty queries
                if not modified_query.strip():
                    continue
                
                # Get embedding for modified query
                modified_embedding  = self.embedder.embed([modified_query])[0]
                modified_similarity = float(cosine_similarity([modified_embedding], [result_embedding])[0][0])
                
                # Calculate impact (positive = token helped, negative = token hurt)
                impact              = baseline_similarity - modified_similarity
                
                token_impacts.append({"token"           : token,
                                      "importance"      : impact,
                                      "impact_category" : self._categorize_impact(impact),
                                    })
                
            except Exception as e:
                logger.warning(f"Failed to analyze token '{token}': {repr(e)}")
                continue
        
        # Sort by importance and return top k
        token_impacts.sort(key     = lambda x: abs(x["importance"]), 
                           reverse = True,
                          )
        
        # Limit to top k tokens
        return token_impacts[:top_k]

    
    def _analyze_cluster_relationship(self, query_embedding: np.ndarray, result_cluster: Optional[Dict]) -> Dict[str, Any]:
        """
        Analyze cluster-based relationship
        
        Arguments:
        ----------
            query_embedding   { np.ndarray }   : Embedding of the query to analyze

            result_cluster  { Optional[Dict] } : Cluster information of the result document, if available

        Returns:
        --------
                     { Dict[str, Any] }        : Analysis dictionary containing:
                                                 - result_cluster,
                                                 - cluster_x,
                                                 - cluster_y,
                                                 - cluster_id,
                                                 - cluster_size,
                                                 - cluster_neighbours
        """
        if not result_cluster:
            return {"status" : "no_cluster_info"}
        
        try:
            # Get cluster for query (we need SOM for this)
            # For now, just return cluster info
            analysis = {"result_cluster" : result_cluster,
                        "cluster_x"      : result_cluster.get("x", -1),
                        "cluster_y"      : result_cluster.get("y", -1),
                        "cluster_id"     : result_cluster.get("id", -1),
                       }
            
            # Get other documents in same cluster for context
            same_cluster_docs             = self._get_cluster_neighbors(result_cluster)
            analysis["cluster_size"]      = len(same_cluster_docs)
            analysis["cluster_neighbors"] = same_cluster_docs[:3]  # Top 3 neighbors
            
            return analysis
            
        except Exception as e:
            logger.error(f"Cluster analysis failed: {repr(e)}")
            return {"status" : "error", 
                    "error"  : repr(e),
                   }
    

    def _get_cluster_neighbors(self, cluster_info: Dict) -> List[Dict[str, Any]]:
        """
        Get other documents in the same cluster

        Arguments:
        ----------
            cluster_info { Dict } : Information about the cluster to find neighbors for

        Returns:
        --------
                  { List[Dict[str, Any]] } : List of dictionaries containing:
                                             - id,
                                             - collection,
                                             - snippet,
                                             - source
        """
        try:
            collections = ["reports", "queries", "drug_reviews"]
            neighbors   = list()
            
            for collection in collections:
                docs = self.mongo.find_many(collection,
                                            query      = {"som_cluster" : cluster_info},
                                            projection = {"_id"        : 1, 
                                                          "clean_text" : 1, 
                                                          "source"     : 1,
                                                         },
                                            limit      = 10,
                                           )
                
                for doc in docs:
                    neighbors.append({"id"         : doc["_id"],
                                      "collection" : collection,
                                      "snippet"    : doc.get("clean_text", "")[:100],
                                      "source"     : doc.get("source", collection),
                                    })
            
            return neighbors
            
        except Exception as e:
            logger.error(f"Failed to get cluster neighbors: {repr(e)}")
            return []
    

    def _categorize_similarity(self, similarity: float) -> str:
        """
        Categorize similarity score

        Arguments:
        ----------
            similarity { float } : Similarity score to categorize

        Returns:
        -------- 
               { str }           : Category of similarity according to the score
        """
        if (similarity >= 0.8):
            return "very_high"

        elif (similarity >= 0.6):
            return "high" 

        elif (similarity >= 0.4):
            return "medium"

        elif (similarity >= 0.2):
            return "low"

        else:
            return "very_low"


    def explain_query_result(self, query_text: str, result_id: str, result_collection: str, top_k_tokens: int = 5) -> Dict[str, Any]:
        """
        Comprehensive explanation of why a result matches a query
        
        Arguments:
        ----------
            query_text        { str } : The query text to explain

            result_id         { str } : The ID of the result document to explain

            result_collection { str } : The collection where the result document is stored

        Results:
        --------
              { Dict[str, Any] }      : Explanation dictionary containing: 
                                        - base similarity score,
                                        - token-level importance,
                                        - cluster analysis,
                                        - document metadata,
                                        - similarity category,
                                        - result snippet 
        """
        try:
            # Get the result document
            result_doc = self.mongo.find_one(result_collection, {"_id": result_id})
            
            if not result_doc:
                return {"error" : f"Result document {result_id} not found"}
            
            # Generate query embedding
            clean_query = clean_text(query_text)
            if not clean_query:
                return {"error" : "Invalid query text"}
            
            # Generate embedding for the query
            query_embedding  = self.embedder.embed([clean_query])[0]

            # Get embedding for the result document
            result_embedding = np.array(result_doc.get("embedding", []))
            
            # Check if result embedding is valid
            if (result_embedding.size == 0):
                return {"error": "Result document has no embedding"}
            
            # Base similarity
            base_similarity  = float(cosine_similarity([query_embedding], [result_embedding])[0][0])
            
            # Token-level importance analysis
            token_impacts    = self._analyze_token_importance(query_text       = clean_query, 
                                                              result_embedding = result_embedding, 
                                                              top_k            = top_k_tokens,
                                                             )
            
            # Cluster analysis
            cluster_analysis = self._analyze_cluster_relationship(query_embedding, 
                                                                  result_doc.get("som_cluster")
                                                                 )
            
            # Document metadata
            doc_metadata     = {"source"            : result_doc.get("source", "unknown"),
                                "text_length"       : len(result_doc.get("clean_text", "")),
                                "medical_specialty" : result_doc.get("medical_specialty", ""),
                                "drug_name"         : result_doc.get("drug_name", ""),
                                "condition"         : result_doc.get("condition", ""),
                               }
            
            explanation      = {"query_text"          : query_text,
                                "result_id"           : result_id,
                                "result_collection"   : result_collection,
                                "base_similarity"     : base_similarity,
                                "similarity_category" : self._categorize_similarity(base_similarity),
                                "token_importance"    : token_impacts,
                                "cluster_analysis"    : cluster_analysis,
                                "document_metadata"   : doc_metadata,
                                "result_snippet"      : result_doc.get("clean_text", "")[:300],
                               }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Explanation failed for {result_id}: {repr(e)}")
            return {"error": str(e)}
    

    def explain_search_results(self, query_text: str, results: List[Dict[str, Any]], top_explanations: int = 3) -> Dict[str, Any]:
        """
        Explain a set of search results
        """
        explanations = list()
        
        for i, result in enumerate(results[:top_explanations]):
            explanation                 = self.explain_query_result(query_text,
                                                                    result["id"],
                                                                    result["collection"],
                                                                   )
            explanation["rank"]         = i + 1
            explanation["search_score"] = result.get("cosine_score", 0.0)
            explanations.append(explanation)
        
        # Overall analysis
        overall_analysis = {"query_text"       : query_text,
                            "num_results"      : len(results),
                            "num_explained"    : len(explanations),
                            "avg_similarity"   : np.mean([exp.get("base_similarity", 0) for exp in explanations]),
                            "result_diversity" : self._analyze_result_diversity(results),
                            "explanations"     : explanations,
                           }
        
        return overall_analysis
    

    def _analyze_result_diversity(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze diversity of search results
        """
        try:
            collections       = [r.get("collection", "unknown") for r in results]
            sources           = [r.get("source", "unknown") for r in results]
            
            collection_counts = dict()
            source_counts     = dict()
            
            for col in collections:
                collection_counts[col] = collection_counts.get(col, 0) + 1
            
            for src in sources:
                source_counts[src] = source_counts.get(src, 0) + 1
            
            return {"collection_distribution" : collection_counts,
                    "source_distribution"     : source_counts,
                    "diversity_score"         : len(set(collections)) / len(collections) if collections else 0,
                   }
            
        except Exception as e:
            logger.error(f"Diversity analysis failed: {repr(e)}")
            return {"error": str(repr(e))}

