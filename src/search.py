# DEPENDENCIES
import logging
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from .db import MongoHandler
from .config import SearchConfig
from sklearn.metrics.pairwise import cosine_similarity


# INITIALIZE LOGGING
logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieval with multi-collection search and combined scoring
    """
    def __init__(self, mongo: MongoHandler):
        """
        Initialize the retriever with MongoDB handler

        mongo  { MongoHandler } : MongoDB handler instance
        """
        self.mongo  = mongo
        self.config = SearchConfig
    

    def retrieve_similar(self, query_embedding: List[float], collection: str, top_k: int = None, include_cluster_score: bool = False) -> List[Dict[str, Any]]:
        """
        Enhanced similarity search with optional cluster scoring

        Arguments:
        ----------
            query_embedding    { List[float ] }  : Embedding of the query text

            collection             { str }       : Name of the collection to search in

            top_k                  { int }       : Number of top results to return

            include_cluster_score { bool }       : Whether to include SOM cluster information in results

        Returns:
        --------
                { List[Dict[str, Any]] }         : List of dictionaries containing document IDs, cosine scores, snippets, and optional cluster info
        """
        # Set default value for top_k
        top_k = top_k or self.config.top_k_default
        
        try:
            # Get documents with embeddings and cluster info
            projection = {"embedding"   : 1, 
                          "clean_text"  : 1,
                          "_id"         : 1,
                          "som_cluster" : 1,
                          "source"      : 1,
                         }
            
            # Retrieve documents from the specified collection
            docs       = self.mongo.find_many(collection, 
                                              projection = projection,
                                             )

            # Check if any documents were found
            if not docs:
                return []
            
            # Filter out documents without embeddings
            valid_docs = [doc for doc in docs if doc.get("embedding")]
            if not valid_docs:
                return []
            
            # Calculate cosine similarities
            embeddings   = np.array([doc["embedding"] for doc in valid_docs])
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            
            results      = list()

            # Prepare results with cosine scores and snippets
            for i, doc in enumerate(valid_docs):
                result = {"id"           : doc["_id"],
                          "collection"   : collection,
                          "cosine_score" : float(similarities[i]),
                          "snippet"      : doc.get("clean_text", "")[:self.config.max_snippet_length],
                          "source"       : doc.get("source", collection),
                         }

                # Add cluster information if available
                if (doc.get("som_cluster") and include_cluster_score):
                    result["cluster"] = doc["som_cluster"]
                
                results.append(result)
            
            # Sort by cosine similarity
            results.sort(key     = lambda x: x["cosine_score"], 
                         reverse = True,
                        )
            
            # Limit to top_k results
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Retrieval failed for {collection}: {repr(e)}")
            return []
            

    def _calculate_cluster_boost(self, query_cluster: Dict[str, int], doc_cluster: Optional[Dict[str, int]]) -> float:
        """
        Calculate cluster proximity boost score

        Arguments:
        ----------
            query_cluster { Dict[str, int ] } : Query's SOM Cluster coordinates

            doc_cluster   { Dict[str, int ] } : Document's SOM Cluster coordinates
        
        Returns:
        --------
                      { float }               : Cluster boost score between 0.0 and 1.0
        """
        if not doc_cluster or not query_cluster:
            return 0.0
        
        try:
            # Calculate Euclidean distance in SOM grid
            dx                  = query_cluster.get("x", 0) - doc_cluster.get("x", 0)
            dy                  = query_cluster.get("y", 0) - doc_cluster.get("y", 0)
            distance            = np.sqrt(dx**2 + dy**2)
            
            # Convert distance to similarity (closer = higher score)
            # Normalize by max possible distance in a 12x12 grid
            max_distance        = np.sqrt(12**2 + 12**2)
            normalized_distance = distance / max_distance
            cluster_similarity  = 1.0 - normalized_distance
            
            return max(0.0, cluster_similarity)
            
        except Exception as e:
            logger.warning(f"Cluster boost calculation failed: {repr(e)}")
            return 0.0


    def multi_collection_search(self, query_embedding: List[float], collections: Optional[List[str]] = None, top_k_per_collection: int = 3, final_top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search across multiple collections and combine results

        Arguments:
        ----------
            query_embedding     { List[float] } : Embedding of the query text

            collections         { List[str] }   : List of collection names to search in

            top_k_per_collection   { int }      : Number of top results to retrieve from each collection

            final_top_k            { int }      : Final number of results to return after combining all collections

        Returns:
        --------
                { List[Dict[str, Any ]] }       : Combined results from all collections, sorted by cosine score
        """
        # Set default collections and top_k values
        collections = collections or ["reports", "queries", "drug_reviews"]
        final_top_k = final_top_k or self.config.top_k_default
        
        all_results = list()
        
        # Iterate through each collection and retrieve results
        for collection in collections:
            try:
                results = self.retrieve_similar(query_embedding       = query_embedding, 
                                                collection            = collection, 
                                                top_k                 = top_k_per_collection,
                                                include_cluster_score = True,
                                               )
                
                all_results.extend(results)

            except Exception as e:
                logger.warning(f"Search failed for collection {collection}: {e}")
        
        # Remove duplicates and sort by score
        seen_ids       = set()
        unique_results = list()
        
        for result in sorted(all_results, key = lambda x: x["cosine_score"], reverse = True):
            if (result["id"] not in seen_ids):
                seen_ids.add(result["id"])
                unique_results.append(result)
        
        return unique_results[:final_top_k]
    

    def search_with_cluster_boost(self, query_embedding: List[float], query_cluster: Optional[Dict[str, int]] = None, collections: Optional[List[str]] = None, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Enhanced search with cluster proximity boosting

        Arguments:
        ----------
            query_embedding      { List[float] }  : Embedding of the query text

            query_cluster      { Dict[str, int] } : SOM cluster coordinates of the query

            collections           { List[str] }   : List of collection names to search in

            top_k                    { int }      : Number of top results to return

        Returns:
        --------
                 { List[Dict[str, Any]] }         : Combined results from all collections, sorted by combined score
        """
        # Set default collections and top_k values
        collections = collections or ["reports", "queries", "drug_reviews"]
        top_k       = top_k or self.config.top_k_default
        
        all_results = []
        
        # Iterate through each collection and retrieve results
        for collection in collections:
            results = self.retrieve_similar(query_embedding       = query_embedding, 
                                            collection            = collection, 
                                            top_k                 = top_k * 2,  # Get more candidates for cluster boosting
                                            include_cluster_score = True,
                                           )

            all_results.extend(results)
        
        # Apply cluster boosting if query cluster is provided
        if query_cluster:
            for result in all_results:
                # Ensure result has cluster information
                cluster_boost            = self._calculate_cluster_boost(query_cluster = query_cluster, 
                                                                         doc_cluster   = result.get("cluster"),
                                                                       )

                # Combine scores with cluster boost                                             
                combined_score           = (result["cosine_score"] * self.config.similarity_weight + cluster_boost * self.config.cluster_weight)
                result["combined_score"] = combined_score
                result["cluster_boost"]  = cluster_boost
            
            # Sort by combined score
            all_results.sort(key     = lambda x: x.get("combined_score", x["cosine_score"]), 
                             reverse = True,
                            )

        else:
            # Sort by cosine score only
            all_results.sort(key=lambda x: x["cosine_score"], reverse=True)
        
        # Remove duplicates
        seen_ids       = set()
        unique_results = list()
        
        # Iterate through sorted results and filter out duplicates
        for result in all_results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                unique_results.append(result)
        
        return unique_results[:top_k]
    

    

