# DEPENDENCIES
import logging
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from minisom import MiniSom
from .db import MongoHandler
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# SETUP LOGGING
logger = logging.getLogger(__name__)


class SOMVisualizer:
    """
    Visualization utilities for SOM analysis
    """
    def __init__(self, som: MiniSom, mongo: MongoHandler):
        """
        Initializes the SOMVisualizer with a MiniSom instance and MongoDB handler

        Arguments:
        ----------
            som      { MiniSom }   : Pre-trained SOM instance to visualize

            mongo { MongoHandler } : MongoDB handler for data retrieval
        """
        self.som               = som
        self.mongo             = mongo
        self.map_x, self.map_y = self._get_som_dimensions()
    

    def _get_som_dimensions(self) -> Tuple[int, int]:
        """
        Get SOM grid dimensions

        Returns:
        --------
            { Tuple[int, int] } : Dimensions of the SOM grid (map_x, map_y)
        """
        try:
            weights = self.som.get_weights()
            # Weights are in shape (map_x, map_y, num_features)
            map_x = weights.shape[0]
            map_y = weights.shape[1]
            logger.info(f"SOM dimensions: {map_x, map_y}")

            return map_x, map_y
        
        except Exception as e:
            logger.error(f"Failed to get SOM dimensions: {repr(e)}")
            # Return default fallback
            return 12, 12  
    

    def _fetch_collection_data(self, collection: str) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]]]:
        """
        Fetch embeddings, IDs, and cluster coordinates for a collection

        Arguments:
        ----------
            collection        { str }                               : Name of the collection to fetch data from

        Returns:
        --------
            { Tuple[np.ndarray, List[str], List[Tuple[int, int]]] } : A tuple of embeddings, ids and cluster coordinates
        """
        try:
            documents  = self.mongo.find_many(collection,
                                              projection = {"embedding"   : 1, 
                                                            "_id"         : 1, 
                                                            "som_cluster" : 1, 
                                                            "clean_text"  : 1,
                                                           }
                                             )
            
            embeddings = list()
            ids        = list()
            clusters   = list()
            texts      = list()
            
            # Iterate through documents and extract relevant data 
            for doc in documents:
                embedding = doc.get("embedding")
                cluster   = doc.get("som_cluster", {})
                
                # Skip if no embedding or cluster data
                if not embedding or not cluster:
                    continue
                
                try:
                    embeddings.append(np.array(embedding, 
                                               dtype = float,
                                              )
                                     )

                    ids.append(doc["_id"])
                    clusters.append((cluster.get("x", -1), cluster.get("y", -1)))
                    texts.append(doc.get("clean_text", ""))

                except Exception as e:
                    logger.warning(f"Skipping document {doc.get('_id')}: {repr(e)}")
            
            # If no valid embeddings found, return empty arrays
            if not embeddings:
                return np.array([]), [], [], []
            
            return np.vstack(embeddings), ids, clusters, texts
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {collection}: {repr(e)}")
            return np.array([]), [], [], []
    

    def plot_cluster_occupancy(self, collection: str, save_path: Optional[str] = None) -> np.ndarray:
        """
        Create and optionally save cluster occupancy heatmap

        Arguments:
        ----------
            collection { str } : Name of the collection to visualize 

            save_path  { str } : Path to save the plot, if None the plot will not be saved

        Returns:
        --------
            { np.ndarray }     : 2D occupancy grid array where each cell contains the count of documents in that cluster
        """ 
        try:
            _, _, clusters, _ = self._fetch_collection_data(collection)
            
            if not clusters:
                logger.warning(f"No cluster data found for {collection}")
                return np.zeros((self.map_x, self.map_y))
            
            # Create occupancy grid
            grid = np.zeros((self.map_x, self.map_y))
            
            for x, y in clusters:
                if ((0 <= x < self.map_x) and (0 <= y < self.map_y)):
                    grid[x, y] += 1
            
            # Create plot
            plt.figure(figsize = (10, 8))
            plt.title(f"Cluster Occupancy: {collection.title()}")
            
            im  = plt.imshow(grid.T, 
                             origin = "lower", 
                             aspect = "auto", 
                             cmap   = "YlOrRd",
                            )

            plt.colorbar(im, label = "Document Count")
            plt.xlabel(xlabel = "SOM X")
            plt.ylabel(ylabel = "SOM Y")
            
            # Add text annotations for non-zero cells
            for i in range(self.map_x):
                for j in range(self.map_y):
                    if (grid[i, j] > 0):
                        plt.text(x        = i, 
                                 y        = j, 
                                 s        = f'{int(grid[i, j])}', 
                                 ha       = 'center', 
                                 va       = 'center', 
                                 fontsize = 8,
                                )
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(fname       = save_path, 
                            dpi         = 300,
                            bbox_inches = 'tight',
                           )

                logger.info(f"Occupancy plot saved to {save_path}")
            
            plt.show()
            return grid
            
        except Exception as e:
            logger.error(f"Failed to create occupancy plot for {collection}: {repr(e)}")
            return np.zeros((self.map_x, self.map_y))
    

    def plot_u_matrix(self, save_path: Optional[str] = None) -> np.ndarray:
        """
        Create U-Matrix showing cluster boundaries

        Arguments:
        ----------
            save_path { str } : Path to save the U-matrix plot, if None the plot will not be saved

        Returns:
        --------
            { np.ndarray }    : 2D array representing the U-matrix 
        """
        try:
            u_matrix = np.zeros((self.map_x, self.map_y))
            weights  = self.som.get_weights()
            
            for i in range(self.map_x):
                for j in range(self.map_y):
                    neighbor_distances = list()
                    
                    # Check all 8 neighbors
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if ((dx == 0) and (dy == 0)):
                                continue
                            
                            ni = i + dx
                            nj = j + dy
                            # Ensure neighbor indices are within bounds
                            if ((0 <= ni < self.map_x) and (0 <= nj < self.map_y)):
                                distance = np.linalg.norm(weights[i, j] - weights[ni, nj])
                                neighbor_distances.append(distance)
                    
                    u_matrix[i, j] = np.mean(neighbor_distances) if neighbor_distances else 0
            
            # Create plot
            plt.figure(figsize = (10, 8))
            plt.title("U-Matrix: Cluster Boundaries")
            
            im = plt.imshow(u_matrix.T, 
                            origin = "lower", 
                            aspect = "auto", 
                            cmap   = "RdYlBu_r",
                           )

            plt.colorbar(im, label = "Average Distance to Neighbors")
            plt.xlabel(xlabel = "SOM X")
            plt.ylabel(ylabel = "SOM Y")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(fname       = save_path, 
                            dpi         = 300, 
                            bbox_inches = 'tight',
                           )

                logger.info(f"U-Matrix saved to {save_path}")
            
            plt.show()
            return u_matrix
            
        except Exception as e:
            logger.error(f"Failed to create U-Matrix: {repr(e)}")
            return np.zeros((self.map_x, self.map_y))
    

    def get_cell_keywords(self, x: int, y: int, top_n: int = 5) -> List[str]:
        """
        Get top keywords for a specific SOM cell

        Arguments:
        ----------
           x     { int }  : X-coordinate of the SOM cell

           y     { int }  : Y-coordinate of the SOM cell 
          
           top_n { int }  : Number of top keywords to return, default is 5 

        Returns:
        --------
            { List[str] } : List of top keywords for the specified cell
        """
        try:
            # Get all documents in this cell
            collections = ["reports", "queries", "drug_reviews"]
            cell_texts  = list()
            
            for collection in collections:
                docs = self.mongo.find_many(collection,
                                            query      = {"som_cluster.x" : x, 
                                                          "som_cluster.y" : y,
                                                         },
                                            projection = {"clean_text" : 1}
                                           )

                cell_texts.extend([doc.get("clean_text", "") for doc in docs])
            
            if not cell_texts:
                return []
            
            # Extract keywords using TF-IDF
            vectorizer    = TfidfVectorizer(stop_words   = "english",
                                            max_features = 5000,
                                            ngram_range  = (1, 5),
                                           )
            
            tfidf_matrix  = vectorizer.fit_transform(cell_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get mean TF-IDF scores
            mean_scores   = np.array(tfidf_matrix.mean(axis = 0)).flatten()
            top_indices   = mean_scores.argsort()[-top_n:][::-1]
            
            return [feature_names[i] for i in top_indices]
            
        except Exception as e:
            logger.error(f"Failed to get keywords for cell ({x}, {y}): {repr(e)}")
            return []
    

    def create_keyword_overlay(self, save_path: Optional[str] = None) -> Dict[Tuple[int, int], List[str]]:
        """
        Create keyword overlay for entire SOM

        Arguments:
        ----------
            save_path          { str }           : The path to save the keyword overlay plot, if None the plot will not be saved

        Returns:
        --------
            { Dict[Tuple[int, int], List[str]] } : A dictionary mapping cell coordinates to lists of top keywords
        """
        try:
            keyword_map = dict()
            
            for i in range(self.map_x):
                for j in range(self.map_y):
                    keywords = self.get_cell_keywords(i, j, top_n = 3)
                    if keywords:
                        keyword_map[(i, j)] = keywords
            
            # Create visualization
            plt.figure(figsize = (15, 12))
            plt.title("SOM Keyword Overlay")
            
            # Create empty grid
            grid = np.zeros((self.map_x, 
                             self.map_y,
                           ))
            
            # Add keywords as text
            for (x, y), keywords in keyword_map.items():
                if keywords:
                    keyword_text = "\n".join(keywords[:2])  # Top 2 keywords
                    plt.text(x  = x, 
                             y  = y, 
                             s  = keyword_text, 
                             ha = 'center', 
                             va = 'center', 
                             fontsize = 8, 
                             wrap     = True, 
                             bbox     = dict(boxstyle  = "round,pad=0.3", 
                                             facecolor = 'lightblue', 
                                             alpha     = 0.7,
                                            ),
                            )
                    grid[x, y] = 1  # Mark cells with content
            
            plt.imshow(grid.T, 
                       origin = "lower", 
                       aspect = "auto", 
                       alpha  = 0.3, 
                       cmap   = "Greys",
                      )

            plt.xlabel(xlabel = "SOM X")
            plt.ylabel(ylabel = "SOM Y")
            plt.grid(True, alpha = 0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(fname       = save_path, 
                            dpi         = 300, 
                            bbox_inches = 'tight',
                           )
                logger.info(f"Keyword overlay saved to {save_path}")
            
            plt.show()
            return keyword_map
            
        except Exception as e:
            logger.error(f"Failed to create keyword overlay: {repr(e)}")
            return {}
    

    def analyze_cluster_composition(self) -> Dict[str, Any]:
        """
        Analyze composition of clusters across collections

        Returns:
        --------
            { Dict[str, Any] } : A dictionary containing analysis results including total active cells, 
                                 average documents per cell, maximum documents per cell, and composite 
                                 composition by cell
        """
        try:
            collections = ["reports", "queries", "drug_reviews"]
            composition = dict()
            
            for collection in collections:
                _, _, clusters, _ = self._fetch_collection_data(collection)
                
                for x, y in clusters:
                    if ((0 <= x < self.map_x) and (0 <= y < self.map_y)):
                        cell_key = (x, y)
                        if cell_key not in composition:
                            composition[cell_key] = {"total": 0}
                        
                        composition[cell_key]["total"]   += 1
                        composition[cell_key][collection] = composition[cell_key].get(collection, 0) + 1
            
            # Calculate diversity metrics
            analysis = {"total_active_cells"     : len(composition),
                        "avg_documents_per_cell" : np.mean([cell["total"] for cell in composition.values()]),
                        "max_documents_per_cell" : max([cell["total"] for cell in composition.values()]),
                        "composition_by_cell"    : composition,
                       }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze cluster composition: {repr(e)}")
            return {"error": repr(e)}

