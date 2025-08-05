# DEPENDENCIES
import pickle
import logging
import numpy as np
from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from typing import Optional
from minisom import MiniSom
from .db import MongoHandler
from .config import SOMConfig
from .config import SOM_MODEL_PATH

# SETUP LOGGING
logger = logging.getLogger(__name__)


class SOMClusterer:
    """
    SOM clustering with configuration and monitoring
    """
    def __init__(self, mongo: MongoHandler, config: Optional[Dict] = None) -> None:
        """
        Initialize SOMClusterer with MongoDB handler and configurations

        Arguments:
        ----------
            mongo  { MongoHandler } : MongoDB handler for database interactions

            config  { Dict }        : Configuration dictionary for SOM Parameters, defaults to SOMConfig
        """
        self.mongo            = mongo
        self.config           = config or SOMConfig
        self.som              = None
        self.training_history = []
    

    def _load_all_embeddings(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Load embeddings from all collections with better error handling
        """
        # Define collections to load embeddings from
        collections         = ["reports", "queries", "drug_reviews"]

        # Initialize lists to hold embeddings and references
        all_embeddings      = list()
        document_references = list()


        # Iterate through each collection and load embeddings
        for collection in collections:
            try:
                # Fetch documents with embeddings
                documents = self.mongo.find_many(collection, 
                                                 projection = {"embedding" : 1, 
                                                               "_id"       : 1, 
                                                               "source"    : 1,
                                                              }
                                                )
                # Check if documents are empty
                for document in documents:
                    embedding = document.get("embedding")
                    
                    # Skip if embedding is None
                    if embedding is None:
                        continue
                    
                    try:
                        # Convert embedding to numpy array
                        vector = np.array(embedding, 
                                          dtype = float,
                                         )
                        
                        # Normalize the vector
                        if (vector.size == 0):
                            continue
                    
                        all_embeddings.append(vector)
                        document_references.append({"collection" : collection,
                                                    "_id"        : document["_id"],
                                                    "source"     : document.get("source", collection),
                                                  })
                        
                    except Exception as e:
                        logger.warning(f"Skipping malformed embedding in {collection}: {document.get('_id')}: {repr(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to load embeddings from {collection}: {repr(e)}")
                continue
        
        if not all_embeddings:
            raise ValueError("No valid embeddings found for SOM training")
        
        embeddings_matrix = np.vstack(all_embeddings)
        logger.info(f"Loaded {len(all_embeddings)} embeddings for SOM training")
        
        return embeddings_matrix, document_references
    

    def _assign_clusters(self, embeddings: np.ndarray, document_references: List[Dict[str, Any]]) -> None:
        """
        Assign SOM clusters to documents and update in MongoDB

        Arguments:
        ----------
            embeddings              { np.ndarray }       : Array of embeddings to assign clusters to

            document_references { List[Dict[str, Any]] } : List of document references with collection and ID

        Raises:
        -------
                        RuntimeError                     : If SOM is not trained before assigning clusters

        Returns:
        --------
                             { None }                    : Update documents with cluster information                      
        """
        if self.som is None:
            raise RuntimeError("SOM must be trained before assigning clusters")
        
        # Initialize counters for successful and failed assignments
        successful_assignments = 0
        failed_assignments     = 0
        
        for idx, reference in enumerate(document_references):
            try:
                vector       = embeddings[idx]
                winner       = self.som.winner(vector)  # Returns (x, y)
                cluster_id   = winner[0] * self.config["map_y"] + winner[1]
                
                cluster_info = {"som_cluster": {"x"  : int(winner[0]),
                                                "y"  : int(winner[1]), 
                                                "id" : int(cluster_id),
                                               },
                               }
                
                # Get existing document and update with cluster info
                collection  = reference["collection"]
                document_id = reference["_id"]
                existing    = self.mongo.find_one(collection, {"_id" : document_id})
                
                if existing:
                    existing.update(cluster_info)

                    # Upsert the document with new cluster information
                    if self.mongo.upsert(collection, existing):
                        successful_assignments += 1
                    
                    # Log if the document was updated
                    else:
                        failed_assignments += 1
                        logger.warning(f"Failed to update document {document_id} in {collection}")
 
                else:
                    logger.warning(f"Document {document_id} not found in {collection}")
                    failed_assignments += 1
                    
            except Exception as e:
                logger.error(f"Failed to assign cluster for {reference.get('_id')}: {repr(e)}")
                failed_assignments += 1
        
        logger.info(f"Cluster assignment complete: {successful_assignments} successful, {failed_assignments} failed")
    

    def train(self, num_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Train SOM with comprehensive monitoring
        
        Arguments:
        ----------
            num_iterations { Optional[int] } : Number of training iterations, defaults to config value
        """
        num_iterations = num_iterations or self.config["num_iterations"]
        
        try:
            # Load embeddings
            embeddings, document_references = self._load_all_embeddings()
            data_dimension                  = embeddings.shape[1]
            
            logger.info(f"Training SOM on {embeddings.shape[0]} vectors of dimension {data_dimension}")
            
            # Initialize SOM
            self.som = MiniSom(self.config["map_x"],
                               self.config["map_y"], 
                               data_dimension,
                               sigma         = self.config["sigma"],
                               learning_rate = self.config["learning_rate"],
                               random_seed   = self.config["random_seed"],
                              )
            
            # Initialize weights
            self.som.random_weights_init(embeddings)
            
            # Train with monitoring
            logger.info(f"Starting SOM training for {num_iterations} iterations...")
            self.som.train_random(embeddings, num_iterations)
            
            logger.info("SOM training complete. Assigning clusters to documents...")
            
            # Assign clusters
            self._assign_clusters(embeddings, document_references)
            
            training_result = {"success"             : True,
                               "num_documents"       : len(document_references),
                               "embedding_dimension" : data_dimension,
                               "som_dimensions"      : (self.config["map_x"], self.config["map_y"]),
                               "iterations"          : num_iterations,
                              }
            
            logger.info(f"SOM training successful: {training_result}")
            return training_result
            
        except Exception as e:
            logger.error(f"SOM training failed: {repr(e)}")
            return {"success": False, "error": repr(e)}
    

    def save(self, path: Optional[str] = None) -> bool:
        """
        Save trained SOM model
        """
        if self.som is None:
            logger.error("No trained SOM to save")
            return False
        
        # Use provided path or default path for saving the model 
        save_path = path or str(SOM_MODEL_PATH)
        
        try:
            with open(save_path, "wb") as f:
                pickle.dump(self.som, f)
            
            logger.info(f"SOM model saved to {save_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save SOM model: {e}")
            return False
    

    def load(self, path: Optional[str] = None) -> bool:
        """
        Load pre-trained SOM model
        """
        load_path = path or str(SOM_MODEL_PATH)
        
        try:
            with open(load_path, "rb") as f:
                self.som = pickle.load(f)
            
            logger.info(f"SOM model loaded from {load_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load SOM model: {e}")
            return False
    

    def get_cluster_for_embedding(self, embedding: np.ndarray) -> Optional[Dict[str, int]]:
        """
        Get cluster assignment for a single embedding

        Aeguments:
        ----------
            embedding { np.ndarray } : Embedding vector to find the cluster for

        Returns:
        --------
              { Dict[str, int] }     : Cluster information with keys 'x', 'y', and 'id' if found, None otherwise
        """
        if self.som is None:
            logger.error("SOM not trained or loaded")
            return None
        
        try:
            winner     = self.som.winner(embedding)
            cluster_id = winner[0] * self.config["map_y"] + winner[1]
            
            return {"x"  : int(winner[0]),
                    "y"  : int(winner[1]),
                    "id" : int(cluster_id),
                   }

        except Exception as e:
            logger.error(f"Failed to get cluster for embedding: {repr(e)}")
            return None

