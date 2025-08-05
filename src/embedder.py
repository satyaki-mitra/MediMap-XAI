# DEPENDENCIES
import torch
import logging
import warnings
import numpy as np
from typing import List
from typing import Optional
from .config import EMBEDDING_MODEL_NAME
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# IGNORE FutureWarning
warnings.filterwarnings(action   = 'ignore',
                        category = FutureWarning,
                       )

# INITIALIZE LOGGING
logger = logging.getLogger(__name__)

class Embedder:
    """
    Embedding class of documents using sentence-transformers with caching and batch processing
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device     = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the model
        self._load_model()
    

    def _load_model(self) -> None:
        """
        Load the sentence transformer model
        """
        try:
            self.model = SentenceTransformer(model_name_or_path = self.model_name, 
                                             device             = self.device,
                                            )

            logger.info(f"Loaded embedding model: {self.model_name} on {self.device}")
        
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            # Fallback to a basic model
            try:
                self.model = SentenceTransformer(model_name_or_path = 'all-MiniLM-L6-v2', 
                                                 device             = self.device,
                                                )
                logger.warning("Fallback to all-MiniLM-L6-v2 model")

            except Exception as fallback_e:
                raise RuntimeError(f"Failed to load any embedding model: {repr(fallback_e)}")

    
    def embed(self, texts: List[str], batch_size: int = 32, normalize_embeddings: bool = True) -> np.ndarray:
        """
        Generate embeddings with batch processing
        
        Arguments:
        ----------
            texts             { List[str] } : List of input texts to embed

            batch_size           { int }    : Size of batches for processing, default is 32

            normalize_embeddings { bool }   : Whether to normalize the embeddings, default is True

        Returns:
        --------
                { np.ndarray }              : Array of embeddings, shape (n_samples, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Clean texts
        cleaned_texts = [str(text).strip() for text in texts if text and str(text).strip()]
        
        if not cleaned_texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(sentences            = cleaned_texts,
                                           batch_size           = batch_size,
                                           convert_to_numpy     = True,
                                           show_progress_bar    = len(cleaned_texts) > 100,
                                           normalize_embeddings = False,
                                          )
            
            if normalize_embeddings:
                embeddings = normalize(X    = embeddings, 
                                       norm = 'l2',
                                      )
            
            logger.debug(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {repr(e)}")
            raise
    

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings
        """
        try:
            test_embedding = self.model.encode(["test"], 
                                               convert_to_numpy = True,
                                              )

            return test_embedding.shape[1]
       
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {repr(e)}")
            # Default BERT dimension
            return 768  

