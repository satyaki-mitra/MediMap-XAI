# DEPENDENCIES
import logging
from typing import Dict
from typing import Tuple
from src.db import MongoHandler
from src.search import Retriever
from src.embedder import Embedder
from src.explainer import Explainer
from src.som_clusterer import SOMClusterer
from core.search_engine import SearchEngine

# INITIALIZE LOGGING
logger = logging.getLogger(__name__)


class SystemManager:
    def __init__(self):
        self.mongo         = None
        self.embedder      = None
        self.retriever     = None
        self.explainer     = None
        self.som_clusterer = None
        self.search_engine = None


    def initialize_system(self) -> SearchEngine:
        """
        Initialize all system components
        """
        try:
            self.mongo         = MongoHandler()
            self.embedder      = Embedder()
            self.retriever     = Retriever(mongo = self.mongo)
            self.som_clusterer = SOMClusterer(mongo = self.mongo)
            self.explainer     = Explainer(mongo    = self.mongo, 
                                           embedder = self.embedder)
            
            # Load or train SOM model
            try:
                self.som_clusterer.load("models/som_model.pkl")
                logger.info("Loaded existing SOM model")

            except Exception:
                logger.info("Training new SOM model")
                training_result = self.som_clusterer.train(num_iterations = 500)

                if training_result.get("success"):
                    self.som_clusterer.save("models/som_model.pkl")
            
            # Create search engine
            self.search_engine = SearchEngine(mongo     = self.mongo, 
                                              embedder  = self.embedder,
                                              explainer = self.explainer,
                                             )
            return self.search_engine
        
        except Exception as e:
            logger.error(f"System initialization failed: {repr(e)}")
            raise


    def check_data_availability(self) -> Dict:
        """
        Check if we have data in the database
        """
        collections = ["reports", "queries", "drug_reviews"]
        data_status = dict()
        
        for collection in collections:
            count                   = self.mongo.count_documents(collection)
            data_status[collection] = count
        
        return data_status
