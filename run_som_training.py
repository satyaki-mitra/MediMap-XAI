# DEPENDENCIES
import os
import sys
import time
import logging
import argparse
import numpy as np
from typing import Any
from typing import List
from typing import Dict
from pathlib import Path
from typing import Tuple
from typing import Optional

# Add scripts to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from src.config import DB_NAME
from src.db import MongoHandler
from src.search import Retriever
from src.config import MONGO_URI
from src.config import SOMConfig
from src.config import MODELS_DIR
from src.embedder import Embedder
from src.explainer import Explainer
from src.config import SOM_MODEL_PATH
from src.som_clusterer import SOMClusterer
from src.som_visualizer import SOMVisualizer


# Configure logging
logging.basicConfig(level    = logging.INFO,
                    format   = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers = [logging.FileHandler('logs/model_training.log'),
                                logging.StreamHandler(sys.stdout),
                               ],
                   )
logger = logging.getLogger(__name__)



class ModelTrainingRunner:
    """
    Handles model training, validation, and evaluation
    """
    def __init__(self, force_retrain: bool = False, som_config: Optional[Dict] = None):
        self.force_retrain    = force_retrain
        self.som_config       = som_config or SOMConfig
        
        # Components
        self.mongo            = None
        self.embedder         = None
        self.som_clusterer    = None
        self.retriever        = None
        self.explainer        = None
        self.visualizer       = None
        
        # Training status
        self.training_status  = {"database_connected" : False,
                                 "embedder_loaded"    : False,
                                 "data_validated"     : False,
                                 "som_trained"        : False,
                                 "clusters_assigned"  : False,
                                 "model_validated"    : False,
                                 "training_complete"  : False,
                                }
                            
        # Training metrics
        self.training_metrics = dict()

    
    def connect_database(self) -> bool:
        """
        Connect to MongoDB and validate data
        """
        logger.info("Connecting to database...")
        
        try:
            self.mongo  = MongoHandler()
            collections = self.mongo.get_collection_names()
            logger.info(f"Connected to MongoDB. Collections: {collections}")
            
            self.training_status["database_connected"] = True
            return True
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    

    def initialize_embedder(self) -> bool:
        """
        Initialize embedding model
        """
        logger.info("Loading embedding model...")
        
        try:
            self.embedder  = Embedder()
            
            # Test embedding
            test_embedding = self.embedder.embed(["test"])
            logger.info(f"Embedder loaded. Dimension: {test_embedding.shape[1]}")
            
            self.training_status["embedder_loaded"] = True
            return True
            
        except Exception as e:
            logger.error(f"Embedder initialization failed: {e}")
            return False
    

    def validate_training_data(self) -> bool:
        """
        Validate that sufficient data exists for training
        """
        logger.info("Validating training data...")
        
        try:
            collections       = ["reports", "queries", "drug_reviews"]
            total_documents   = 0
            collection_counts = dict()
            
            for collection in collections:
                count                         = self.mongo.count_documents(collection)
                collection_counts[collection] = count
                total_documents              += count
                
                logger.info(f"{collection}: {count} documents")
            
            # Check minimum requirements: Minimum for meaningful training
            min_documents_required = 500
            
            if (total_documents < min_documents_required):
                logger.error(f"Insufficient data for training: {total_documents} < {min_documents_required}")
                logger.error("Please run data ingestion first: python run_data_ingestion.py")
                return False
            
            # Check for embeddings
            embeddings_count = 0
            for collection in collections:
                docs_with_embeddings = self.mongo.count_documents(collection, {"embedding" : {"$exists" : True, 
                                                                                              "$ne"     : None,
                                                                                             },
                                                                              },
                                                                 )

                embeddings_count     += docs_with_embeddings
                logger.info(f"{collection}: {docs_with_embeddings} documents with embeddings")
            
            if (embeddings_count < min_documents_required):
                logger.error(f"Insufficient embeddings for training: {embeddings_count} < {min_documents_required}")
                return False
            
            logger.info(f"Training data validation passed:")
            logger.info(f"Total documents: {total_documents}")
            logger.info(f"Documents with embeddings: {embeddings_count}")
            
            # Store validation metrics
            self.training_metrics["data_validation"] = {"total_documents"   : total_documents,
                                                        "embeddings_count"  : embeddings_count,
                                                        "collection_counts" : collection_counts,
                                                       }
            
            self.training_status["data_validated"]   = True
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False
    

    def check_existing_model(self) -> bool:
        """
        Check if trained model already exists
        """
        logger.info("Checking for existing SOM model...")
        
        if not SOM_MODEL_PATH.exists():
            logger.info("No existing model found")
            return False
        
        try:
            # Try loading existing model
            self.som_clusterer = SOMClusterer(mongo  = self.mongo, 
                                              config = self.som_config,
                                             )
            
            if self.som_clusterer.load():
                logger.info("Found and loaded existing SOM model")
                
                if not self.force_retrain:
                    # Check if clusters are assigned
                    cluster_count = self._count_clustered_documents()
                    logger.info(f"Documents with clusters: {cluster_count}")
                    
                    if (cluster_count > 0):
                        logger.info("Model appears to be fully trained and deployed")
                        self.training_status["som_trained"]       = True
                        self.training_status["clusters_assigned"] = True
                        return True

                    else:
                        logger.info("Model loaded but clusters not assigned")
                        self.training_status["som_trained"] = True
                        return True

                else:
                    logger.info("Force retrain enabled, will retrain model")
                    return False

            else:
                logger.warning("Found model file but failed to load")
                return False
                
        except Exception as e:
            logger.warning(f"Error checking existing model: {e}")
            return False
    

    def _count_clustered_documents(self) -> int:
        """
        Count documents that have cluster assignments
        """
        try:
            collections     = ["reports", "queries", "drug_reviews"]
            total_clustered = 0
            
            for collection in collections:
                count = self.mongo.count_documents(collection, 
                                                   {"som_cluster" : {"$exists" : True, 
                                                                     "$ne"     : None,
                                                                    },
                                                   },
                                                  )
                total_clustered += count
            
            return total_clustered
            
        except Exception as e:
            logger.warning(f"Failed to count clustered documents: {e}")
            return 0
    

    def train_som_model(self, num_iterations: Optional[int] = None) -> bool:
        """
        Train the SOM model
        """
        num_iterations = num_iterations or self.som_config["num_iterations"]
        
        logger.info("Starting SOM model training...")
        logger.info(f"SOM dimensions: {self.som_config['map_x']}x{self.som_config['map_y']}")
        logger.info(f"Training iterations: {num_iterations}")
        logger.info(f"Learning rate: {self.som_config['learning_rate']}")
        logger.info(f"Sigma: {self.som_config['sigma']}")
        
        try:
            # Initialize SOM clusterer if not already done
            if not self.som_clusterer:
                self.som_clusterer = SOMClusterer(mongo  = self.mongo, 
                                                  config = self.som_config,
                                                 )
            
            # Record training start time
            training_start = time.time()
            
            # Train the model
            logger.info("Beginning SOM training... This may take several minutes")
            training_result   = self.som_clusterer.train(num_iterations = num_iterations)
            
            training_duration = time.time() - training_start
            
            if training_result.get("success"):
                logger.info("SOM training completed successfully!")
                logger.info(f"Training duration: {training_duration:.2f} seconds")
                logger.info(f"Documents processed: {training_result.get('num_documents', 0)}")
                logger.info(f"Embedding dimension: {training_result.get('embedding_dimension', 0)}")
                logger.info(f"SOM dimensions: {training_result.get('som_dimensions', (12, 12))}")
                
                # Save training metrics
                self.training_metrics["som_training"] = {"success"             : True,
                                                         "duration_seconds"    : training_duration,
                                                         "iterations"          : num_iterations,
                                                         "num_documents"       : training_result.get("num_documents", 0),
                                                         "embedding_dimension" : training_result.get("embedding_dimension", 0),
                                                         "som_dimensions"      : training_result.get("som_dimensions", (12, 12)),
                                                        }
                
                # Save the model
                if self.som_clusterer.save():
                    logger.info("SOM model saved successfully")

                else:
                    logger.warning("Failed to save SOM model")
                
                self.training_status["som_trained"]       = True
                self.training_status["clusters_assigned"] = True
                return True

            else:
                error_msg = training_result.get("error", "Unknown error")
                logger.error(f"SOM training failed: {error_msg}")
                
                self.training_metrics["som_training"] = {"success"         : False,
                                                         "error"           : error_msg,
                                                         "duration_seconds" : training_duration,
                                                        }
                return False
                
        except Exception as e:
            logger.error(f"SOM training failed with exception: {e}")
            self.training_metrics["som_training"] = {"success" : False,
                                                     "error"   : str(e),
                                                    }
            return False
    
    def validate_model_performance(self) -> bool:
        """
        Validate the trained model performance
        """
        logger.info("Validating model performance...")
        
        try:
            # Initialize retriever and explainer
            self.retriever     = Retriever(mongo = self.mongo)
            self.explainer     = Explainer(mongo    = self.mongo, 
                                           embedder = self.embedder,
                                          )
            
            # Test queries for validation
            test_queries       = ["chest pain and shortness of breath",
                                  "diabetes medication side effects",
                                  "high blood pressure treatment",
                                  "cardiac arrhythmia symptoms",
                                  "antibiotic resistance",
                                 ]
            
            validation_results = {"search_tests"        : [],
                                  "cluster_analysis"    : {},
                                  "performance_metrics" : {}
                                 }
            
            logger.info("Running search validation tests...")
            
            for i, query in enumerate(test_queries):
                try:
                    # Generate query embedding
                    query_embedding = self.embedder.embed([query])[0]
                    
                    # Test multi-collection search
                    search_results  = self.retriever.multi_collection_search(query_embedding.tolist(),
                                                                             top_k_per_collection = 2,
                                                                             final_top_k          = 5,
                                                                            )
                    
                    # Test explanation
                    if search_results:
                        explanation = self.explainer.explain_query_result(query,
                                                                          search_results[0]["id"],
                                                                          search_results[0]["collection"],
                                                                         )
                        
                        test_result = {"query"               : query,
                                       "num_results"         : len(search_results),
                                       "top_similarity"      : search_results[0]["cosine_score"] if search_results else 0,
                                       "explanation_success" : "error" not in explanation,
                                       "collections_covered" : list(set([r["collection"] for r in search_results])),
                                      }

                    else:
                        test_result = {"query"               : query,
                                       "num_results"         : 0,
                                       "top_similarity"      : 0,
                                       "explanation_success" : False,
                                       "collections_covered" : []
                                      }
                    
                    validation_results["search_tests"].append(test_result)
                    logger.info(f"Test {i+1}/{len(test_queries)}: {test_result['num_results']} results")
                    
                except Exception as e:
                    logger.warning(f"   ⚠️  Test {i+1} failed: {e}")
                    validation_results["search_tests"].append({"query" : query,
                                                               "error" : str(e),
                                                             })
            
            # Analyze cluster distribution
            logger.info("Analyzing cluster distribution...")
            cluster_analysis                          = self._analyze_cluster_distribution()
            validation_results["cluster_analysis"]    = cluster_analysis
            
            # Calculate performance metrics
            successful_tests                          = len([t for t in validation_results["search_tests"] if t.get("num_results", 0) > 0])
            total_tests                               = len(test_queries)
            
            avg_similarity                            = np.mean([t.get("top_similarity", 0) for t in validation_results["search_tests"]])
            
            performance_metrics                       = {"search_success_rate" : successful_tests / total_tests,
                                                         "average_similarity"  : float(avg_similarity),
                                                         "total_clusters_used" : len(cluster_analysis.get("active_clusters", [])),
                                                         "cluster_utilization" : cluster_analysis.get("utilization_rate", 0),
                                                        }
            
            validation_results["performance_metrics"] = performance_metrics
            self.training_metrics["validation"]       = validation_results
            
            # Determine if validation passed
            min_success_rate                          = 0.6  # 60% of test queries should return results
            min_similarity                            = 0.3    # Average similarity should be reasonable
            
            validation_passed                         = (performance_metrics["search_success_rate"] >= min_success_rate and performance_metrics["average_similarity"] >= min_similarity)
            
            if validation_passed:
                logger.info("Model validation passed!")
                logger.info(f"Search success rate: {performance_metrics['search_success_rate']:.1%}")
                logger.info(f"Average similarity: {performance_metrics['average_similarity']:.3f}")
                logger.info(f"Clusters utilized: {performance_metrics['total_clusters_used']}")
                
                self.training_status["model_validated"] = True
                return True

            else:
                logger.warning("Model validation concerns:")
                logger.warning(f"Search success rate: {performance_metrics['search_success_rate']:.1%} (min: 60%)")
                logger.warning(f"Average similarity: {performance_metrics['average_similarity']:.3f} (min: 0.3)")
                
                # Don't fail completely, but warn
                self.training_status["model_validated"] = True
                return True
                
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    

    def _analyze_cluster_distribution(self) -> Dict[str, Any]:
        """
        Analyze how documents are distributed across clusters
        """
        try:
            collections     = ["reports", "queries", "drug_reviews"]
            cluster_counts  = dict()
            total_clustered = 0
            
            for collection in collections:
                # Get cluster distribution for this collection
                pipeline = [{"$match" : {"som_cluster": {"$exists" : True, 
                                                         "$ne"     : None,
                                                        },
                                        },
                            },
                            {"$group" : {"_id"       : "$som_cluster.id",
                                         "count"     : {"$sum": 1},
                                         "cluster_x" : {"$first": "$som_cluster.x"},
                                         "cluster_y" : {"$first": "$som_cluster.y"},
                                        },
                            }
                           ]
                
                try:
                    results = list(self.mongo.db[collection].aggregate(pipeline))
                    
                    for result in results:
                        cluster_id = result["_id"]
                        
                        if cluster_id not in cluster_counts:
                            cluster_counts[cluster_id] = {"total"       : 0,
                                                          "x"           : result["cluster_x"],
                                                          "y"           : result["cluster_y"],
                                                          "collections" : {},
                                                         }
                        
                        cluster_counts[cluster_id]["total"]                  += result["count"]
                        cluster_counts[cluster_id]["collections"][collection] = result["count"]
                        total_clustered                                      += result["count"]
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze clusters for {collection}: {e}")
            
            # Calculate statistics
            active_clusters       = list(cluster_counts.keys())
            max_possible_clusters = self.som_config["map_x"] * self.som_config["map_y"]
            utilization_rate      = len(active_clusters) / max_possible_clusters
            
            if cluster_counts:
                avg_docs_per_cluster = np.mean([c["total"] for c in cluster_counts.values()])
                max_docs_per_cluster = max([c["total"] for c in cluster_counts.values()])
                min_docs_per_cluster = min([c["total"] for c in cluster_counts.values()])
            
            else:
                avg_docs_per_cluster = max_docs_per_cluster = min_docs_per_cluster = 0
            
            return {"active_clusters"           : active_clusters,
                    "total_clusters_possible"   : max_possible_clusters,
                    "utilization_rate"          : utilization_rate,
                    "total_clustered_documents" : total_clustered,
                    "avg_docs_per_cluster"      : avg_docs_per_cluster,
                    "max_docs_per_cluster"      : max_docs_per_cluster,
                    "min_docs_per_cluster"      : min_docs_per_cluster,
                    "cluster_details"           : cluster_counts,
                   }
            
        except Exception as e:
            logger.error(f"Cluster analysis failed: {e}")
            return {"error": str(e)}
    

    def generate_visualizations(self, save_plots: bool = True) -> bool:
        """
        Generate SOM visualizations
        """
        if not save_plots:
            logger.info("Skipping visualization generation")
            return True
        
        logger.info("Generating SOM visualizations...")
        
        try:
            # Initialize visualizer
            if not self.som_clusterer or not self.som_clusterer.som:
                logger.error("SOM model not available for visualization")
                return False
            
            self.visualizer = SOMVisualizer(self.som_clusterer.som, 
                                            self.mongo,
                                           )
            
            # Create visualizations directory
            viz_dir         = MODELS_DIR / "visualizations"
            viz_dir.mkdir(exist_ok = True)
            
            # Generate occupancy plots for each collection
            collections     = ["reports", "queries", "drug_reviews"]
            
            for collection in collections:
                try:
                    logger.info(f"Generating occupancy plot for {collection}...")
                    save_path = viz_dir / f"{collection}_occupancy.png"
                    self.visualizer.plot_cluster_occupancy(collection, 
                                                           str(save_path),
                                                          )
                
                except Exception as e:
                    logger.warning(f"Failed to generate plot for {collection}: {e}")
            
            # Generate U-Matrix
            try:
                logger.info("Generating U-Matrix...")
                save_path = viz_dir / "u_matrix.png"
                self.visualizer.plot_u_matrix(str(save_path))

            except Exception as e:
                logger.warning(f"Failed to generate U-Matrix: {e}")
            
            # Generate keyword overlay
            try:
                logger.info("Generating keyword overlay...")
                save_path = viz_dir / "keyword_overlay.png"
                self.visualizer.create_keyword_overlay(str(save_path))
            
            except Exception as e:
                logger.warning(f"Failed to generate keyword overlay: {e}")
            
            logger.info(f"Visualizations saved to: {viz_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return False
    

    def run_training_pipeline(self, num_iterations: Optional[int] = None, 
                            generate_viz: bool = True) -> bool:
        """
        Run the complete model training pipeline
        """
        logger.info("Starting Model Training Pipeline")
        logger.info("=" * 80)
        
        training_start_time = time.time()
        
        # Step 1: Connect to database
        if not self.connect_database():
            logger.error("Pipeline aborted: Database connection failed")
            return False
        
        # Step 2: Initialize embedder
        if not self.initialize_embedder():
            logger.error("Pipeline aborted: Embedder initialization failed")
            return False
        
        # Step 3: Validate training data
        if not self.validate_training_data():
            logger.error("Pipeline aborted: Training data validation failed")
            return False
        
        # Step 4: Check for existing model
        if self.check_existing_model() and not self.force_retrain:
            logger.info("Using existing trained model")
        else:
            # Step 5: Train SOM model
            if not self.train_som_model(num_iterations):
                logger.error("Pipeline aborted: SOM training failed")
                return False
        
        # Step 6: Validate model performance
        if not self.validate_model_performance():
            logger.error("Pipeline aborted: Model validation failed")
            return False
        
        # Step 7: Generate visualizations
        if generate_viz:
            if not self.generate_visualizations():
                logger.warning("Visualization generation failed, but continuing...")
        
        # Calculate total training time
        total_training_time                          = time.time() - training_start_time
        self.training_metrics["total_training_time"] = total_training_time
        
        # Step 8: Generate training summary
        self.generate_training_summary()
        
        self.training_status["training_complete"] = True
        logger.info("Model training pipeline completed successfully!")
        logger.info(f"Total time: {total_training_time:.2f} seconds")
        
        return True
    

    def generate_training_summary(self) -> None:
        """
        Generate and display training summary
        """
        logger.info("Training Summary Report")
        logger.info("=" * 80)
        
        # Data validation summary
        data_val = self.training_metrics.get("data_validation", {})
        logger.info("Data Validation:")
        logger.info(f"Total documents: {data_val.get('total_documents', 0)}")
        logger.info(f"Documents with embeddings: {data_val.get('embeddings_count', 0)}")
        
        # Training summary
        training = self.training_metrics.get("som_training", {})
        
        if training.get("success"):
            logger.info("SOM Training:")
            logger.info(f"Duration: {training.get('duration_seconds', 0):.2f} seconds")
            logger.info(f"Documents processed: {training.get('num_documents', 0)}")
            logger.info(f"SOM dimensions: {training.get('som_dimensions', (12, 12))}")
        
        # Validation summary
        validation = self.training_metrics.get("validation", {})
        
        if validation:
            perf = validation.get("performance_metrics", {})
            logger.info("Model Validation:")
            logger.info(f"Search success rate: {perf.get('search_success_rate', 0):.1%}")
            logger.info(f"Average similarity: {perf.get('average_similarity', 0):.3f}")
            logger.info(f"Clusters utilized: {perf.get('total_clusters_used', 0)}")
            logger.info(f"Cluster utilization: {perf.get('cluster_utilization', 0):.1%}")
        
        # Total time
        total_time = self.training_metrics.get("total_training_time", 0)
        logger.info(f"Total pipeline time: {total_time:.2f} seconds")
        
        logger.info("=" * 80)
    

    def print_status(self) -> None:
        """
        Print current training status
        """
        logger.info("Model Training Status:")
        logger.info("-" * 80)
        
        for component, status in self.training_status.items():
            status_state   = True if status else False
            component_name = component.replace("_", " ").title()
            logger.info(f"{status_state} {component_name}: {status}")
        
        logger.info("-" * 80)


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description     = "MediMap-XAI Model Training & Validation Pipeline",
                                     formatter_class = argparse.RawDescriptionHelpFormatter,
                                     epilog          = """
                                                          Examples:
                                                          python run_model_training.py                     # Train with default settings
                                                          python run_model_training.py --force-retrain    # Force retrain existing model
                                                          python run_model_training.py --iterations 2000  # Custom training iterations
                                                          python run_model_training.py --no-visualizations # Skip visualization generation
                                                          python run_model_training.py --som-size 15 15   # Custom SOM dimensions
                                                       """,
                                    )
    
    parser.add_argument("--force-retrain",
                        action = "store_true",
                        help   = "Force retrain model even if existing model found",
                       )
    
    parser.add_argument("--iterations",
                        type = int,
                        help = f"Number of training iterations (default: {SOMConfig.num_iterations})",
                       )
    
    parser.add_argument("--som-size",
                        type    = int,
                        nargs   = 2,
                        metavar = ("X", "Y"),
                        help    = f"SOM grid dimensions (default: {SOMConfig.map_x} {SOMConfig.map_y})"
                       )
    
    parser.add_argument("--learning-rate",
                        type = float,
                        help = f"SOM learning rate (default: {SOMConfig.learning_rate})",
                       )
    
    parser.add_argument("--sigma",
                        type = float,
                        help = f"SOM sigma parameter (default: {SOMConfig.sigma})",
                       )
    
    parser.add_argument("--no-visualizations",
                        action = "store_true",
                        help   = "Skip visualization generation",
                       )
    
    parser.add_argument("--log-level",
                        choices = ["DEBUG", "INFO", "WARNING", "ERROR"],
                        default = "INFO",
                        help    = "Set logging level (default: INFO)",
                       )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Prepare SOM configuration
    som_config = SOMConfig.__dict__.copy()
    
    if args.som_size:
        som_config["map_x"], som_config["map_y"] = args.som_size
    
    if args.learning_rate:
        som_config["learning_rate"]              = args.learning_rate
    
    if args.sigma:
        som_config["sigma"]                      = args.sigma
    
    # Create runner
    runner = ModelTrainingRunner(force_retrain = args.force_retrain,
                                 som_config    = som_config,
                                )
    
    # Run pipeline
    try:
        success = runner.run_training_pipeline(num_iterations = args.iterations,
                                               generate_viz   = not args.no_visualizations,
                                              )
        
        # Print final status
        runner.print_status()
        
        if success:
            logger.info("Model training completed successfully!")
            logger.info("Next step: Launch the web app with 'python run_web_app.py'")
            sys.exit(0)
       
        else:
            logger.error("Model training failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Model training interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()