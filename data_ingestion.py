# DEPENDENCIES
import os
import sys
import logging
import argparse
from typing import Any
from typing import List
from typing import Dict
from pathlib import Path


# Add scripts to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import QA_CSV 
from src.config import DB_NAME
from src.db import MongoHandler
from src.ingest import Ingestor
from src.config import BASE_DIR
from src.config import MONGO_URI
from src.utils import clean_text
from src.embedder import Embedder
from src.config import MODELS_DIR
from src.config import REPORTS_CSV
from src.config import RAW_DATA_DIR
from src.config import DRUG_REVIEWS_CSV


# Configure logging
logging.basicConfig(level    = logging.INFO,
                    format   = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers = [logging.FileHandler('logs/data_ingestion.log'),
                                logging.StreamHandler(sys.stdout),
                               ]
                   )

logger = logging.getLogger(__name__)


class DataIngestionRunner:
    """
    Handles database setup and data ingestion
    """
    def __init__(self, force_reingest: bool = False, collections: List[str] = None):
        self.force_reingest     = force_reingest
        self.target_collections = collections or ["reports", "queries", "drug_reviews"]
        self.mongo              = None
        self.embedder           = None
        self.ingestor           = None
        
        # Track ingestion status
        self.ingestion_status   = {"database_connected"    : False,
                                   "embedder_ready"        : False,
                                   "reports_ingested"      : False,
                                   "queries_ingested"      : False,
                                   "drug_reviews_ingested" : False,
                                   "ingestion_complete"    : False,
                                  }
    

    def setup_directories(self) -> bool:
        """
        Create necessary directories
        """
        logger.info("Setting up directory structure...")
        
        try:
            directories = [BASE_DIR, 
                           RAW_DATA_DIR, 
                           MODELS_DIR,
                          ]

            for directory in directories:
                directory.mkdir(parents  = True, 
                                exist_ok = True,
                               )

                logger.info(f"Directory ready: {directory}")
            
            logger.info("Directory structure setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup directories: {e}")
            return False
    

    def connect_database(self) -> bool:
        """
        Initialize MongoDB connection
        """
        logger.info("Connecting to MongoDB...")
        logger.info(f"URI: {MONGO_URI}")
        logger.info(f"Database: {DB_NAME}")
        
        try:
            self.mongo  = MongoHandler()
            
            # Test connection
            collections = self.mongo.get_collection_names()
            logger.info(f"Connected successfully. Existing collections: {collections}")
            
            self.ingestion_status["database_connected"] = True
            return True
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            logger.error("Please ensure MongoDB is running and accessible")
            return False
    

    def initialize_embedder(self) -> bool:
        """
        Initialize the embedding model
        """
        logger.info("Loading embedding model...")
        
        try:
            self.embedder  = Embedder()
            
            # Test embedding generation
            test_embedding = self.embedder.embed(["test sentence"])
            logger.info(f"Embedder ready. Dimension: {test_embedding.shape[1]}")
            
            self.ingestion_status["embedder_ready"] = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            return False
    

    def check_data_files(self) -> Dict[str, Dict[str, Any]]:
        """
        Check availability and basic info of data files
        """
        logger.info("Checking data file availability...")
        
        data_files = {"reports"      : {"path"       : REPORTS_CSV,
                                        "exists"     : REPORTS_CSV.exists(),
                                        "size"       : REPORTS_CSV.stat().st_size if REPORTS_CSV.exists() else 0,
                                        "collection" : "reports"
                                       },
                      "qa"           : {"path"       : QA_CSV,
                                        "exists"     : QA_CSV.exists(),
                                        "size"       : QA_CSV.stat().st_size if QA_CSV.exists() else 0,
                                        "collection" : "queries"
                                       },
                      "drug_reviews" : {"path"       : DRUG_REVIEWS_CSV,
                                        "exists"     : DRUG_REVIEWS_CSV.exists(),
                                        "size"       : DRUG_REVIEWS_CSV.stat().st_size if DRUG_REVIEWS_CSV.exists() else 0,
                                        "collection" : "drug_reviews"
                                       }
                     }
        
        logger.info("Data file status:")
        total_files     = 0
        available_files = 0
        
        for name, info in data_files.items():
            total_files += 1
            status       = True if info["exists"] else False
            size_mb      = info["size"] / (1024 * 1024) if info["size"] > 0 else 0
            
            logger.info(f"{status} {name}: {info['exists']} ({size_mb:.2f} MB)")
            
            if info["exists"]:
                available_files += 1
        
        logger.info(f"Summary: {available_files}/{total_files} data files available")
        return data_files
    

    def check_existing_data(self) -> Dict[str, int]:
        """
        Check existing data in database collections
        """
        logger.info("Checking existing data in database...")
        
        existing_counts = dict()
        total_documents = 0
        
        for collection in ["reports", "queries", "drug_reviews"]:
            try:
                count                       = self.mongo.count_documents(collection)
                existing_counts[collection] = count
                total_documents            += count
                
                logger.info(f"{collection}: {count} documents")
                
            except Exception as e:
                logger.warning(f"Failed to count {collection}: {e}")
                existing_counts[collection] = 0
        
        logger.info(f"Total existing documents: {total_documents}")
        return existing_counts
    

    def ingest_reports(self) -> Dict[str, Any]:
        """
        Ingest medical reports
        """
        if "reports" not in self.target_collections:
            logger.info("Skipping reports ingestion (not in target collections)")
            return {"success": True, "skipped": True}
        
        logger.info("Starting medical reports ingestion...")
        
        try:
            result = self.ingestor.ingest_reports()
            
            if result.get("success"):
                processed = result.get("processed", 0)
                failed    = result.get("failed", 0)
                total     = result.get("total", 0)
                
                logger.info(f"Reports ingestion completed:")
                logger.info(f"Processed: {processed}/{total}")
                logger.info(f"Failed: {failed}")
                logger.info(f"Success rate: {(processed/total*100):.1f}%" if total > 0 else "  Success rate: 0%")
                
                self.ingestion_status["reports_ingested"] = True
            else:
                logger.error(f"Reports ingestion failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Reports ingestion failed with exception: {e}")
            return {"success": False, "error": str(e)}
    

    def ingest_qa(self) -> Dict[str, Any]:
        """
        Ingest Q&A pairs
        """
        if "queries" not in self.target_collections:
            logger.info(f"Skipping Q&A ingestion (not in target collections)")
            return {"success": True, "skipped": True}
        
        logger.info("Starting Q&A pairs ingestion...")
        
        try:
            result = self.ingestor.ingest_qa()
            
            if result.get("success"):
                processed = result.get("processed", 0)
                failed    = result.get("failed", 0)
                total     = result.get("total", 0)
                
                logger.info(f"Q&A ingestion completed:")
                logger.info(f"Processed: {processed}/{total}")
                logger.info(f"Failed: {failed}")
                logger.info(f"Success rate: {(processed/total*100):.1f}%" if total > 0 else " Success rate: 0%")
                
                self.ingestion_status["queries_ingested"] = True
            else:
                logger.error(f"Q&A ingestion failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Q&A ingestion failed with exception: {e}")
            return {"success": False, "error": str(e)}
    

    def ingest_drug_reviews(self) -> Dict[str, Any]:
        """
        Ingest drug reviews
        """
        if "drug_reviews" not in self.target_collections:
            logger.info("Skipping drug reviews ingestion (not in target collections)")
            return {"success": True, "skipped": True}
        
        logger.info("Starting drug reviews ingestion...")
        
        try:
            result = self.ingestor.ingest_drug_reviews()
            
            if result.get("success"):
                processed = result.get("processed", 0)
                failed    = result.get("failed", 0)
                total     = result.get("total", 0)
                
                logger.info(f"Drug reviews ingestion completed:")
                logger.info(f"Processed: {processed}/{total}")
                logger.info(f"Failed: {failed}")
                logger.info(f"Success rate: {(processed/total*100):.1f}%" if total > 0 else "   📈 Success rate: 0%")
                
                self.ingestion_status["drug_reviews_ingested"] = True
            else:
                logger.error(f"Drug reviews ingestion failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Drug reviews ingestion failed with exception: {e}")
            return {"success": False, "error": str(e)}
    

    def run_ingestion_pipeline(self) -> bool:
        """
        Run the complete data ingestion pipeline
        """
        logger.info("Starting Data Ingestion Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Setup directories
        if not self.setup_directories():
            logger.error("Pipeline aborted: Directory setup failed")
            return False
        
        # Step 2: Connect to database
        if not self.connect_database():
            logger.error("Pipeline aborted: Database connection failed")
            return False
        
        # Step 3: Initialize embedder
        if not self.initialize_embedder():
            logger.error("Pipeline aborted: Embedder initialization failed")
            return False
        
        # Step 4: Check data files
        data_files      = self.check_data_files()
        available_files = [name for name, info in data_files.items() if info["exists"]]
        
        if not available_files:
            logger.error("Pipeline aborted: No data files found")
            logger.error(f"Please place data files in: {RAW_DATA_DIR}")
            return False
        
        # Step 5: Check existing data
        existing_data  = self.check_existing_data()
        total_existing = sum(existing_data.values())
        
        if ((total_existing > 0) and (not self.force_reingest)):
            logger.info(f"Found {total_existing} existing documents")
            response = input("Re-ingest data? This will replace existing data (y/N): ").strip().lower()
            
            if (response not in ['y', 'yes']):
                logger.info("Skipping ingestion, using existing data")
                self.ingestion_status["ingestion_complete"] = True
                return True
        
        # Step 6: Initialize ingestor
        logger.info("Initializing data ingestor...")
        self.ingestor                = Ingestor(self.mongo, 
                                                self.embedder
                                               )
        
        # Step 7: Run ingestion for each dataset
        ingestion_results            = dict()
        
        # Ingest reports
        ingestion_results["reports"] = self.ingest_reports()
        
        # Ingest Q&A
        ingestion_results["qa"]      = self.ingest_qa()
        
        # Ingest drug reviews
        ingestion_results["drug_reviews"] = self.ingest_drug_reviews()
        
        # Step 8: Generate summary
        self.generate_ingestion_summary(ingestion_results)
        
        # Check if ingestion was successful
        successful_ingestions = sum(1 for result in ingestion_results.values() 
                                  if result.get("success", False))
        
        if successful_ingestions > 0:
            self.ingestion_status["ingestion_complete"] = True
            logger.info("Data ingestion pipeline completed successfully!")
            return True
        else:
            logger.error("Data ingestion pipeline failed - no successful ingestions")
            return False
    
    def generate_ingestion_summary(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Generate and display ingestion summary
        """
        logger.info("Ingestion Summary Report")
        logger.info("=" * 80)
        
        total_processed = 0
        total_failed    = 0
        total_files     = 0
        
        for dataset, result in results.items():
            if result.get("skipped"):
                logger.info(f"{dataset}: Skipped")
                continue
            
            if result.get("success"):
                processed        = result.get("processed", 0)
                failed           = result.get("failed", 0)
                total            = result.get("total", 0)
                
                total_processed += processed
                total_failed    += failed
                total_files     += total
                
                success_rate = (processed / total * 100) if total > 0 else 0
                logger.info(f"{dataset}: {processed}/{total} ({success_rate:.1f}%)")
            else:
                logger.info(f"{dataset}: Failed - {result.get('error', 'Unknown error')}")
        
        logger.info("-" * 80)
        logger.info(f"Overall Summary:")
        logger.info(f"Total documents processed: {total_processed}")
        logger.info(f"Total failed: {total_failed}")
        logger.info(f"Total files processed: {total_files}")
        
        if (total_files > 0):
            overall_success_rate = (total_processed / total_files) * 100
            logger.info(f"Overall success rate: {overall_success_rate:.1f}%")
        
        logger.info("=" * 80)
        
        # Final database state
        final_counts = self.check_existing_data()
        logger.info("Final database state:")
        for collection, count in final_counts.items():
            logger.info(f"{collection}: {count} documents")
    
    def print_status(self) -> None:
        """
        Print current status
        """
        logger.info("Data Ingestion Status:")
        logger.info("-" * 80)
        
        for component, status in self.ingestion_status.items():
            status_state   = True if status else False
            component_name = component.replace("_", " ").title()
            logger.info(f"{status_state} {component_name}: {status}")
        
        logger.info("-" * 80)


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description     = "MediMap-XAI Data Ingestion Pipeline",
                                     formatter_class = argparse.RawDescriptionHelpFormatter,
                                     epilog          = """Examples:
                                                          python run_data_ingestion.py                    # Ingest all available data
                                                          python run_data_ingestion.py --force-reingest  # Force re-ingestion
                                                          python run_data_ingestion.py --collections reports queries  # Ingest specific collections
                                                          python run_data_ingestion.py --log-level DEBUG # Enable debug logging
                                                       """,
                                    )
    
    parser.add_argument("--force-reingest",
                        action = "store_true",
                        help   = "Force re-ingestion even if data already exists"
                       )
    
    parser.add_argument("--collections",
                        nargs   = "+",
                        choices = ["reports", "queries", "drug_reviews"],
                        help    = "Specific collections to ingest (default: all)",
                       )
    
    parser.add_argument("--log-level",
                        choices = ["DEBUG", "INFO", "WARNING", "ERROR"],
                        default = "INFO",
                        help    = "Set logging level (default: INFO)",
                       )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create runner
    runner = DataIngestionRunner(force_reingest = args.force_reingest,
                                 collections    = args.collections,
                                )
    
    # Run pipeline
    try:
        success = runner.run_ingestion_pipeline()
        
        # Print final status
        runner.print_status()
        
        if success:
            logger.info("Data ingestion completed successfully!")
            logger.info("Next step: Run model training with 'python run_model_training.py'")
            sys.exit(0)

        else:
            logger.error("Data ingestion failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Data ingestion interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
