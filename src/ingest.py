# DEPENDENCIES
import logging
import pandas as pd
from tqdm import tqdm
from typing import Any
from typing import List
from typing import Dict
from .config import QA_CSV
from typing import Optional
from .db import MongoHandler
from .utils import clean_text
from .embedder import Embedder
from .config import REPORTS_CSV
from .config import DRUG_REVIEWS_CSV
from .utils import generate_document_id
from .utils import validate_csv_columns
from .utils import safe_get_column_value

# INITIALIZE LOGGING
logger = logging.getLogger(__name__)


class Ingestor:
    """
    Enhanced ingestion pipeline with better error handling and validation
    """
    def __init__(self, mongo: MongoHandler, embedder: Embedder) -> None:
        self.mongo      = mongo
        self.embedder   = embedder
        self.batch_size = 100
    

    def ingest_reports(self, csv_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Ingest medical reports with comprehensive error handling
        """
        data_path = csv_path or REPORTS_CSV
        
        if not data_path.exists():
            logger.error(f"Reports CSV not found: {data_path}")
            return {"success" : False, 
                    "error"   : f"File not found: {data_path}",
                   }
        
        try:
            dataframe         = pd.read_csv(filepath_or_buffer = data_path)

            logger.info(f"Loaded {len(dataframe)} reports from {data_path}")
            
            # Try different column name variations
            text_columns      = ['transcription', 'transcript', 'text', 'description', 'medical_specialty']
            specialty_columns = ['medical_specialty', 'specialty', 'category']
            
            processed         = 0
            failed            = 0
            
            for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Ingesting reports"):
                try:
                    # Get text content
                    raw_text = safe_get_column_value(row, text_columns)
                    if not raw_text:
                        failed += 1
                        continue
                    
                    cleaned_text = clean_text(raw_text)
                    if not cleaned_text:
                        failed += 1
                        continue
                    
                    # Generate embedding
                    embedding = self.embedder.embed([cleaned_text])[0].tolist()
                    
                    # Get additional metadata
                    specialty = safe_get_column_value(row, specialty_columns)
                    
                    document  = {"_id"               : generate_document_id(cleaned_text, "report", idx),
                                 "source"            : "medical_reports",
                                 "raw_text"          : raw_text,
                                 "clean_text"        : cleaned_text,
                                 "embedding"         : embedding,
                                 "medical_specialty" : specialty,
                                 "metadata"          : {"original_index"      : idx,
                                                        "text_length"         : len(cleaned_text),
                                                        "embedding_dimension" : len(embedding),
                                                       },
                                }
                    
                    if self.mongo.upsert("reports", document):
                        processed += 1
                    else:
                        failed += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to process report {idx}: {e}")
                    failed += 1
            
            result = {"success"   : True,
                      "processed" : processed,
                      "failed"    : failed,
                      "total"     : len(dataframe),
                     }
            logger.info(f"Reports ingestion complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Reports ingestion failed: {e}")
            return {"success": False, "error": str(e)}
    

    def ingest_qa(self, csv_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Ingest Q&A pairs with enhanced processing
        """
        data_path = csv_path or QA_CSV
        
        if not data_path.exists():
            logger.error(f"QA CSV not found: {data_path}")
            return {"success": False, "error": f"File not found: {data_path}"}
        
        try:
            dataframe = pd.read_csv(data_path)
            logger.info(f"Loaded {len(dataframe)} Q&A pairs from {data_path}")
            
            question_columns = ['question', 'Question', 'qtext', 'query']
            answer_columns   = ['answer', 'Answer', 'atext', 'response']
            
            processed        = 0
            failed           = 0
            
            for idx, row in tqdm(dataframe.iterrows(), total = len(dataframe), desc = "Ingesting Q&A"):
                try:
                    question = safe_get_column_value(row, question_columns)
                    answer   = safe_get_column_value(row, answer_columns)
                    
                    if not question and not answer:
                        failed += 1
                        continue
                    
                    # Combine question and answer
                    combined_text = f"Q: {question} A: {answer}" if question and answer else question or answer
                    cleaned_text  = clean_text(combined_text)
                    
                    if not cleaned_text:
                        failed += 1
                        continue
                    
                    embedding = self.embedder.embed([cleaned_text])[0].tolist()
                    
                    document  = {"_id"        : generate_document_id(cleaned_text, "query", idx),
                                 "source"     : "medical_qa",
                                 "question"   : question,
                                 "answer"     : answer,
                                 "clean_text" : cleaned_text,
                                 "embedding"  : embedding,
                                 "metadata"   : {"original_index" : idx,
                                                 "has_question"   : bool(question),
                                                 "has_answer"     : bool(answer),
                                                 "text_length"    : len(cleaned_text),
                                                },
                                }
                    
                    if self.mongo.upsert("queries", document):
                        processed += 1

                    else:
                        failed += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to process Q&A {idx}: {e}")
                    failed += 1
            
            result = {"success"   : True,
                      "processed" : processed,
                      "failed"    : failed,
                      "total"     : len(dataframe),
                     }
            logger.info(f"Q&A ingestion complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Q&A ingestion failed: {e}")
            return {"success": False, "error": str(e)}
    

    def ingest_drug_reviews(self, csv_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Ingest drug reviews with metadata extraction
        """
        data_path = csv_path or DRUG_REVIEWS_CSV
        
        if not data_path.exists():
            logger.error(f"Drug reviews CSV not found: {data_path}")
            return {"success" : False, 
                    "error"   : f"File not found: {data_path}",
                   }
        
        try:
            dataframe         = pd.read_csv(filepath_or_buffer = data_path)
            logger.info(f"Loaded {len(dataframe)} drug reviews from {data_path}")
            
            review_columns    = ['review', 'Review', 'reviewText', 'comment']
            drug_columns      = ['drugName', 'drug_name', 'drug', 'medication']
            condition_columns = ['condition', 'Condition', 'indication']
            rating_columns    = ['rating', 'Rating', 'overall_rating', 'effectiveness']
            
            processed         = 0
            failed            = 0
            
            for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Ingesting drug reviews"):
                try:
                    review_text = safe_get_column_value(row, review_columns)
                    if not review_text:
                        failed += 1
                        continue
                    
                    cleaned_text = clean_text(review_text)
                    if not cleaned_text:
                        failed += 1
                        continue
                    
                    embedding = self.embedder.embed([cleaned_text])[0].tolist()
                    
                    # Extract metadata
                    drug_name = safe_get_column_value(row, drug_columns)
                    condition = safe_get_column_value(row, condition_columns)
                    rating    = safe_get_column_value(row, rating_columns)
                    
                    document  = {"_id"         : generate_document_id(cleaned_text, "review", idx),
                                 "source"      : "drug_reviews",
                                 "drug_name"   : drug_name,
                                 "condition"   : condition,
                                 "rating"      : rating,
                                 "review_text" : review_text,
                                 "clean_text"  : cleaned_text,
                                 "embedding"   : embedding,
                                 "metadata"    : {"original_index" : idx,
                                                  "text_length"    : len(cleaned_text),
                                                  "has_rating"     : bool(rating),
                                                 }
                                }
                    
                    if self.mongo.upsert("drug_reviews", document):
                        processed += 1
                    
                    else:
                        failed += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to process drug review {idx}: {repr(e)}")
                    failed += 1
            
            result = {"success"   : True,
                      "processed" : processed,
                      "failed"    : failed,
                      "total"     : len(dataframe),
                     }
            logger.info(f"Drug reviews ingestion complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Drug reviews ingestion failed: {e}")
            return {"success": False, "error": str(e)}
    

    def ingest_all(self) -> Dict[str, Any]:
        """
        Ingest all datasets
        """
        results                 = dict()
        
        logger.info("Starting complete ingestion pipeline...")
        
        results["reports"]      = self.ingest_reports()
        results["qa"]           = self.ingest_qa()
        results["drug_reviews"] = self.ingest_drug_reviews()
        
        # Summary
        total_processed         = sum([r.get("processed", 0) for r in results.values() if isinstance(r, dict)])
        total_failed            = sum([r.get("failed", 0) for r in results.values() if isinstance(r, dict)])
        
        results["summary"]      = {"total_processed" : total_processed,
                                   "total_failed"    : total_failed,
                                   "success_rate"    : total_processed / (total_processed + total_failed) if (total_processed + total_failed) > 0 else 0,
                                  }
                                
        logger.info(f"Ingestion complete. Summary: {results['summary']}")

        return results

