# DEPENDENCIES
import os
from typing import Dict
from pathlib import Path
from dataclasses import field
from dataclasses import dataclass

# PATHS
BASE_DIR             = Path(__file__).resolve().parent.parent
RAW_DATA_DIR         = BASE_DIR / "data" / "raw_data"
MODELS_DIR           = BASE_DIR / "models"
EMBEDDINGS_DIR       = BASE_DIR / "embeddings"

# CREATE REQUIRED DIRECTORIES IF THEY DON'T EXIST
def ensure_dirs():
    """
    Create required directories if they don't exist
    """
    directory_list = [RAW_DATA_DIR, MODELS_DIR, EMBEDDINGS_DIR]
    for directory in directory_list:
        directory.mkdir(parents  = True,
                        exist_ok = True,
                       )

# DATABASE CONFIGURATIONS
MONGO_URI            = "mongodb://localhost:27017"

DB_NAME              = "medimap_xai"

# EMBEDDINGS & SOM DEFAULTS
EMBEDDING_MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

SOM_MODEL_PATH       = MODELS_DIR / "som_model.pkl"

# DATASET FILENAMES WITH FALLBACKS
REPORTS_CSV          = RAW_DATA_DIR / "medical_reports.csv"
QA_CSV               = RAW_DATA_DIR / "medical_qa.csv"
DRUG_REVIEWS_CSV     = RAW_DATA_DIR / "drug_reviews.csv"


# CONFIG STRUCTURES
@dataclass(frozen = True)
class SOMConfig:
    map_x          : int   = 12
    map_y          : int   = 12
    sigma          : float = 0.5
    learning_rate  : float = 0.001
    random_seed    : int   = 42
    num_iterations : int   = 1000
    model_path     : Path  = field(default_factory = lambda: SOM_MODEL_PATH)


@dataclass(frozen = True)
class SearchConfig:
    top_k_default      : int   = 5
    similarity_weight  : float = 0.7
    cluster_weight     : float = 0.2
    max_snippet_length : int   = 500
    max_context_length : int   = 2500


# CREATING INSTANCES FOR IMPORT
SOM_CONFIGURATION    = SOMConfig()
SEARCH_CONFIGURATION = SearchConfig()

