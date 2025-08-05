# DEPENDENCIES
import re
import logging
import hashlib
import pandas as pd
from typing import Any
from typing import List
from typing import Dict
from typing import Optional

# INITIALIZE LOGGING
logger = logging.getLogger(__name__)

def clean_text(text: Optional[str], max_length: int = 50000) -> str:
    """
    Text cleaning with medical domain specifics

    - Handles None, int, float
    - Removes script injections
    - Normalizes medical abbreviations
    - Collapses whitespace
    - Trucates if too long

    Arguments:
    ----------
        text      { Optional[str] } : Input text to clean

        max_length     { int }      : Maximum allowed length of the text

    Returns:
    --------
                   { str }          : Cleaned and normalized text, truncated if necessary
    """
    if ((not text) or (not isinstance(text, (str, int, float)))):
        return ""
    
    try:
        # Convert to string if not already
        text                 = str(text)
        
        # Remove HTML tags and scripts
        text                 = re.sub(pattern = r'<[^>]+>', 
                                      repl    = '', 
                                      string  = text,
                                     )

        text                  = re.sub(pattern  = r'<script.*?</script>', 
                                       repl     = '', 
                                       string   = text, 
                                       flags    = re.DOTALL | re.IGNORECASE,
                                      )

        text                  = re.sub(pattern = r'javascript:', 
                                       repl    = '', 
                                       string  = text, 
                                       flags   = re.IGNORECASE,
                                      )
        
        # Enhanced medical abbreviations
        medical_abbreviations = {r'\bpt\.?\b'     : 'patient',
                                 r'\bdx\b'        : 'diagnosis',
                                 r'\btx\b'        : 'treatment',
                                 r'\bhx\b'        : 'history',
                                 r'\bc/o\b'       : 'complains of',
                                 r'\bs/p\b'       : 'status post',
                                 r'\bw/\b'        : 'with',
                                 r'\bw/o\b'       : 'without',
                                 r'\br/o\b'       : 'rule out',
                                 r'\bp\.o\.\b'    : 'by mouth',
                                 r'\bb\.i\.d\.\b' : 'twice daily',
                                 r'\bt\.i\.d\.\b' : 'three times daily',
                                 r'\bq\.d\.\b'    : 'once daily',
                                 r'\bprn\b'       : 'as needed',
                                }
        
        for abbreviation, full_form in medical_abbreviations.items():
            text = re.sub(pattern = abbreviation, 
                          repl    = full_form, 
                          string  = text, 
                          flags   = re.IGNORECASE,
                         )
        
        # Clean up whitespace and special characters
        text                  = re.sub(pattern = r'\s+', 
                                       repl    = ' ', 
                                       string  = text,
                                      )

        text                  = re.sub(pattern = r'[^\w\s\-\.,;:()/?!]', 
                                       repl    = '', 
                                       string  = text,
                                      )

        text                  = text.strip()
        
        # Truncate if too long
        if (len(text) > max_length):
            text = text[:max_length] + "..."
            logger.warning(f"Text truncated to {max_length} characters")
        
        return text
        
    except Exception as e:
        logger.error(f"Text cleaning failed: {e}")
        return ""


def generate_document_id(text: str, source: str, index: int) -> str:
    """
    Generate consistent document IDs

    Arguments:
    ----------
        text   { str } : Text content to hash

        source { str } : Source identifier (e.g., 'reports', 'qa', 'reviews')

        index  { int } : Index of the document in the source

    Returns:
    --------
            { str }    : Generated document ID in the format "source_index_hash"
    """
    text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{source}_{index}_{text_hash}"


def validate_csv_columns(df: pd.DataFrame, required_columns: List[str], csv_name: str) -> bool:
    """
    Validate CSV has required columns
    
    Arguments:
    ----------
        df             { pd.DataFrame } : DataFrame to validate 

        required_columns { List[str] }  : List of required column names

        csv_name            { str }     : Name of the CSV file for logging

    Returns:
    --------
                  { bool }              : True if all requiered columns are present, False otherwise  
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"{csv_name} missing required columns: {missing_columns}")
        logger.info(f"Available columns: {list(df.columns)}")
        
        return False
    
    return True


def safe_get_column_value(row: pd.Series, possible_columns: List[str], default: str = "") -> str:
    """
    Safely get value from multiple possible column names
    
    Arguments:
    ----------
        row               { pd.Series } : Row of the DataFrame to check
        
        possible_columns  { List[str] } : List of possible column names to check

        default              { str }    : Default value to return if no column has a valid value

    Returns:
    --------
                  { str }               : Value from the first available column, or default if none found  
    """
    for column in possible_columns:
        if column in row.index and pd.notna(row[column]):
            
            return str(row[column])
    
    return default
