# DEPENDENCIES
import logging
import streamlit as st
from typing import Optional


# INITALIZE LOGGING
logger = logging.getLogger(__name__)


class FileHandler:
    def extract_text_from_upload(self, uploaded_file) -> Optional[str]:
        """
        Extract text from uploaded file
        """
        try:
            raw = uploaded_file.read()
            
            if isinstance(raw, bytes):
                if (uploaded_file.type == "application/pdf"):
                    st.warning("📄 PDF uploaded; using basic text extraction. Results may vary.")

                    text = raw.decode(encoding = "utf-8", 
                                      errors   = "ignore",
                                     )
                return text

            return str(raw)

        except Exception as e:
            logger.exception("Failed to extract text from upload")
            return None