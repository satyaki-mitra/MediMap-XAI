# DEPENDENCIES
import time
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from .config import DB_NAME
from .config import MONGO_URI
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# INITIALIZE LOGGING
logger = logging.getLogger(__name__)

class MongoHandler:
    """
    MongoDB wrapper with retry logic, validation, and basic helpers
    """
    def __init__(self, uri: str = MONGO_URI, db_name: str = DB_NAME) -> None:
        self.client      = None
        self.db          = None
        self.max_retries = 5
        self.retry_delay = 10

        self._connect(uri     = uri, 
                      db_name = db_name,
                     )
    

    def _connect(self, uri: str, db_name: str) -> None:
        """
        Establish connection with retry logic

        Arguments:
        ----------
            uri     { str } : MongoDB URI to connect to

            db_name { str } : Name of the database to use

        Errors:
        -------
            ConnectionError : If connection fails after max retries

        Returns:
        --------
            { None }        : None, but sets up self.client and self.db attributes
        """
        for attempt in range(self.max_retries):
            try:
                self.client = MongoClient(host                     = uri,                 
                                          serverSelectionTimeoutMS = 5000,
                                          maxPoolSize              = 50,
                                          retryWrites              = True,
                                          w                        = "majority",
                                         )
                # Trigger connection
                self.client.server_info()

                self.db = self.client[db_name]

                logger.info(f"Connected to MongoDB at: {uri} (attempted: {attempt + 1})")
                return

            except Exception as ConnectionError:
                logger.warning(f"Connection attempt {attempt + 1} failed: {repr(ConnectionError)}")

                # Handle specific connection errors
                if (attempt == self.max_retries - 1):
                    raise ConnectionError(f"Failed to connect after {self.max_retries} attempts: {repr(ConnectionError)}")
                
                # Wait before retrying
                time.sleep(self.retry_delay)
    

    def upsert(self, collection: str, document: Dict[str, Any]) -> bool:
        """
        Enhanced upsert with better error handling

        Arguments:
        ----------
            collection       { str }      : Name of the collection to upsert into

            document   { Dict[str, Any] } : Document to upsert, must contain '_id' field

        Errors:
        -------
            ValueError                    : If '_id' field is missing in the document

            PyMongoError                  : If upsert operation fails for any reason

        Returns:
        --------
                        { bool }          : True if upseert was successful, False otherwise
        """
        # Check if _id is present in the document
        if ("_id" not in document):
            raise ValueError("Document must contain '_id' field")
        
        try:
            # Perform upsert operation
            result = self.db[collection].replace_one(filter      = {"_id" : document["_id"]},
                                                     replacement = document,
                                                     upsert      = True,
                                                    )

            logger.debug(f"Upserted document {document['_id']} in {collection}")
            return True

        except PyMongoError as e:
            logger.error(f"Upsert failed for {document.get('_id')} in {collection}: {repr(e)}")
            return False

    
    def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find a single document matching `query`
        
        Arguments:
        ----------

            collection    { str }         : Name of the collection to search in

            query     { Dict[str, Any] }  : Query to find a single document

        Errors:
        -------
            ValueError                    : If query is not a non-empty dictionary

            PyMongoError                  : If find_one operation fails for any reason

        Returns:
        --------
            { Dict[str, Any] | None }     : The found document or None if not found or an error occurred
        """
        # Validate query
        if not (isinstance(query, dict)):
            raise ValueError ("Query must be a dictionary")

        try:
            # Perform find_one operation
            query_response = self.db[collection].find_one(filter     = query,
                                                          projection = None,
                                                         )
            
            return query_response

        except PyMongoError as e:
            logger.error(f"find_one failed on {collection}: {e}")
            return None
    

    def find_many(self, collection: str, query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Find multiple documents matching `query` with optional projection and limit
        
        Arguments:
        ----------
            collection      { str }       : Name of the collection to search in

            query      { Dict[str, Any] } : Query to find documents, defaults to empty dict

            projection { Dict[str, Any] } : Fields to include or exclude, defaults to None

            limit          { int }        : Maximum number of documents to return, defaults to None 

        Errors:
        -------
            ValueError                    : If query is not a dictionary or projection is not a dictionary

            PyMongoError                  : If find_many operation fails for any reason

        Returns:
        --------
                { List[Dict[str, Any]] }  : List of found documents, empty list if none found or an error occurred
        """
        # Validate query
        if ((query is not None) and (not isinstance(query, dict))):
            raise ValueError ("Query must be a dictionary or None")

        # Validate projection
        if ((projection is not None) and (not isinstance(projection, dict))):
            raise ValueError("Projection must be a dictionary or None")

        # Validate limit
        if ((limit is not None) and (not isinstance(limit, int) or limit <= 0)):
            raise ValueError("Limit must be a positive integer or None")

        try:
            # Perform find_many operation
            cursor = self.db[collection].find(query or {}, projection)
            
            # Apply limit if specified
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert cursor to list
            found_documents = list(cursor)
            logger.info(f"Found {len(found_documents)} documents in {collection} for query: {query}")

            return found_documents

        except PyMongoError as e:
            logger.error(f"find_many failed on {collection}: {repr(e)}")
            return []

    
    def count_documents(self, collection: str, query: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents in collection
        
        Arguments:
        ----------
            collection       { str }       : Name of the collection to count documents in

            query      { Dict[str, Any ] } : Query to count_documents, defaults to empty dict

        Errors:
        -------
            ValueError                     : If query is not a dictionary

            PyMongoError                   : If count_documents operation fails for any reason

        Returns:
        --------
                    { int }                : Number of documents matching the query, 0 if an error occurs
        """
        # Validate query
        if query is not None and not isinstance(query, dict):
            raise ValueError("Query must be a dictionary or None")
        
        # Convert None to empty dict
        if query is None:
            query = {}
            
        try:
            # Perform the count_documents operation
            documents_count = self.db[collection].count_documents(query)
            logger.info(f"Counted {documents_count} documents in {collection} for query: {query}")

            return documents_count

        except PyMongoError as e:
            logger.error(f"count_documents failed on {collection}: {repr(e)}")
            return 0

    
    def get_collection_names(self) -> List[str]:
        """
        Returns list of collection names
        
        Errors:
        -------
            PyMongoError    : If getting collection names fails for any reason

        Returns:
        --------
            { List[str] }   : List of collection names, empty list if an error occurs
        """
        try:
            # Get collection names
            collection_names_list = self.db.list_collection_names()
            logger.info(f"Collection names: {collection_names_list}")

            return collection_names_list

        except PyMongoError as e:
            logger.error(f"Failed to get collection names: {repr(e)}")
            return []
    

    def close(self) -> None:
        """
        Close database connection
        """
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
