#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core Document Processing Module - Handles PDF processing, indexing, and retrieval
"""

import os
import logging
import fitz  # PyMuPDF
import tika
from tika import parser
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import sqlite3
import json
from typing import List, Dict, Any, Optional, Union, Tuple
import tempfile
import shutil
import psycopg2

# convert to postgresql

# Optional Ray for distributed processing
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DocumentProcessor")

# Constants
VECTOR_DIM = 768  # Dimension for sentence-transformers embeddings
CHUNK_SIZE = 1000  # Characters per chunk for text splitting

class DocumentProcessor:
    """Core document processing capabilities for PDF handling"""
    
    def __init__(self, storage_dir="./documents", 
                 metadata_db_path="./documents/metadata.db",
                 vector_index_path="./documents/vector_index",
                 embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize the document processor
        
        Args:
            storage_dir: Base directory for document storage
            metadata_db_path: Path to SQLite database for metadata
            vector_index_path: Path to store FAISS indices
            embedding_model: SentenceTransformer model to use for embeddings
        """
        self.storage_dir = storage_dir
        self.metadata_db_path = metadata_db_path
        self.vector_index_path = vector_index_path
        self.embedding_model_name = embedding_model
        
        # Create directories if needed
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(os.path.dirname(metadata_db_path), exist_ok=True)
        os.makedirs(vector_index_path, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize Tika
        tika.initVM()
        
        # Load or create FAISS index
        self._initialize_vector_index()
        
        logger.info(f"Document processor initialized with storage at {storage_dir}")

    def _initialize_database(self):
        """Initialize the metadata database schema"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        # Create document metadata table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            title TEXT,
            doc_type TEXT,  -- 'patent', 'trademark', 'legal', etc.
            mime_type TEXT,
            page_count INTEGER,
            file_path TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT  -- JSON blob for flexible metadata
        )
        ''')
        
        # Create document chunks table for text storage
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            page_numbers TEXT NOT NULL,  -- JSON array of page numbers
            vector_id INTEGER,  -- ID in the FAISS index
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _initialize_vector_index(self):
        """Initialize or load the FAISS vector index"""
        index_path = os.path.join(self.vector_index_path, "document_index.faiss")
        
        if os.path.exists(index_path):
            # Load existing index
            self.vector_index = faiss.read_index(index_path)
            logger.info(f"Loaded existing vector index with {self.vector_index.ntotal} vectors")
        else:
            # Create new index
            self.vector_index = faiss.IndexFlatL2(VECTOR_DIM)
            logger.info("Created new vector index")
    
    def process_document(self, file_path: str, doc_type: Optional[str] = None) -> str:
        """
        Process a document and store it in the system
        
        Args:
            file_path: Path to the document file
            doc_type: Type of document (patent, trademark, etc.) or None for auto-detection
            
        Returns:
            document_id: Unique ID for the processed document
        """
        # Generate a unique ID for this document
        import uuid
        document_id = str(uuid.uuid4())
        
        # Extract filename from path
        filename = os.path.basename(file_path)
        
        # Determine document type if not provided
        if doc_type is None:
            doc_type = self._detect_document_type(filename, file_path)
        
        # Copy the file to storage directory
        target_dir = os.path.join(self.storage_dir, doc_type)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, f"{document_id}_{filename}")
        shutil.copy2(file_path, target_path)
        
        # Process the document
        metadata = self._extract_metadata(target_path)
        chunks = self._extract_and_chunk_text(target_path)
        
        # Store in database
        self._store_document_metadata(document_id, filename, doc_type, target_path, metadata)
        self._store_document_chunks(document_id, chunks)
        
        # Update vector index
        self._update_vector_index()
        
        logger.info(f"Document {filename} processed with ID {document_id}")
        return document_id
    
    def batch_process_documents(self, file_paths: List[str], 
                                doc_types: Optional[List[str]] = None,
                                use_ray: bool = False) -> List[str]:
        """
        Process multiple documents in batch
        
        Args:
            file_paths: List of document file paths
            doc_types: Optional list of document types (must match file_paths length)
            use_ray: Whether to use Ray for parallel processing
            
        Returns:
            List of document IDs
        """
        # Initialize doc_types if not provided
        if doc_types is None:
            doc_types = [None] * len(file_paths)
        elif len(doc_types) != len(file_paths):
            raise ValueError("doc_types list must match file_paths length")
        
        document_ids = []
        
        if use_ray and RAY_AVAILABLE:
            # Initialize Ray if not already
            if not ray.is_initialized():
                ray.init()
            
            # Define remote function
            @ray.remote
            def process_doc_remote(processor, file_path, doc_type):
                return processor.process_document(file_path, doc_type)
            
            # Process in parallel
            results = []
            for i, (file_path, doc_type) in enumerate(zip(file_paths, doc_types)):
                results.append(process_doc_remote.remote(self, file_path, doc_type))
            
            # Get results
            document_ids = ray.get(results)
        else:
            # Process sequentially
            for file_path, doc_type in zip(file_paths, doc_types):
                doc_id = self.process_document(file_path, doc_type)
                document_ids.append(doc_id)
        
        return document_ids
    
    def _detect_document_type(self, filename: str, file_path: str) -> str:
        """
        Automatically detect document type from filename and content
        
        Args:
            filename: Name of the file
            file_path: Path to the file
            
        Returns:
            Detected document type
        """
        # First try to detect from filename patterns
        filename_lower = filename.lower()
        
        if filename_lower.startswith(("patent", "pat-", "p-")):
            return "patent"
        elif filename_lower.startswith(("tm-", "trademark")):
            return "trademark"
        elif filename_lower.startswith(("ptab-", "ipr-", "pgr-")):
            return "ptab"
        
        # If not detected from filename, try content analysis
        try:
            # Use PyMuPDF to get first page text
            doc = fitz.open(file_path)
            if doc.page_count > 0:
                first_page_text = doc[0].get_text()
                
                # Check for patent indicators
                if any(term in first_page_text for term in 
                       ["United States Patent", "Patent No", "US Patent"]):
                    return "patent"
                
                # Check for trademark indicators
                if any(term in first_page_text for term in 
                       ["Trademark", "USTPO", "Registration No.", "Serial No."]):
                    return "trademark"
                
                # Check for PTAB indicators
                if any(term in first_page_text for term in 
                       ["Patent Trial and Appeal Board", "PTAB", "IPR", "PGR"]):
                    return "ptab"
        except Exception as e:
            logger.warning(f"Error detecting document type from content: {str(e)}")
        
        # Default to general if not detected
        return "general"
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a document using Tika
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary of metadata
        """
        try:
            # Extract metadata using Tika
            parsed = parser.from_file(file_path)
            metadata = parsed["metadata"]
            
            # Extract PyMuPDF metadata as well
            try:
                doc = fitz.open(file_path)
                # Add page count
                metadata["page_count"] = doc.page_count
                # Add document info
                for key, value in doc.metadata.items():
                    if key not in metadata and value:
                        metadata[key] = value
            except Exception as e:
                logger.warning(f"Error extracting PyMuPDF metadata: {str(e)}")
            
            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata with Tika: {str(e)}")
            # Return basic metadata as fallback
            doc = fitz.open(file_path)
            return {
                "page_count": doc.page_count,
                **doc.metadata
            }

    def extract_patent_metadata(self, file_path, text):
        """
        Extract patent-specific metadata
        
        Args:
            file_path: Path to the document
            text: Extracted text from first few pages
            
        Returns:
            Dictionary of patent metadata
        """
        metadata = {}
        
        # Extract patent number
        patent_patterns = [
            r"United States Patent\s*(?:No\.?)?\s*([0-9,]{1,3}(?:,[0-9]{3})+|[0-9]{5,})",
            r"US\s*(?:D|RE)?\s*([0-9,]{1,3}(?:,[0-9]{3})+|[0-9]{5,})",
            r"Patent No\.?\s*:?\s*([0-9,]{1,3}(?:,[0-9]{3})+|[0-9]{5,})",
        ]
        
        for pattern in patent_patterns:
            match = re.search(pattern, text)
            if match:
                patent_number = match.group(1).replace(",", "")
                metadata["patent_number"] = patent_number
                break
        
        # Extract filing date
        date_pattern = r"Filed:\s*([A-Za-z]{3}\.?\s+\d{1,2},\s+\d{4})"
        match = re.search(date_pattern, text)
        if match:
            metadata["filing_date"] = match.group(1)
        
        # Extract assignee/patent owner
        assignee_pattern = r"Assignee:\s*(.*?)(?:\n|\(|$)"
        match = re.search(assignee_pattern, text)
        if match:
            metadata["patent_owner"] = match.group(1).strip()
        
        return metadata

    def extract_ptab_metadata(self, file_path, text):
        """
        Extract PTAB-specific metadata
        
        Args:
            file_path: Path to the document
            text: Extracted text from first few pages
            
        Returns:
            Dictionary of PTAB metadata
        """
        metadata = {}
        
        # Determine PTAB document type
        if re.search(r"PETITION FOR (?:INTER PARTES|POST-GRANT|COVERED BUSINESS METHOD) REVIEW", text, re.IGNORECASE):
            metadata["ptab_doc_type"] = "petition"
        elif re.search(r"PATENT OWNER(?:'S)?\s*(?:PRELIMINARY)?\s*RESPONSE", text, re.IGNORECASE):
            metadata["ptab_doc_type"] = "patent_owner_response"
        elif re.search(r"DECISION\s*(?:Denying|Granting|on)\s*Institution", text, re.IGNORECASE):
            metadata["ptab_doc_type"] = "institution_decision"
        elif re.search(r"FINAL WRITTEN DECISION", text, re.IGNORECASE):
            metadata["ptab_doc_type"] = "final_written_decision"
        elif re.search(r"DECISION\s*(?:Denying|Granting)\s*Rehearing", text, re.IGNORECASE):
            metadata["ptab_doc_type"] = "rehearing_decision"
        else:
            metadata["ptab_doc_type"] = "other"
        
        # Determine proceeding type
        if re.search(r"IPR\d{4}-\d{5}", text):
            metadata["proceeding_type"] = "ipr"
            # Extract IPR number
            ipr_match = re.search(r"(IPR\d{4}-\d{5})", text)
            if ipr_match:
                metadata["proceeding_number"] = ipr_match.group(1)
        elif re.search(r"PGR\d{4}-\d{5}", text):
            metadata["proceeding_type"] = "pgr"
            # Extract PGR number
            pgr_match = re.search(r"(PGR\d{4}-\d{5})", text)
            if pgr_match:
                metadata["proceeding_number"] = pgr_match.group(1)
        elif re.search(r"CBM\d{4}-\d{5}", text):
            metadata["proceeding_type"] = "cbm"
            # Extract CBM number
            cbm_match = re.search(r"(CBM\d{4}-\d{5})", text)
            if cbm_match:
                metadata["proceeding_number"] = cbm_match.group(1)
        
        # Extract patent number
        patent_match = re.search(r"Patent\s*(?:No\.?)?:?\s*(?:US)?\s*([0-9,]{5,})", text)
        if patent_match:
            metadata["patent_number"] = patent_match.group(1).replace(",", "")
        
        # Extract petitioner
        petitioner_pattern = r"Petitioner:?\s*(.*?)(?:[\n,]|$)"
        match = re.search(petitioner_pattern, text)
        if match:
            metadata["petitioner"] = match.group(1).strip()
        
        # Extract patent owner
        owner_pattern = r"Patent Owner:?\s*(.*?)(?:[\n,]|$)"
        match = re.search(owner_pattern, text)
        if match:
            metadata["patent_owner"] = match.group(1).strip()
        
        # Extract filing date
        date_pattern = r"Filed:?\s*([A-Za-z]{3}\.?\s+\d{1,2},\s+\d{4})"
        match = re.search(date_pattern, text)
        if match:
            metadata["filing_date"] = match.group(1)
        
        # Extract claims at issue
        claims_pattern = r"Claims? (?:at issue|challenged)(?::|are|:are)\s*((?:\d+(?:[-,\s]+\d+)*)+)"
        match = re.search(claims_pattern, text, re.IGNORECASE)
        if match:
            metadata["claims_at_issue"] = match.group(1).strip()
        
        # Extract result if it's a decision
        if "decision" in metadata.get("ptab_doc_type", ""):
            if re.search(r"institute\s*(?:a|an)\s*(?:inter partes|post-grant) review", text, re.IGNORECASE):
                metadata["result"] = "instituted"
            elif re.search(r"denied\s*institution", text, re.IGNORECASE):
                metadata["result"] = "denied"
            elif re.search(r"claims?\s*\d+.*\s*(?:is|are)\s*unpatentable", text, re.IGNORECASE):
                metadata["result"] = "claims invalid"
            elif re.search(r"claims?\s*\d+.*\s*(?:is|are)\s*not\s*(?:shown to be)?\s*unpatentable", text, re.IGNORECASE):
                metadata["result"] = "claims not invalid"
        
        return metadata

    def extract_trademark_metadata(self, file_path, text):
        """
        Extract trademark-specific metadata
        
        Args:
            file_path: Path to the document
            text: Extracted text from first few pages
            
        Returns:
            Dictionary of trademark metadata
        """
        metadata = {}
        
        # Extract serial number
        serial_pattern = r"Serial (?:No\.?|Number):\s*(\d{8})"
        match = re.search(serial_pattern, text)
        if match:
            metadata["serial_number"] = match.group(1)
        
        # Extract registration number if registered
        reg_pattern = r"Registration (?:No\.?|Number):\s*(\d{7,})"
        match = re.search(reg_pattern, text)
        if match:
            metadata["registration_number"] = match.group(1)
        
        # Extract mark
        mark_pattern = r"Mark:\s*(.*?)(?:\n|$)"
        match = re.search(mark_pattern, text)
        if match:
            metadata["mark"] = match.group(1).strip()
        
        # Extract owner
        owner_pattern = r"Owner:\s*(.*?)(?:\n|$)"
        match = re.search(owner_pattern, text)
        if match:
            metadata["owner"] = match.group(1).strip()
        
        # Extract filing date
        filing_pattern = r"Filed:\s*([A-Za-z]{3}\.?\s+\d{1,2},\s+\d{4})"
        match = re.search(filing_pattern, text)
        if match:
            metadata["filing_date"] = match.group(1)
        
        # Determine if it's a TTAB document
        if re.search(r"Trademark Trial and Appeal Board", text, re.IGNORECASE):
            metadata["is_ttab"] = True
            
            # Determine TTAB document type
            if re.search(r"Notice of Opposition", text, re.IGNORECASE):
                metadata["ttab_doc_type"] = "opposition"
            elif re.search(r"Petition for Cancellation", text, re.IGNORECASE):
                metadata["ttab_doc_type"] = "cancellation"
            elif re.search(r"Answer", text, re.IGNORECASE):
                metadata["ttab_doc_type"] = "answer"
            elif re.search(r"Decision", text, re.IGNORECASE):
                metadata["ttab_doc_type"] = "decision"
        
        return metadata

    def generate_standardized_filename(self, doc_type, metadata):
        """
        Generate standardized filename based on metadata
        
        Args:
            doc_type: Document type
            metadata: Document metadata
            
        Returns:
            Standardized filename
        """
        import datetime
        
        # Format date
        try:
            date_str = metadata.get("filing_date")
            if date_str:
                date_obj = datetime.datetime.strptime(date_str, "%b. %d, %Y")
                date_formatted = date_obj.strftime("%Y%m%d")
            else:
                date_formatted = datetime.datetime.now().strftime("%Y%m%d")
        except:
            date_formatted = datetime.datetime.now().strftime("%Y%m%d")
        
        if doc_type == "patent":
            patent_number = metadata.get("patent_number", "unknown")
            return f"{patent_number}-patent-grant-{date_formatted}.pdf"
        
        elif doc_type == "ptab":
            patent_number = metadata.get("patent_number", "unknown")
            ptab_type = metadata.get("ptab_doc_type", "document")
            return f"{patent_number}-ptab-{ptab_type}-{date_formatted}.pdf"
        
        elif doc_type == "trademark":
            serial_number = metadata.get("serial_number", "unknown")
            if metadata.get("is_ttab"):
                ttab_type = metadata.get("ttab_doc_type", "document")
                return f"{serial_number}-ttab-{ttab_type}-{date_formatted}.pdf"
            else:
                return f"{serial_number}-tm-document-{date_formatted}.pdf"
        
        else:
            # Generic filename for unknown types
            import uuid
            return f"doc-{uuid.uuid4()[:8]}-{date_formatted}.pdf"


    def _extract_and_chunk_text(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from a document and split into chunks
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of chunks with page numbers and text
        """
        chunks = []
        
        try:
            doc = fitz.open(file_path)
            
            # Process each page
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                
                # Skip empty pages
                if not text.strip():
                    continue
                
                # Split into chunks
                for i in range(0, len(text), CHUNK_SIZE):
                    chunk_text = text[i:i+CHUNK_SIZE]
                    # Only add non-empty chunks
                    if chunk_text.strip():
                        chunks.append({
                            "text": chunk_text,
                            "page_numbers": [page_num + 1],  # 1-based page numbers for user-friendly display
                            "chunk_index": len(chunks)
                        })
            
            return chunks
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return []
    
    def _store_document_metadata(self, document_id: str, filename: str, 
                               doc_type: str, file_path: str, 
                               metadata: Dict[str, Any]) -> None:
        """
        Store document metadata in the database
        
        Args:
            document_id: Unique document ID
            filename: Original filename
            doc_type: Document type
            file_path: Path to the stored file
            metadata: Document metadata
        """
        conn = sqlite3.connect(self.metadata_db_path)
        try:
            cursor = conn.cursor()
            
            # Extract title from metadata if available
            title = metadata.get("title", filename)
            
            # Get page count
            page_count = metadata.get("page_count", 0)
            
            # Get MIME type
            mime_type = metadata.get("Content-Type", "application/pdf")
            
            # Store metadata as JSON
            metadata_json = json.dumps(metadata)
            
            # Insert into database
            cursor.execute('''
            INSERT OR REPLACE INTO documents 
            (id, filename, title, doc_type, mime_type, page_count, file_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (document_id, filename, title, doc_type, mime_type, page_count, file_path, metadata_json))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing document metadata: {str(e)}")
            conn.rollback()
        finally:
            conn.close()
    
    def _store_document_chunks(self, document_id: str, chunks: List[Dict[str, Any]]) -> None:
        """
        Store document chunks in the database
        
        Args:
            document_id: Unique document ID
            chunks: List of text chunks with page numbers
        """
        if not chunks:
            return
        
        conn = sqlite3.connect(self.metadata_db_path)
        try:
            cursor = conn.cursor()
            
            # Generate embeddings for all chunks
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_model.encode(texts)
            
            # Add vectors to FAISS index
            vector_ids = list(range(self.vector_index.ntotal, self.vector_index.ntotal + len(embeddings)))
            self.vector_index.add(np.array(embeddings).astype('float32'))
            
            # Store chunks in database
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_{chunk['chunk_index']}"
                page_numbers_json = json.dumps(chunk["page_numbers"])
                
                cursor.execute('''
                INSERT INTO document_chunks
                (id, document_id, chunk_index, text, page_numbers, vector_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (chunk_id, document_id, chunk["chunk_index"], 
                      chunk["text"], page_numbers_json, vector_ids[i]))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing document chunks: {str(e)}")
            conn.rollback()
        finally:
            conn.close()
    
    def _update_vector_index(self) -> None:
        """Save the updated vector index to disk"""
        try:
            index_path = os.path.join(self.vector_index_path, "document_index.faiss")
            faiss.write_index(self.vector_index, index_path)
            logger.info(f"Vector index updated with {self.vector_index.ntotal} total vectors")
        except Exception as e:
            logger.error(f"Error saving vector index: {str(e)}")
    
    def query_documents(self, query: str, k: int = 5, 
                      filter_doc_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for documents matching a query
        
        Args:
            query: The search query
            k: Number of results to return
            filter_doc_type: Optional document type filter
            
        Returns:
            List of relevant document chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search in FAISS
        distances, indices = self.vector_index.search(
            np.array([query_embedding]).astype('float32'), k*3)  # Get more results for filtering
        
        # Retrieve the actual chunks
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        results = []
        for idx in indices[0]:
            # Skip invalid indices
            if idx == -1 or idx >= self.vector_index.ntotal:
                continue
                
            # Get the chunk
            cursor.execute('''
            SELECT c.id, c.document_id, c.text, c.page_numbers, 
                   d.filename, d.title, d.doc_type
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.vector_id = ?
            ''', (int(idx),))
            
            row = cursor.fetchone()
            if row:
                chunk_id, doc_id, text, page_numbers, filename, title, doc_type = row
                
                # Apply doc_type filter if specified
                if filter_doc_type and doc_type != filter_doc_type:
                    continue
                
                results.append({
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "content": text,
                    "page_numbers": json.loads(page_numbers),
                    "filename": filename,
                    "title": title,
                    "doc_type": doc_type,
                    "citation": f"{title or filename}, Page {json.loads(page_numbers)[0]}"
                })
                
                # Stop when we have enough results
                if len(results) >= k:
                    break
        
        conn.close()
        return results
    
    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific document
        
        Args:
            document_id: The document ID
            
        Returns:
            Document metadata or None if not found
        """
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, filename, title, doc_type, mime_type, page_count, 
               file_path, upload_date, metadata
        FROM documents
        WHERE id = ?
        ''', (document_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            "id": row[0],
            "filename": row[1],
            "title": row[2],
            "doc_type": row[3],
            "mime_type": row[4],
            "page_count": row[5],
            "file_path": row[6],
            "upload_date": row[7],
            "metadata": json.loads(row[8]) if row[8] else {}
        }
    
    def get_document_content(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all content chunks for a document
        
        Args:
            document_id: The document ID
            
        Returns:
            List of document chunks
        """
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, chunk_index, text, page_numbers
        FROM document_chunks
        WHERE document_id = ?
        ORDER BY chunk_index
        ''', (document_id,))
        
        chunks = []
        for row in cursor.fetchall():
            chunks.append({
                "id": row[0],
                "chunk_index": row[1],
                "text": row[2],
                "page_numbers": json.loads(row[3])
            })
        
        conn.close()
        return chunks