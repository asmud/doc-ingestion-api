"""
Embedding service for generating document and chunk-level embeddings using the nomic-embed-text model.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from docling_core.types.doc.document import DoclingDocument

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings using the local nomic-embed-text model."""
    
    def __init__(self, tokenizer_path: str, device: str = "cpu"):
        """
        Initialize the embedding service.
        
        Args:
            tokenizer_path: Path to the nomic tokenizer model
            device: Device to run the model on (cpu/cuda)
        """
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.model = None
        self.model_name = "nomic-embed-text-v1.5"
        self.embedding_dimension = 768
        
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                # Construct the actual model path with snapshots
                actual_model_path = Path(self.tokenizer_path) / "snapshots" / "f752c1ee2994831dcef5b1e446383bc1e1996d52"
                
                if not actual_model_path.exists():
                    raise FileNotFoundError(f"Model path does not exist: {actual_model_path}")
                
                logger.info(f"Loading embedding model from: {actual_model_path}")
                self.model = SentenceTransformer(str(actual_model_path), device=self.device)
                logger.info("Embedding model loaded successfully")
                
            except ImportError:
                raise ImportError("sentence-transformers library is required for embedding generation. Install with: pip install sentence-transformers")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
    
    def generate_document_embedding(self, document: DoclingDocument) -> List[float]:
        """
        Generate a single embedding vector for the entire document.
        
        Args:
            document: DoclingDocument to embed
            
        Returns:
            List of floats representing the document embedding
        """
        self._load_model()
        
        try:
            # Extract full text from document
            full_text = document.export_to_text()
            
            if not full_text.strip():
                logger.warning("Document has no text content for embedding")
                return [0.0] * self.embedding_dimension
            
            # Generate embedding
            embedding = self.model.encode(full_text, convert_to_tensor=False)
            
            # Ensure it's a list of floats
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            logger.debug(f"Generated document embedding with dimension: {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate document embedding: {e}")
            raise
    
    def generate_chunk_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Generate embedding vectors for each chunk.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            
        Returns:
            List of embedding vectors, one for each chunk
        """
        self._load_model()
        
        try:
            if not chunks:
                logger.warning("No chunks provided for embedding")
                return []
            
            # Extract text from chunks
            chunk_texts = []
            for chunk in chunks:
                text = chunk.get('text', '').strip()
                if not text:
                    # Handle empty chunks with zero vectors
                    chunk_texts.append("")
                else:
                    chunk_texts.append(text)
            
            if not any(chunk_texts):
                logger.warning("All chunks are empty for embedding")
                return [[0.0] * self.embedding_dimension] * len(chunks)
            
            # Generate embeddings in batch for efficiency
            embeddings = self.model.encode(chunk_texts, convert_to_tensor=False)
            
            # Ensure proper format
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            # Handle case where single chunk returns 1D array
            if len(chunks) == 1 and isinstance(embeddings[0], (int, float)):
                embeddings = [embeddings]
            
            logger.debug(f"Generated embeddings for {len(embeddings)} chunks")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate chunk embeddings: {e}")
            raise
    
    def generate_embeddings(self, document: DoclingDocument, chunks: List[Dict[str, Any]]) -> Tuple[List[float], List[List[float]]]:
        """
        Generate both document-level and chunk-level embeddings.
        
        Args:
            document: DoclingDocument to embed
            chunks: List of chunk dictionaries
            
        Returns:
            Tuple of (document_embedding, chunk_embeddings)
        """
        logger.info("Generating document and chunk embeddings")
        
        # Generate document embedding
        document_embedding = self.generate_document_embedding(document)
        
        # Generate chunk embeddings
        chunk_embeddings = self.generate_chunk_embeddings(chunks)
        
        logger.info(f"Generated embeddings: document_dim={len(document_embedding)}, chunks_count={len(chunk_embeddings)}")
        
        return document_embedding, chunk_embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "model_path": self.tokenizer_path
        }