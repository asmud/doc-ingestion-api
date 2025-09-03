"""
Embedding service for generating document and chunk-level embeddings using ONNX Runtime.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from docling_core.types.doc.document import DoclingDocument

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings using ONNX Runtime with Indonesian BERT model."""
    
    def __init__(self, tokenizer_path: str, device: str = "cpu"):
        """
        Initialize the embedding service.
        
        Args:
            tokenizer_path: Path to the ONNX tokenizer model
            device: Device to run the model on (cpu/cuda)
        """
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.tokenizer = None
        self.onnx_session = None
        self.model_name = "LazarusNLP-indobert-onnx"
        self.embedding_dimension = 768
        
    def _load_model(self):
        """Lazy load the ONNX model and tokenizer."""
        if self.tokenizer is None or self.onnx_session is None:
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer
                
                model_path = Path(self.tokenizer_path)
                if not model_path.exists():
                    raise FileNotFoundError(f"Model path does not exist: {model_path}")
                
                logger.info(f"Loading ONNX embedding model from: {model_path}")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                
                # Load ONNX model
                onnx_model_path = model_path / "model.onnx"
                if not onnx_model_path.exists():
                    raise FileNotFoundError(f"ONNX model file not found: {onnx_model_path}")
                
                # Set ONNX Runtime providers based on device
                providers = []
                if self.device.lower() == "cuda":
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                else:
                    providers = ['CPUExecutionProvider']
                
                self.onnx_session = ort.InferenceSession(str(onnx_model_path), providers=providers)
                logger.info("ONNX embedding model loaded successfully")
                
            except ImportError as e:
                raise ImportError(f"Required libraries missing. Install with: pip install onnxruntime transformers. Error: {e}")
            except Exception as e:
                logger.error(f"Failed to load ONNX embedding model: {e}")
                raise
    
    def _encode_text(self, text: str) -> List[float]:
        """
        Encode a single text string using ONNX Runtime.
        
        Args:
            text: Text to encode
            
        Returns:
            List of floats representing the text embedding
        """
        if not text.strip():
            return [0.0] * self.embedding_dimension
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="np"
        )
        
        # Run ONNX inference
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        
        # Note: This ONNX model doesn't use token_type_ids, so we skip it
        
        outputs = self.onnx_session.run(None, onnx_inputs)
        
        # Extract embeddings (usually the last hidden state)
        # Take mean pooling of token embeddings
        last_hidden_state = outputs[0]  # Shape: (batch_size, seq_len, hidden_size)
        attention_mask = inputs["attention_mask"]
        
        # Mean pooling with attention mask
        mask_expanded = np.expand_dims(attention_mask, -1)
        masked_embeddings = last_hidden_state * mask_expanded
        summed_embeddings = np.sum(masked_embeddings, axis=1)
        summed_mask = np.sum(mask_expanded, axis=1)
        mean_embeddings = summed_embeddings / summed_mask
        
        # Convert to list and return
        embedding = mean_embeddings[0].tolist()  # Take first (and only) item from batch
        
        return embedding

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
            
            # Generate embedding using ONNX
            embedding = self._encode_text(full_text)
            
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
            
            # Extract text from chunks and generate embeddings individually
            embeddings = []
            for chunk in chunks:
                text = chunk.get('text', '').strip()
                if not text:
                    # Handle empty chunks with zero vectors
                    embedding = [0.0] * self.embedding_dimension
                else:
                    # Generate embedding for this chunk
                    embedding = self._encode_text(text)
                
                embeddings.append(embedding)
            
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