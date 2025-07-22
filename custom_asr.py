"""
Custom ASR processor for Whisper model integration.
This module provides direct integration with local Whisper models using the transformers library,
bypassing Docling's built-in ASR limitations.
"""

import torch
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor
)
import librosa
from logging_config import get_asr_logger

# Module-level logger initialization
logger = get_asr_logger(__name__)

# Global warning suppression for transformers
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*generation_config.*")
warnings.filterwarnings("ignore", message=".*SuppressTokensAtBeginLogitsProcessor.*")
warnings.filterwarnings("ignore", message=".*logits_process.*")
warnings.filterwarnings("ignore", message=".*model_kwargs.*")
warnings.filterwarnings("ignore", message=".*default values have been modified.*")

class IndonesianWhisperProcessor:
    """Custom ASR processor for Whisper models using transformers library."""
    
    def __init__(self, model_path: str, device: str = "cpu", max_new_tokens: int = 420, 
                 chunk_duration: float = 20.0, chunk_overlap: float = 3.0, model_name: str = "cahya/whisper-medium-id"):
        """
        Initialize Whisper processor using transformers library.
        
        Args:
            model_path: Path to the local Whisper model (transformers format)
            device: Device to run inference on (cpu/cuda/mps)
            max_new_tokens: Maximum new tokens to generate (must be < 448 for Whisper)
            chunk_duration: Duration of each audio chunk in seconds
            chunk_overlap: Overlap between chunks in seconds
            model_name: Model name for metadata (e.g., "cahya/whisper-medium-id")
        """
        self.model_path = Path(model_path)
        self.device = device
        self.max_new_tokens = min(max_new_tokens, 440)  # Safely under 448 limit
        self.chunk_duration = chunk_duration
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.model: Optional[AutoModelForSpeechSeq2Seq] = None
        self.processor: Optional[AutoProcessor] = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model from local path using transformers library."""
        try:
            logger.info(f"🔄 Loading Whisper model: {self.model_name}")
            logger.info(f"📁 Model path: {self.model_path}")
            logger.info(f"📂 Path exists: {self.model_path.exists()}")
            logger.info(f"🖥️  Target device: {self.device}")
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model directory not found: {self.model_path}")
            
            # Load model and processor from local path
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=True   # Force local loading only
            )
            
            self.processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                local_files_only=True
            )
            
            # Move model to device
            self.model.to(self.device)  # type: ignore
            
            logger.info(f"✅ Successfully loaded: {self.model_name}")
            logger.info(f"🎯 Model device: {self.device}")
            logger.info(f"🔧 Max tokens: {self.max_new_tokens}")
            logger.info(f"⏱️  Chunk duration: {self.chunk_duration}s")
            logger.info(f"🔄 Chunk overlap: {self.chunk_overlap}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model '{self.model_name}': {e}")
            logger.error(f"💡 Verify model exists at: {self.model_path}")
            raise
    
    def transcribe_audio(self, audio_path: str, language: str = "id") -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper model directly.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: "id" for Indonesian, "auto" for auto-detection)
            
        Returns:
            Dict containing transcription and metadata
        """
        try:
            logger.info(f"🎵 Transcribing audio: {audio_path}")
            
            # Check if model and processor are loaded
            if self.model is None or self.processor is None:
                raise RuntimeError("Model or processor not loaded. Call _load_model() first.")
            
            # Load audio file with error handling and warning suppression
            try:
                with warnings.catch_warnings():
                    # Suppress PySoundFile and librosa deprecation warnings
                    warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
                    warnings.filterwarnings("ignore", message=".*audioread_load.*")
                    warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
                    warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
                    
                    audio_array, sample_rate = librosa.load(audio_path, sr=16000)
                    audio_duration = len(audio_array) / sample_rate
            except Exception as e:
                logger.error(f"Failed to load audio file {audio_path}: {e}")
                raise RuntimeError(f"Audio loading failed: {e}")
            
            logger.info(f"📊 Audio duration: {audio_duration:.2f} seconds")
            
            # Process long audio in overlapping chunks for better accuracy
            chunk_samples = int(self.chunk_duration * 16000)
            overlap_samples = int(self.chunk_overlap * 16000)
            step_samples = chunk_samples - overlap_samples  # Step size with overlap
            
            transcription_parts = []
            previous_text = ""  # To handle overlap deduplication
            
            # Process audio in overlapping chunks
            total_chunks = (len(audio_array) - 1) // step_samples + 1
            logger.info(f"📊 Processing {total_chunks} audio chunks (duration: {self.chunk_duration}s, overlap: {self.chunk_overlap}s)")
            
            for chunk_idx, start_idx in enumerate(range(0, len(audio_array), step_samples)):
                end_idx = min(start_idx + chunk_samples, len(audio_array))
                chunk = audio_array[start_idx:end_idx]
                
                # Skip very short chunks (less than 2 seconds for better efficiency)
                if len(chunk) < 32000:
                    logger.info(f"⏭️  Skipping chunk {chunk_idx + 1}/{total_chunks} (too short: {len(chunk)/16000:.1f}s)")
                    continue
                
                logger.info(f"🔄 Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk)/16000:.1f}s)")
                
                # Process audio chunk with the processor
                inputs = self.processor(  # type: ignore
                    chunk,
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                
                # Move inputs to device
                input_features = inputs.input_features.to(self.device)
                
                # Create attention mask for reliable results
                attention_mask = torch.ones(
                    input_features.shape[:2], 
                    dtype=torch.long, 
                    device=self.device
                )
                
                with torch.no_grad():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # Ignore ALL warnings during generation
                        
                        predicted_ids = self.model.generate(  # type: ignore
                            input_features,
                            attention_mask=attention_mask,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,  # Deterministic sampling
                            temperature=0.0,  # Zero temperature for maximum determinism
                            pad_token_id=self.processor.tokenizer.pad_token_id,  # type: ignore
                            eos_token_id=self.processor.tokenizer.eos_token_id  # type: ignore
                        )
                
                # Decode the chunk transcription
                chunk_text = self.processor.batch_decode(  # type: ignore
                    predicted_ids,
                    skip_special_tokens=True
                )[0]
                
                if chunk_text.strip():  # Only add non-empty transcriptions
                    # Simplified overlap detection for better performance
                    if previous_text and len(transcription_parts) > 0:
                        chunk_words = chunk_text.strip().split()
                        prev_words = previous_text.split()
                        
                        # Quick overlap check - limit to 5 words for performance
                        max_overlap = min(len(chunk_words), len(prev_words), 5)
                        overlap_found = 0
                        
                        for i in range(1, max_overlap + 1):
                            if prev_words[-i:] == chunk_words[:i]:
                                overlap_found = i
                                break  # Take first match for performance
                        
                        if overlap_found > 0:
                            chunk_text = " ".join(chunk_words[overlap_found:])
                    
                    if chunk_text.strip():  # Check again after overlap removal
                        transcription_parts.append(chunk_text.strip())
                        previous_text = chunk_text.strip()
                        if chunk_idx % 5 == 0:  # Log every 5th chunk to reduce log spam
                            logger.info(f"🎵 Processed {chunk_idx + 1}/{total_chunks} chunks")
            
            transcription_text = " ".join(transcription_parts)
            
            transcription = {
                "text": transcription_text,
                "chunks": [],  # No chunks for now
                "language": language,
                "model_used": self.model_name,
                "word_count": len(transcription_text.split()) if transcription_text else 0,
                "char_count": len(transcription_text) if transcription_text else 0
            }
            
            logger.info(f"✅ Transcription completed: {transcription['word_count']} words, {transcription['char_count']} chars")
            return transcription
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            raise
    
    def format_as_docling_texts(self, transcription: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format transcription result as Docling text elements.
        
        Args:
            transcription: Result from transcribe_audio()
            
        Returns:
            List of text elements compatible with DoclingDocument
        """
        texts = []
        
        if transcription.get("chunks") and len(transcription["chunks"]) > 0:
            # Use chunk-based formatting with timestamps
            for i, chunk in enumerate(transcription["chunks"]):
                timestamp = chunk.get("timestamp", [0, 0])
                start_time = timestamp[0] if len(timestamp) > 0 else 0
                end_time = timestamp[1] if len(timestamp) > 1 else start_time
                
                text_element = {
                    "self_ref": f"#/texts/{i}",
                    "parent": {"$ref": "#/body"},
                    "children": [],
                    "content_layer": "body",
                    "label": "text",
                    "prov": [],
                    "orig": f"[time: {start_time}-{end_time}]  {chunk['text']}",
                    "text": f"[time: {start_time}-{end_time}]  {chunk['text']}"
                }
                texts.append(text_element)
        else:
            # Single text element without timestamps
            text_content = transcription.get("text", "")
            text_element = {
                "self_ref": "#/texts/0",
                "parent": {"$ref": "#/body"},
                "children": [],
                "content_layer": "body", 
                "label": "text",
                "prov": [],
                "orig": text_content,
                "text": text_content
            }
            texts.append(text_element)
        
        return texts

# Global instance for reuse
_indonesian_processor: Optional[IndonesianWhisperProcessor] = None

def get_indonesian_processor(model_path: str, device: str = "cpu", max_new_tokens: int = 420, 
                           chunk_duration: float = 20.0, chunk_overlap: float = 3.0, model_name: str = "cahya/whisper-medium-id") -> IndonesianWhisperProcessor:
    """Get or create Indonesian Whisper processor instance."""
    global _indonesian_processor
    
    if _indonesian_processor is None:
        _indonesian_processor = IndonesianWhisperProcessor(model_path, device, max_new_tokens, chunk_duration, chunk_overlap, model_name)
    
    return _indonesian_processor

def process_indonesian_audio(audio_path: str, model_path: str, device: str = "cpu", max_new_tokens: int = 420,
                           chunk_duration: float = 20.0, chunk_overlap: float = 3.0, model_name: str = "cahya/whisper-medium-id") -> Dict[str, Any]:
    """
    Process Indonesian audio file and return Docling-compatible result.
    
    Args:
        audio_path: Path to audio file
        model_path: Path to Indonesian Whisper model
        device: Device for inference
        model_name: Model name for metadata (e.g., "cahya/whisper-medium-id")
        
    Returns:
        Docling-compatible document structure
    """
    try:
        # Get processor instance
        processor = get_indonesian_processor(model_path, device, max_new_tokens, chunk_duration, chunk_overlap, model_name)
        
        # Transcribe audio
        transcription = processor.transcribe_audio(audio_path, language="id")
        
        # Format as Docling document structure
        texts = processor.format_as_docling_texts(transcription)
        
        # Create Docling-compatible document structure
        document_data = {
            "schema_name": "DoclingDocument",
            "version": "1.5.0",
            "name": Path(audio_path).stem,
            "origin": {
                "mimetype": "audio/mpeg",
                "binary_hash": abs(hash(audio_path)),  # Ensure positive hash
                "filename": Path(audio_path).name
            },
            "furniture": {
                "self_ref": "#/furniture",
                "children": [],
                "content_layer": "furniture",
                "name": "_root_",
                "label": "unspecified"
            },
            "body": {
                "self_ref": "#/body",
                "children": [{"$ref": f"#/texts/{i}"} for i in range(len(texts))],
                "content_layer": "body",
                "name": "_root_",
                "label": "unspecified"
            },
            "groups": [],
            "texts": texts,
            "pictures": [],
            "tables": [],
            "key_value_items": [],
            "form_items": [],
            "pages": {}
        }
        
        return {
            "document_data": document_data,
            "transcription": transcription,
            "extraction_method": "custom_indonesian_whisper",
            "model_used": model_name
        }
        
    except Exception as e:
        logger.error(f"❌ Indonesian audio processing failed: {e}")
        raise