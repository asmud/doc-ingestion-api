import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from docling.document_converter import DocumentConverter, PdfFormatOption, AudioFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, TesseractCliOcrOptions, AsrPipelineOptions
from docling.datamodel.pipeline_options_asr_model import InlineAsrNativeWhisperOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from config import ModelConfig
from custom_asr import process_indonesian_audio
from logging_config import get_pipeline_logger

logger = get_pipeline_logger(__name__)

class DocumentIntelligencePipeline:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig.from_project_root()
        self.config.validate_models_exist()
        self._setup_environment_variables()
        self._setup_pipeline()
    
    def _setup_environment_variables(self):
        os.environ['EASYOCR_MODEL_DIR'] = str(self.config.easyocr_models_dir)
        
        # Set HuggingFace cache to our local models directory
        os.environ['HF_HOME'] = str(self.config.models_dir)
        os.environ['HF_HUB_CACHE'] = str(self.config.models_dir)  
        os.environ['TRANSFORMERS_CACHE'] = str(self.config.models_dir)
        os.environ['HF_DATASETS_CACHE'] = str(self.config.models_dir)
        
        # Force offline mode to use local models only
        os.environ['TRANSFORMERS_OFFLINE'] = '1' if self.config.transformers_offline else '0'
        os.environ['HF_DATASETS_OFFLINE'] = '1' if self.config.hf_datasets_offline else '0'
        os.environ['HF_HUB_OFFLINE'] = '1' if self.config.transformers_offline else '0'
        
        # Set ASR language for Whisper if specified
        if self.config.asr_language and self.config.asr_language != "auto":
            os.environ['WHISPER_LANGUAGE'] = self.config.asr_language
            
    def _setup_pipeline(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.artifacts_path = str(self.config.models_dir)
        
        easyocr_options = EasyOcrOptions()
        easyocr_options.model_storage_directory = str(self.config.easyocr_models_dir)
        easyocr_options.download_enabled = False
        easyocr_options.lang = ["id", "en"]
        easyocr_options.force_full_page_ocr = self.config.force_full_page_ocr
        easyocr_options.use_gpu = (self.config.device.lower() != "cpu")
        
        tesseract_options = TesseractCliOcrOptions()
        tesseract_options.lang = ["ind", "eng"]
        tesseract_options.force_full_page_ocr = self.config.force_full_page_ocr
        
        if self.config.default_ocr_engine.lower() == "tesseract":
            pipeline_options.ocr_options = tesseract_options
        else:
            pipeline_options.ocr_options = easyocr_options
        
        # Configure ASR with Indonesian language if specified
        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend
            )
        }
        
        # Configure AUDIO format option with Indonesian language
        # Docling only supports predefined OpenAI models, so use 'small' with Indonesian language
        try:
            asr_pipeline_options = AsrPipelineOptions()
            
            # Use OpenAI small model with forced Indonesian language
            indonesian_asr_options = InlineAsrNativeWhisperOptions(
                repo_id="small",       # Use OpenAI small model (supported by Docling)
                language="id",         # Force Indonesian language detection
                verbose=True,
                timestamps=True,
                temperature=0.0
            )
            
            asr_pipeline_options.asr_options = indonesian_asr_options
            
            audio_format_option = AudioFormatOption(
                pipeline_options=asr_pipeline_options
            )
            format_options[InputFormat.AUDIO] = audio_format_option
            
        except Exception as e:
            print(f"⚠️ Warning: Indonesian ASR configuration failed, trying fallback: {e}")
            # Fallback to tiny model with Indonesian language
            try:
                asr_pipeline_options = AsrPipelineOptions()
                fallback_asr_options = InlineAsrNativeWhisperOptions(
                    repo_id="tiny",    # Smallest OpenAI model as fallback
                    language="id",     # Indonesian language
                    verbose=True
                )
                asr_pipeline_options.asr_options = fallback_asr_options
                
                audio_format_option = AudioFormatOption(
                    pipeline_options=asr_pipeline_options
                )
                format_options[InputFormat.AUDIO] = audio_format_option
                print(f"✅ Configured fallback Whisper 'tiny' model with Indonesian language")
            except Exception as e2:
                print(f"⚠️ Warning: All ASR configurations failed, using system default: {e2}")
        
        self.converter = DocumentConverter(format_options=format_options)
    
    def _is_audio_file(self, file_path: Union[str, Path]) -> bool:
        """Check if file is an audio file."""
        audio_extensions = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.wma', '.opus'}
        return Path(file_path).suffix.lower() in audio_extensions
    
    def _process_indonesian_audio(self, audio_path: Union[str, Path]) -> DoclingDocument:
        """Process audio file using custom Indonesian Whisper model."""
        try:
            logger.info(f"🎵 Processing Indonesian audio: {audio_path}")
            
            # Check if Indonesian model is available
            if not (hasattr(self.config, 'whisper_model_dir') and self.config.whisper_model_dir.exists()):
                raise ValueError("Indonesian Whisper model not found, falling back to standard processing")
            
            # Process audio with custom Indonesian processor
            result = process_indonesian_audio(
                audio_path=str(audio_path),
                model_path=str(self.config.whisper_model_dir),
                device=self.config.device,
                max_new_tokens=getattr(self.config, 'whisper_max_new_tokens', 420),
                chunk_duration=getattr(self.config, 'whisper_chunk_duration', 20.0),
                chunk_overlap=getattr(self.config, 'whisper_chunk_overlap', 3.0),
                model_name=getattr(self.config, 'whisper_model_name', 'cahya/whisper-medium-id')
            )
            
            # Create DoclingDocument from our custom result
            doc_data = result["document_data"]
            logger.info(f"🔍 Document data keys: {list(doc_data.keys())}")
            logger.info(f"🔍 Origin data: {doc_data.get('origin', {})}")
            
            try:
                document = DoclingDocument.model_validate(doc_data)
                logger.info(f"✅ DoclingDocument created successfully")
            except Exception as e:
                logger.error(f"❌ DoclingDocument validation failed: {e}")
                raise
            
            logger.info(f"✅ Indonesian audio processed successfully: {result['transcription']['word_count']} words")
            return document
            
        except Exception as e:
            logger.warning(f"⚠️ Indonesian audio processing failed: {e}")
            logger.info("🔄 Falling back to standard Docling audio processing")
            raise
    
    def process_document(self, document_path: Union[str, Path]) -> DoclingDocument:
        if isinstance(document_path, str) and (document_path.startswith('http://') or document_path.startswith('https://')):
            result = self.converter.convert(document_path)
        else:
            document_path = Path(document_path)
            if not document_path.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")
            
            # Try custom Indonesian audio processing for audio files
            if self._is_audio_file(document_path):
                try:
                    return self._process_indonesian_audio(document_path)
                except Exception as e:
                    logger.error(f"❌ Custom Indonesian audio processing failed: {e}")
                    logger.error(f"📍 Error type: {type(e).__name__}")
                    logger.info("🔄 Falling back to standard Docling processing")
                    # Continue to standard processing below
            
            # Standard Docling processing for all other files or audio fallback
            result = self.converter.convert(document_path)
        
        if not result.document:
            raise ValueError(f"Failed to process document: {document_path}")
        return result.document
    
    def format_document(self, document: DoclingDocument, output_format: str = "json") -> Union[str, Dict[str, Any]]:
        format_map = {
            "markdown": document.export_to_markdown,
            "text": document.export_to_text,
            "html": document.export_to_html,
            "json": document.export_to_dict
        }
        
        formatter = format_map.get(output_format.lower())
        if not formatter:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return formatter()
    
    def process_and_format_document(self, document_path: Union[str, Path], output_format: str = "json") -> Union[str, Dict[str, Any]]:
        document = self.process_document(document_path)
        return self.format_document(document, output_format)
    
    def process_batch(self, document_paths: List[Union[str, Path]]) -> List[DoclingDocument]:
        results = []
        for doc_path in document_paths:
            document = self.process_document(doc_path)
            results.append(document)
        return results
    
    def process_and_format_batch(self, document_paths: List[Union[str, Path]], output_format: str = "json") -> List[Union[str, Dict[str, Any]]]:
        results = []
        for doc_path in document_paths:
            try:
                formatted_content = self.process_and_format_document(doc_path, output_format)
                results.append(formatted_content)
            except Exception as e:
                results.append({
                    "error": f"Failed to process {doc_path}: {str(e)}",
                    "filename": str(doc_path)
                })
        return results
    
    def chunk_document(self, document: DoclingDocument, chunk_size: Optional[int] = None, overlap: Optional[int] = None, tokenizer_name: Optional[str] = None) -> List[Dict[str, Any]]:
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap
        tokenizer_path = tokenizer_name or str(self.config.tokenizer_model_dir)
        
        from transformers import AutoTokenizer
        actual_model_path = Path(tokenizer_path) / "snapshots" / "f752c1ee2994831dcef5b1e446383bc1e1996d52"
        hf_tokenizer = AutoTokenizer.from_pretrained(str(actual_model_path))
        
        tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=chunk_size)
        chunker = HybridChunker(tokenizer=tokenizer, merge_peers=self.config.merge_peers)
        chunks = chunker.chunk(document)
        
        chunk_list = []
        for i, chunk in enumerate(chunks):
            try:
                tokens = hf_tokenizer.encode(chunk.text)
                num_tokens = len(tokens)
            except Exception:
                num_tokens = None
            
            page_numbers = []
            chapter = None
            section = None
            chunk_type = 'text'
            
            if hasattr(chunk, 'meta'):
                meta = chunk.meta
                
                if hasattr(meta, 'doc_items') and meta.doc_items:
                    page_nums = set()
                    for item in meta.doc_items:
                        if hasattr(item, 'prov') and item.prov:
                            prov_item = item.prov[0]
                            if hasattr(prov_item, 'page_no'):
                                page_nums.add(prov_item.page_no)
                    page_numbers = sorted(list(page_nums)) if page_nums else [1]
                
                if hasattr(meta, 'headings') and meta.headings:
                    headings = meta.headings
                    if len(headings) > 0:
                        chapter = headings[0]
                    if len(headings) > 1:
                        section = headings[1]
                    elif len(headings) == 1:
                        heading = headings[0]
                        section_patterns = ['section', 'chapter', '1.', '2.', '3.', 'part']
                        if any(pattern in heading.lower() for pattern in section_patterns):
                            section = heading
                        else:
                            chapter = heading
                
                if hasattr(meta, 'captions') and meta.captions and not section:
                    section = meta.captions[0]
                
                if hasattr(meta, 'doc_items') and meta.doc_items:
                    for item in meta.doc_items:
                        if hasattr(item, 'label'):
                            chunk_type = item.label.name if hasattr(item.label, 'name') else str(item.label)
                            break
            
            chunk_dict = {
                "chunk_id": i,
                "text": chunk.text,
                "num_tokens": num_tokens,
                "metadata": {
                    "page_numbers": page_numbers,
                    "chapter": chapter,
                    "section": section,
                    "chunk_type": chunk_type,
                    "char_count": len(chunk.text)
                }
            }
            chunk_list.append(chunk_dict)
        
        return chunk_list
    
    def process_and_chunk_document(self, document_path: Union[str, Path], chunk_size: Optional[int] = None, overlap: Optional[int] = None, tokenizer_name: Optional[str] = None) -> List[Dict[str, Any]]:
        document = self.process_document(document_path)
        return self.chunk_document(document, chunk_size, overlap, tokenizer_name)
    
    def chunk_batch(self, documents: List[DoclingDocument], chunk_size: Optional[int] = None, overlap: Optional[int] = None, tokenizer_name: Optional[str] = None) -> List[List[Dict[str, Any]]]:
        results = []
        for document in documents:
            try:
                chunks = self.chunk_document(document, chunk_size, overlap, tokenizer_name)
                results.append(chunks)
            except Exception:
                results.append([])
        return results
    
    def get_supported_formats(self) -> List[str]:
        return [format.name for format in self.converter.allowed_formats]
    
    def get_model_info(self) -> Dict[str, Any]:
        model_info = {
            "models_directory": str(self.config.models_dir),
            "easyocr_models": str(self.config.easyocr_models_dir),
            "code_formula_model": str(self.config.code_formula_model_dir),
            "figure_classifier_model": str(self.config.figure_classifier_model_dir),
            "layout_model": str(self.config.layout_model_dir),
            "tokenizer_model": str(self.config.tokenizer_model_dir),
            "device": self.config.device,
            "ocr_languages": ["id", "en"],
            "ocr_engine": self.config.default_ocr_engine,
            "force_full_page_ocr": self.config.force_full_page_ocr,
            "chunking": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "merge_peers": self.config.merge_peers,
                "tokenizer_model": str(self.config.tokenizer_model_dir)
            }
        }
        
        # Add ASR/Whisper model information if available
        if hasattr(self.config, 'whisper_model_dir'):
            model_info["asr"] = {
                "standard_whisper_model": "OpenAI small/tiny (fallback)",
                "custom_indonesian_model": str(self.config.whisper_model_dir),
                "indonesian_model_exists": self.config.whisper_model_dir.exists(),
                "asr_language": self.config.asr_language,
                "processing_method": "Custom Indonesian Whisper (cahya/whisper-small-id) with fallback to OpenAI models",
                "supported_audio_formats": [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".opus"]
            }
        
        return model_info
    
    def cleanup(self):
        """Clean up pipeline resources to prevent memory leaks"""
        try:
            # Clear converter resources
            if hasattr(self, 'converter'):
                logger.info("Cleaning up document converter...")
                del self.converter
            
            # Clear any cached models/tokenizers
            if hasattr(self, '_cached_tokenizer'):
                del self._cached_tokenizer
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if using GPU
            if self.config.device.lower() in ['cuda', 'mps']:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("CUDA cache cleared")
                except ImportError:
                    pass
            
            logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during pipeline cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion"""
        try:
            self.cleanup()
        except:
            pass
    
