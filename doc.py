from fastapi import UploadFile, HTTPException
from typing import Dict, Any, List
import os
import tempfile
import time
from pathlib import Path
from logging_config import get_processing_logger
from pipeline import DocumentIntelligencePipeline
from utils import (
    get_current_timestamp, 
    handle_processing_error, 
    validate_file_size, 
    validate_file_format,
    is_supported_document_format
)
from processing_utils import (
    get_pipeline,
    process_single_item,
    create_processing_result,
    create_error_result,
    extract_processing_stats,
    cleanup_temp_file,
    validate_processing_mode
)

logger = get_processing_logger(__name__)

try:
    pipeline = DocumentIntelligencePipeline()
    SUPPORTED_FORMATS = pipeline.get_supported_formats()
except Exception:
    pipeline = None
    SUPPORTED_FORMATS = []

# Map file extensions to Docling format types
EXTENSION_TO_FORMAT = {
    # Document formats
    'DOCX': 'DOCX',
    'PPTX': 'PPTX', 
    'XLSX': 'XLSX',
    'PDF': 'PDF',
    'HTML': 'HTML',
    'HTM': 'HTML',
    'MD': 'MD',
    'TXT': 'TXT',
    'CSV': 'CSV',
    'ASCIIDOC': 'ASCIIDOC',
    'XML': 'XML_USPTO',  # Default XML mapping
    'JSON': 'JSON_DOCLING',
    
    # Image formats (grouped under IMAGE)
    'PNG': 'IMAGE',
    'JPG': 'IMAGE', 
    'JPEG': 'IMAGE',
    'GIF': 'IMAGE',
    'BMP': 'IMAGE',
    'TIFF': 'IMAGE',
    'TIF': 'IMAGE',
    'WEBP': 'IMAGE',
    
    # Audio formats (grouped under AUDIO)
    'MP3': 'AUDIO',
    'WAV': 'AUDIO',
    'M4A': 'AUDIO', 
    'AAC': 'AUDIO',
    'FLAC': 'AUDIO',
    'OGG': 'AUDIO',
    'WMA': 'AUDIO',
    'OPUS': 'AUDIO'
}

def is_supported_format(file_extension: str) -> bool:
    """Check if a file extension is supported by mapping it to Docling formats or direct processing"""
    ext_upper = file_extension.upper()
    
    # TXT files are supported through direct processing
    if ext_upper == "TXT":
        return True
    
    mapped_format = EXTENSION_TO_FORMAT.get(ext_upper)
    return mapped_format in SUPPORTED_FORMATS if mapped_format else False

def _is_originally_txt_file(file_path: str) -> bool:
    """Check if a .md file was originally a TXT file by examining its content structure"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(500)  # Read first 500 chars to catch more markdown syntax
            # Check for common markdown patterns
            markdown_patterns = [
                '#',      # Headers
                '**',     # Bold
                '*',      # Italic/Bold
                '_',      # Italic/Bold
                '[',      # Links
                ']',      # Links
                '`',      # Code
                '```',    # Code blocks
                '- ',     # Lists (dash followed by space)
                '* ',     # Lists (asterisk followed by space)
                '1. ',    # Numbered lists
            ]
            return not any(pattern in content for pattern in markdown_patterns)
    except:
        return False

def _process_txt_file_directly(file_path: str, output_format: str, processing_mode: str) -> Dict[str, Any]:
    """Process TXT files directly since Docling doesn't support them"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        # Format the content based on output format
        if output_format == "json":
            formatted_content = {
                "content": text_content,
                "format": "text",
                "metadata": {"source": "plain_text_file"}
            }
        elif output_format == "markdown":
            formatted_content = f"```text\n{text_content}\n```"
        elif output_format == "html":
            formatted_content = f"<pre>{text_content}</pre>"
        else:  # text
            formatted_content = text_content
        
        word_count = len(text_content.split()) if text_content else 0
        char_count = len(text_content) if text_content else 0
        
        if processing_mode == "chunks_only":
            # Create simple chunks for text files
            chunks = _create_text_chunks(text_content)
            return {
                "chunks": chunks,
                "total_chunks": len(chunks),
                "word_count": word_count,
                "char_count": char_count,
                "extraction_method": "direct_text_chunks_only",
                "processing_mode": processing_mode
            }
        elif processing_mode == "both":
            chunks = _create_text_chunks(text_content)
            return {
                "content": formatted_content,
                "formatted_content": formatted_content,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "word_count": word_count,
                "char_count": char_count,
                "extraction_method": "direct_text_full_and_chunks",
                "processing_mode": processing_mode
            }
        else:  # full
            return {
                "content": formatted_content,
                "word_count": word_count,
                "char_count": char_count,
                "extraction_method": "direct_text_full",
                "processing_mode": processing_mode
            }
    except Exception as e:
        logger.error(f"Error processing TXT file directly: {e}")
        raise ValueError(f"Failed to process TXT file: {str(e)}")

def _create_text_chunks(text_content: str, chunk_size: int = 200) -> List[Dict[str, Any]]:
    """Create simple text chunks for plain text files"""
    words = text_content.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "text": chunk_text,
            "chunk_id": i // chunk_size,
            "word_count": len(chunk_words),
            "char_count": len(chunk_text)
        })
    
    return chunks

async def process_document_file(file: UploadFile, output_format: str = "json", processing_mode: str = "full") -> Dict[str, Any]:
    if not pipeline:
        raise HTTPException(status_code=500, detail="Document processing pipeline not available")
    
    try:
        # Validate file extension against Docling supported formats
        file_extension = os.path.splitext(file.filename)[1].upper().lstrip('.')

        if not is_supported_format(file_extension):
            supported_extensions = ', '.join(sorted(EXTENSION_TO_FORMAT.keys()))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_extension}. Supported extensions: {supported_extensions}"
            )
        
        # Read file content
        file_content = await file.read()

        # Save the uploaded file to a temp file with correct (or altered) extension
        if file_extension == "TXT":
            suffix = ".md"
        else:
            suffix = os.path.splitext(file.filename)[1]
        
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
            
        try:
            # Process document using Docling pipeline
            processed_data = _process_with_docling(temp_file_path, output_format, processing_mode)
            
            # Add metadata
            result = {
                "filename": file.filename,
                "file_extension": file_extension,
                "file_size": len(file_content),
                "output_format": output_format,
                "processing_mode": processing_mode,
                "processed_content": processed_data,
                "status": "success"
            }
            
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

def _process_with_docling(file_path: str, output_format: str = "json", processing_mode: str = "full") -> Dict[str, Any]:
    """
    Process document using Docling pipeline with different processing modes.
    Handle TXT files separately since Docling doesn't support them.
    
    Args:
        file_path: Path to the file
        output_format: Format for output ("json", "markdown", "text", "html")
        processing_mode: Processing mode ("full", "chunks_only", "both")
        
    Returns:
        Processed document data
    """
    try:
        # Check if this is a TXT file that we need to handle separately
        if file_path.lower().endswith('.md') and _is_originally_txt_file(file_path):
            return _process_txt_file_directly(file_path, output_format, processing_mode)
        if processing_mode == "chunks_only":
            # Only process and chunk the document, skip formatted content
            chunks = pipeline.process_and_chunk_document(file_path)
            
            # Get statistics from chunks
            total_text = " ".join([chunk.get("text", "") for chunk in chunks])
            word_count = len(total_text.split()) if total_text else 0
            char_count = len(total_text) if total_text else 0
            
            return {
                "chunks": chunks,
                "total_chunks": len(chunks),
                "word_count": word_count,
                "char_count": char_count,
                "extraction_method": "docling_chunks_only",
                "processing_mode": processing_mode
            }
            
        elif processing_mode == "both":
            # Process document fully AND chunk it
            chunks = pipeline.process_and_chunk_document(file_path)
            formatted_content = pipeline.process_and_format_document(file_path, output_format)
            
            # Get statistics from formatted content
            if output_format == "text":
                text_content = formatted_content
                word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                char_count = len(text_content) if isinstance(text_content, str) else 0
            else:
                try:
                    text_content = pipeline.process_and_format_document(file_path, "text")
                    word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                    char_count = len(text_content) if isinstance(text_content, str) else 0
                except:
                    word_count = 0
                    char_count = 0
            
            return {
                "content": formatted_content,
                "formatted_content": formatted_content,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "word_count": word_count,
                "char_count": char_count,
                "extraction_method": "docling_full_and_chunks",
                "processing_mode": processing_mode
            }
            
        else:  # processing_mode == "full"
            # Just process and format the document (default behavior)
            formatted_content = pipeline.process_and_format_document(file_path, output_format)
            
            # Get basic statistics
            if output_format == "text":
                text_content = formatted_content
                word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                char_count = len(text_content) if isinstance(text_content, str) else 0
            else:
                # For other formats, try to extract text for statistics
                try:
                    text_content = pipeline.process_and_format_document(file_path, "text")
                    word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                    char_count = len(text_content) if isinstance(text_content, str) else 0
                except:
                    word_count = 0
                    char_count = 0
            
            return {
                "content": formatted_content,
                "word_count": word_count,
                "char_count": char_count,
                "extraction_method": "docling_full",
                "processing_mode": processing_mode
            }
            
    except Exception as e:
        logger.error(f"Error in Docling processing: {e}")
        raise ValueError(f"Failed to process document with Docling: {str(e)}")

async def process_document_files_batch(files: List[UploadFile], output_format: str = "json", processing_mode: str = "full") -> Dict[str, Any]:
    """
    Process multiple uploaded document files using Docling batch pipeline.
    
    Args:
        files: List of uploaded files
        output_format: Format for output ("json", "markdown", "text", "html")
        processing_mode: Processing mode ("full", "chunks_only", "both")
        
    Returns:
        Dictionary containing batch processing results
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Document processing pipeline not available")
    
    try:
        temp_files = []
        file_info = []
        
        # Create temporary files for all uploads
        for file in files:
            filename = file.filename or "unknown"
            file_extension = os.path.splitext(filename)[1].upper().lstrip('.')
            
            if not is_supported_format(file_extension):
                supported_extensions = ', '.join(sorted(EXTENSION_TO_FORMAT.keys()))
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {file_extension} in file {filename}. Supported extensions: {supported_extensions}"
                )
            
            file_content = await file.read()
            
            # Apply same TXT -> MD suffix conversion as single file processing
            if file_extension == "TXT":
                suffix = ".md"
            else:
                suffix = os.path.splitext(filename)[1]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(file_content)
                temp_files.append(temp_file.name)
                file_info.append({
                    "filename": filename,
                    "file_extension": file_extension,
                    "file_size": len(file_content),
                    "temp_path": temp_file.name
                })
        
        try:
            # Process documents using batch pipeline based on processing mode
            if processing_mode == "chunks_only":
                # Only process and chunk documents, skip formatted content
                documents = pipeline.process_batch([Path(temp_path) for temp_path in temp_files])
                all_chunks = pipeline.chunk_batch(documents)
                
                results = []
                for i, (info, chunks) in enumerate(zip(file_info, all_chunks)):
                    # Get statistics from chunks
                    total_text = " ".join([chunk.get("text", "") for chunk in chunks])
                    word_count = len(total_text.split()) if total_text else 0
                    char_count = len(total_text) if total_text else 0
                    
                    result = {
                        "filename": info["filename"],
                        "file_extension": info["file_extension"],
                        "file_size": info["file_size"],
                        "output_format": output_format,
                        "processing_mode": processing_mode,
                        "processed_content": {
                            "chunks": chunks,
                            "total_chunks": len(chunks),
                            "word_count": word_count,
                            "char_count": char_count,
                            "extraction_method": "docling_batch_chunks_only",
                            "processing_mode": processing_mode
                        },
                        "status": "success"
                    }
                    results.append(result)
                    
            elif processing_mode == "both":
                # Process documents fully AND chunk them
                batch_results = pipeline.process_and_format_batch([Path(temp_path) for temp_path in temp_files], output_format)
                documents = pipeline.process_batch([Path(temp_path) for temp_path in temp_files])
                all_chunks = pipeline.chunk_batch(documents)
                
                results = []
                for i, (info, formatted_content, chunks) in enumerate(zip(file_info, batch_results, all_chunks)):
                    # Get statistics from formatted content
                    if output_format == "text":
                        text_content = formatted_content
                        word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                        char_count = len(text_content) if isinstance(text_content, str) else 0
                    else:
                        try:
                            text_content = pipeline.process_and_format_document(Path(temp_files[i]), "text")
                            word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                            char_count = len(text_content) if isinstance(text_content, str) else 0
                        except:
                            word_count = 0
                            char_count = 0
                    
                    result = {
                        "filename": info["filename"],
                        "file_extension": info["file_extension"],
                        "file_size": info["file_size"],
                        "output_format": output_format,
                        "processing_mode": processing_mode,
                        "processed_content": {
                            "content": formatted_content,
                            "formatted_content": formatted_content,
                            "chunks": chunks,
                            "total_chunks": len(chunks),
                            "word_count": word_count,
                            "char_count": char_count,
                            "extraction_method": "docling_batch_full_and_chunks",
                            "processing_mode": processing_mode
                        },
                        "status": "success"
                    }
                    results.append(result)
                    
            else:  # processing_mode == "full"
                # Just process and format all documents
                batch_results = pipeline.process_and_format_batch([Path(temp_path) for temp_path in temp_files], output_format)
                
                results = []
                for i, (info, formatted_content) in enumerate(zip(file_info, batch_results)):
                    # Get basic statistics
                    if output_format == "text":
                        text_content = formatted_content
                        word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                        char_count = len(text_content) if isinstance(text_content, str) else 0
                    else:
                        try:
                            text_content = pipeline.process_and_format_document(Path(temp_files[i]), "text")
                            word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                            char_count = len(text_content) if isinstance(text_content, str) else 0
                        except:
                            word_count = 0
                            char_count = 0
                    
                    result = {
                        "filename": info["filename"],
                        "file_extension": info["file_extension"],
                        "file_size": info["file_size"],
                        "output_format": output_format,
                        "processing_mode": processing_mode,
                        "processed_content": {
                            "content": formatted_content,
                            "word_count": word_count,
                            "char_count": char_count,
                            "extraction_method": "docling_batch_full",
                            "processing_mode": processing_mode
                        },
                        "status": "success"
                    }
                    results.append(result)
            
            # Return batch results
            return {
                "batch_results": results,
                "total_files": len(files),
                "successful_files": len([r for r in results if r["status"] == "success"]),
                "output_format": output_format,
                "processing_mode": processing_mode,
                "processing_method": "batch",
                "status": "success"
            }
            
        finally:
            # Clean up temporary files
            for temp_path in temp_files:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document batch: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document batch: {str(e)}")

def process_document_file_from_bytes(file_data: bytes, filename: str, output_format: str = "json", processing_mode: str = "full") -> Dict[str, Any]:
    """
    Process document file from bytes data (used by Celery tasks).
    
    Args:
        file_data: File content as bytes
        filename: Original filename
        output_format: Format for output ("json", "markdown", "text", "html")
        processing_mode: Processing mode ("full", "chunks_only", "both")
        
    Returns:
        Dictionary containing processed document data
    """
    if not pipeline:
        raise ValueError("Document processing pipeline not available")
    
    try:
        # Validate file extension against Docling supported formats
        file_extension = os.path.splitext(filename)[1].upper().lstrip('.')
        
        if not is_supported_format(file_extension):
            supported_extensions = ', '.join(sorted(EXTENSION_TO_FORMAT.keys()))
            raise ValueError(
                f"Unsupported file format: {file_extension}. Supported extensions: {supported_extensions}"
            )
        
        # Apply same TXT -> MD suffix conversion as other functions
        if file_extension == "TXT":
            suffix = ".md"
        else:
            suffix = os.path.splitext(filename)[1]
        
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_data)
            temp_file_path = temp_file.name
        
        try:
            # Process document using Docling pipeline
            processed_data = _process_with_docling(temp_file_path, output_format, processing_mode)
            
            # Add metadata
            result = {
                "filename": filename,
                "file_extension": file_extension,
                "file_size": len(file_data),
                "output_format": output_format,
                "processing_mode": processing_mode,
                "processed_content": processed_data,
                "status": "success",
                "timestamp": get_current_timestamp()
            }
            
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error processing document {filename}: {e}")
        raise ValueError(f"Error processing document: {str(e)}")

def process_document_files_batch_from_data(files_data: List[Dict[str, Any]], output_format: str = "json", processing_mode: str = "full") -> Dict[str, Any]:
    """
    Process multiple document files from data (used by Celery tasks).
    
    Args:
        files_data: List of file data dictionaries with 'data' (bytes) and 'filename' (str)
        output_format: Format for output ("json", "markdown", "text", "html")
        processing_mode: Processing mode ("full", "chunks_only", "both")
        
    Returns:
        Dictionary containing batch processing results
    """
    if not pipeline:
        raise ValueError("Document processing pipeline not available")
    
    try:
        temp_files = []
        file_info = []
        
        # Create temporary files for all data
        for file_data_dict in files_data:
            file_data = file_data_dict['data']
            filename = file_data_dict['filename']
            file_extension = os.path.splitext(filename)[1].upper().lstrip('.')
            
            if not is_supported_format(file_extension):
                supported_extensions = ', '.join(sorted(EXTENSION_TO_FORMAT.keys()))
                raise ValueError(
                    f"Unsupported file format: {file_extension} in file {filename}. Supported extensions: {supported_extensions}"
                )
            
            # Apply same TXT -> MD suffix conversion as single file processing
            if file_extension == "TXT":
                suffix = ".md"
            else:
                suffix = os.path.splitext(filename)[1]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(file_data)
                temp_files.append(temp_file.name)
                file_info.append({
                    "filename": filename,
                    "file_extension": file_extension,
                    "file_size": len(file_data),
                    "temp_path": temp_file.name
                })
        
        try:
            # Process documents using batch pipeline based on processing mode
            if processing_mode == "chunks_only":
                # Only process and chunk documents, skip formatted content
                documents = pipeline.process_batch([Path(temp_path) for temp_path in temp_files])
                all_chunks = pipeline.chunk_batch(documents)
                
                results = []
                for i, (info, chunks) in enumerate(zip(file_info, all_chunks)):
                    # Get statistics from chunks
                    total_text = " ".join([chunk.get("text", "") for chunk in chunks])
                    word_count = len(total_text.split()) if total_text else 0
                    char_count = len(total_text) if total_text else 0
                    
                    result = {
                        "filename": info["filename"],
                        "file_extension": info["file_extension"],
                        "file_size": info["file_size"],
                        "output_format": output_format,
                        "processing_mode": processing_mode,
                        "processed_content": {
                            "chunks": chunks,
                            "total_chunks": len(chunks),
                            "word_count": word_count,
                            "char_count": char_count,
                            "extraction_method": "docling_batch_chunks_only",
                            "processing_mode": processing_mode
                        },
                        "status": "success",
                        "timestamp": get_current_timestamp()
                    }
                    results.append(result)
                    
            elif processing_mode == "both":
                # Process documents fully AND chunk them
                batch_results = pipeline.process_and_format_batch([Path(temp_path) for temp_path in temp_files], output_format)
                documents = pipeline.process_batch([Path(temp_path) for temp_path in temp_files])
                all_chunks = pipeline.chunk_batch(documents)
                
                results = []
                for i, (info, formatted_content, chunks) in enumerate(zip(file_info, batch_results, all_chunks)):
                    # Get statistics from formatted content
                    if output_format == "text":
                        text_content = formatted_content
                        word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                        char_count = len(text_content) if isinstance(text_content, str) else 0
                    else:
                        try:
                            text_content = pipeline.process_and_format_document(Path(temp_files[i]), "text")
                            word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                            char_count = len(text_content) if isinstance(text_content, str) else 0
                        except:
                            word_count = 0
                            char_count = 0
                    
                    result = {
                        "filename": info["filename"],
                        "file_extension": info["file_extension"],
                        "file_size": info["file_size"],
                        "output_format": output_format,
                        "processing_mode": processing_mode,
                        "processed_content": {
                            "content": formatted_content,
                            "formatted_content": formatted_content,
                            "chunks": chunks,
                            "total_chunks": len(chunks),
                            "word_count": word_count,
                            "char_count": char_count,
                            "extraction_method": "docling_batch_full_and_chunks",
                            "processing_mode": processing_mode
                        },
                        "status": "success",
                        "timestamp": get_current_timestamp()
                    }
                    results.append(result)
                    
            else:  # processing_mode == "full"
                # Just process and format all documents
                batch_results = pipeline.process_and_format_batch([Path(temp_path) for temp_path in temp_files], output_format)
                
                results = []
                for i, (info, formatted_content) in enumerate(zip(file_info, batch_results)):
                    # Get basic statistics
                    if output_format == "text":
                        text_content = formatted_content
                        word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                        char_count = len(text_content) if isinstance(text_content, str) else 0
                    else:
                        try:
                            text_content = pipeline.process_and_format_document(Path(temp_files[i]), "text")
                            word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                            char_count = len(text_content) if isinstance(text_content, str) else 0
                        except:
                            word_count = 0
                            char_count = 0
                    
                    result = {
                        "filename": info["filename"],
                        "file_extension": info["file_extension"],
                        "file_size": info["file_size"],
                        "output_format": output_format,
                        "processing_mode": processing_mode,
                        "processed_content": {
                            "content": formatted_content,
                            "word_count": word_count,
                            "char_count": char_count,
                            "extraction_method": "docling_batch_full",
                            "processing_mode": processing_mode
                        },
                        "status": "success",
                        "timestamp": get_current_timestamp()
                    }
                    results.append(result)
            
            # Return batch results
            return {
                "batch_results": results,
                "total_files": len(files_data),
                "successful_files": len([r for r in results if r["status"] == "success"]),
                "output_format": output_format,
                "processing_mode": processing_mode,
                "processing_method": "batch",
                "status": "success"
            }
            
        finally:
            # Clean up temporary files
            for temp_path in temp_files:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
    except Exception as e:
        logger.error(f"Error processing document batch: {e}")
        raise ValueError(f"Error processing document batch: {str(e)}")

def get_supported_formats() -> List[str]:
    """Get list of supported document formats from Docling."""
    if pipeline:
        return pipeline.get_supported_formats()
    return []
