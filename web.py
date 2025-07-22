from fastapi import HTTPException
from typing import Dict, Any, List
import requests
from urllib.parse import urlparse, urljoin, urlunparse
from bs4 import BeautifulSoup
import time
from logging_config import get_processing_logger
from pipeline import DocumentIntelligencePipeline
from utils import get_current_timestamp, handle_processing_error, validate_url

logger = get_processing_logger(__name__)

try:
    pipeline = DocumentIntelligencePipeline()
except Exception:
    pipeline = None

async def process_document_url(url: str, output_format: str = "json", processing_mode: str = "full") -> Dict[str, Any]:
    if not pipeline:
        raise HTTPException(status_code=500, detail="Document processing pipeline not available")
    
    if not _is_valid_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL format")
    
    content_type, content_length = _get_url_metadata(url)
    
    if processing_mode == "chunks_only":
        chunks = pipeline.process_and_chunk_document(url)
        total_text = " ".join([chunk.get("text", "") for chunk in chunks])
        word_count = len(total_text.split()) if total_text else 0
        char_count = len(total_text) if total_text else 0
            
        processed_data = {
            "chunks": chunks,
            "total_chunks": len(chunks),
            "word_count": word_count,
            "char_count": char_count,
            "extraction_method": "docling_url_chunks_only",
            "processing_mode": processing_mode
        }
    elif processing_mode == "both":
        chunks = pipeline.process_and_chunk_document(url)
        formatted_content = pipeline.process_and_format_document(url, output_format)
        
        if output_format == "text":
            text_content = formatted_content
            word_count = len(text_content.split()) if isinstance(text_content, str) else 0
            char_count = len(text_content) if isinstance(text_content, str) else 0
        else:
            try:
                text_content = pipeline.process_and_format_document(url, "text")
                word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                char_count = len(text_content) if isinstance(text_content, str) else 0
            except:
                word_count = 0
                char_count = 0
            
        processed_data = {
            "content": formatted_content,
            "formatted_content": formatted_content,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "word_count": word_count,
            "char_count": char_count,
            "extraction_method": "docling_url_full_and_chunks",
            "processing_mode": processing_mode
        }
    else:
        formatted_content = pipeline.process_and_format_document(url, output_format)
        
        if output_format == "text":
            text_content = formatted_content
            word_count = len(text_content.split()) if isinstance(text_content, str) else 0
            char_count = len(text_content) if isinstance(text_content, str) else 0
        else:
            try:
                text_content = pipeline.process_and_format_document(url, "text")
                word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                char_count = len(text_content) if isinstance(text_content, str) else 0
            except:
                word_count = 0
                char_count = 0
            
        processed_data = {
            "content": formatted_content,
            "word_count": word_count,
            "char_count": char_count,
            "extraction_method": "docling_url_full",
            "processing_mode": processing_mode
        }
    
    return {
        "url": url,
        "content_type": content_type,
        "content_length": content_length,
        "output_format": output_format,
        "processing_mode": processing_mode,
        "processed_content": processed_data,
        "status": "success",
        "timestamp": time.time()
    }

def _is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def _get_url_metadata(url: str) -> tuple[str, int]:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.head(url, headers=headers, timeout=30, allow_redirects=True)
        content_type = response.headers.get('content-type', 'unknown').lower()
        content_length = int(response.headers.get('content-length', 0))
        return content_type, content_length
    except Exception:
        return 'unknown', 0

async def process_document_urls_batch(urls: List[str], output_format: str = "json", processing_mode: str = "full") -> Dict[str, Any]:
    """
    Process multiple URLs using Docling batch pipeline.
    
    Args:
        urls: List of URLs to process
        output_format: Format for output ("json", "markdown", "text", "html")
        processing_mode: Processing mode ("full", "chunks_only", "both")
        
    Returns:
        Dictionary containing batch processing results
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Document processing pipeline not available")
    
    try:
        # Validate all URLs first
        valid_urls = []
        url_metadata = []
        
        for url in urls:
            if not _is_valid_url(url):
                raise HTTPException(status_code=400, detail=f"Invalid URL format: {url}")
            
            content_type, content_length = _get_url_metadata(url)
            valid_urls.append(url)
            url_metadata.append({
                "url": url,
                "content_type": content_type,
                "content_length": content_length
            })
        
        # Process URLs using batch pipeline based on processing mode
        if processing_mode == "chunks_only":
            # Only process and chunk URLs, skip formatted content
            documents = pipeline.process_batch(valid_urls)
            all_chunks = pipeline.chunk_batch(documents)
            
            results = []
            for i, (metadata, chunks) in enumerate(zip(url_metadata, all_chunks)):
                # Get statistics from chunks
                total_text = " ".join([chunk.get("text", "") for chunk in chunks])
                word_count = len(total_text.split()) if total_text else 0
                char_count = len(total_text) if total_text else 0
                
                result = {
                    "url": metadata["url"],
                    "content_type": metadata["content_type"],
                    "content_length": metadata["content_length"],
                    "output_format": output_format,
                    "processing_mode": processing_mode,
                    "processed_content": {
                        "chunks": chunks,
                        "total_chunks": len(chunks),
                        "word_count": word_count,
                        "char_count": char_count,
                        "extraction_method": "docling_batch_urls_chunks_only",
                        "processing_mode": processing_mode
                    },
                    "status": "success",
                    "timestamp": time.time()
                }
                results.append(result)
                
        elif processing_mode == "both":
            # Process URLs fully AND chunk them
            batch_results = pipeline.process_and_format_batch(valid_urls, output_format)
            documents = pipeline.process_batch(valid_urls)
            all_chunks = pipeline.chunk_batch(documents)
            
            results = []
            for i, (metadata, formatted_content, chunks) in enumerate(zip(url_metadata, batch_results, all_chunks)):
                # Get statistics from formatted content
                if output_format == "text":
                    text_content = formatted_content
                    word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                    char_count = len(text_content) if isinstance(text_content, str) else 0
                else:
                    try:
                        text_content = pipeline.process_and_format_document(valid_urls[i], "text")
                        word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                        char_count = len(text_content) if isinstance(text_content, str) else 0
                    except:
                        word_count = 0
                        char_count = 0
                
                result = {
                    "url": metadata["url"],
                    "content_type": metadata["content_type"],
                    "content_length": metadata["content_length"],
                    "output_format": output_format,
                    "processing_mode": processing_mode,
                    "processed_content": {
                        "content": formatted_content,
                        "formatted_content": formatted_content,
                        "chunks": chunks,
                        "total_chunks": len(chunks),
                        "word_count": word_count,
                        "char_count": char_count,
                        "extraction_method": "docling_batch_urls_full_and_chunks",
                        "processing_mode": processing_mode
                    },
                    "status": "success",
                    "timestamp": time.time()
                }
                results.append(result)
                
        else:  # processing_mode == "full"
            # Just process and format all URLs
            batch_results = pipeline.process_and_format_batch(valid_urls, output_format)
            
            results = []
            for i, (metadata, formatted_content) in enumerate(zip(url_metadata, batch_results)):
                # Get basic statistics
                if output_format == "text":
                    text_content = formatted_content
                    word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                    char_count = len(text_content) if isinstance(text_content, str) else 0
                else:
                    try:
                        text_content = pipeline.process_and_format_document(valid_urls[i], "text")
                        word_count = len(text_content.split()) if isinstance(text_content, str) else 0
                        char_count = len(text_content) if isinstance(text_content, str) else 0
                    except:
                        word_count = 0
                        char_count = 0
                
                result = {
                    "url": metadata["url"],
                    "content_type": metadata["content_type"],
                    "content_length": metadata["content_length"],
                    "output_format": output_format,
                    "processing_mode": processing_mode,
                    "processed_content": {
                        "content": formatted_content,
                        "word_count": word_count,
                        "char_count": char_count,
                        "extraction_method": "docling_batch_urls_full",
                        "processing_mode": processing_mode
                    },
                    "status": "success",
                    "timestamp": time.time()
                }
                results.append(result)
        
        # Return batch results
        return {
            "batch_results": results,
            "total_urls": len(urls),
            "successful_urls": len([r for r in results if r["status"] == "success"]),
            "output_format": output_format,
            "processing_mode": processing_mode,
            "processing_method": "batch_urls",
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing URL batch: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing URL batch: {str(e)}")

async def crawl_website(
    base_url: str, 
    max_depth: int = 2, 
    same_domain_only: bool = True,
    output_format: str = "json",
    processing_mode: str = "full",
    max_pages: int = 50
) -> Dict[str, Any]:
    """
    Crawl a website with configurable depth using Docling pipeline.
    
    Args:
        base_url: Starting URL for crawling
        max_depth: Maximum crawl depth (1-5)
        same_domain_only: Whether to stay within the same domain
        output_format: Format for output ("json", "markdown", "text", "html")
        processing_mode: Processing mode ("full", "chunks_only", "both")
        max_pages: Maximum number of pages to crawl
        
    Returns:
        Dictionary containing crawl results
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Document processing pipeline not available")
    
    if max_depth < 1 or max_depth > 5:
        raise HTTPException(status_code=400, detail="Max depth must be between 1 and 5")
    
    if max_pages > 100:
        raise HTTPException(status_code=400, detail="Max pages cannot exceed 100")
    
    try:
        # Validate base URL
        if not _is_valid_url(base_url):
            raise HTTPException(status_code=400, detail="Invalid base URL format")
        
        # Extract base domain for filtering
        base_domain = urlparse(base_url).netloc if same_domain_only else None
        
        # Crawl and collect URLs
        discovered_urls = await _crawl_urls(base_url, max_depth, base_domain, max_pages)
        
        if not discovered_urls:
            raise HTTPException(status_code=404, detail="No accessible URLs found during crawl")
        
        # Process all discovered URLs using batch processing
        crawl_results = await process_document_urls_batch(discovered_urls, output_format, processing_mode)
        
        # Add crawl-specific metadata
        result = {
            "crawl_info": {
                "base_url": base_url,
                "max_depth": max_depth,
                "same_domain_only": same_domain_only,
                "max_pages": max_pages,
                "discovered_urls": discovered_urls,
                "total_discovered": len(discovered_urls)
            },
            "processing_results": crawl_results,
            "status": "success",
            "timestamp": time.time()
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error crawling website {base_url}: {e}")
        raise HTTPException(status_code=500, detail=f"Error crawling website: {str(e)}")

async def _crawl_urls(base_url: str, max_depth: int, base_domain: str = None, max_pages: int = 50) -> List[str]:
    """
    Crawl URLs recursively up to max_depth.
    
    Args:
        base_url: Starting URL
        max_depth: Maximum depth to crawl
        base_domain: Domain to restrict crawling to (if any)
        max_pages: Maximum number of pages to discover
        
    Returns:
        List of discovered URLs
    """
    discovered = set()
    to_visit = [(base_url, 0)]  # (url, depth)
    visited = set()
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    while to_visit and len(discovered) < max_pages:
        current_url, depth = to_visit.pop(0)
        
        if current_url in visited or depth > max_depth:
            continue
            
        visited.add(current_url)
        
        try:
            # Add current URL to discovered list
            discovered.add(current_url)
            
            # If we've reached max depth, don't extract more links
            if depth >= max_depth:
                continue
            
            # Fetch page to extract links
            response = requests.get(current_url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            # Only extract links from HTML pages
            if 'text/html' in response.headers.get('content-type', '').lower():
                links = _extract_links_from_page(response.text, current_url, base_domain)
                
                # Add new links to visit queue
                for link in links:
                    if link not in visited and len(discovered) < max_pages:
                        to_visit.append((link, depth + 1))
            
            # Add delay to be respectful
            time.sleep(1)
            
        except Exception as e:
            logger.warning(f"Error crawling {current_url}: {e}")
            continue
    
    return list(discovered)

def _extract_links_from_page(html_content: str, base_url: str, base_domain: str = None) -> List[str]:
    """
    Extract internal links from HTML page.
    
    Args:
        html_content: HTML content
        base_url: Base URL for resolving relative links
        base_domain: Domain to filter links (if any)
        
    Returns:
        List of valid internal links
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Resolve relative URLs
            full_url = urljoin(base_url, href)
            
            # Validate URL
            if not _is_valid_url(full_url):
                continue
            
            # Filter by domain if specified
            if base_domain and urlparse(full_url).netloc != base_domain:
                continue
            
            # Skip fragments and parameters for cleaner crawling
            parsed = urlparse(full_url)
            clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
            
            # Skip common non-content URLs
            if any(skip in clean_url.lower() for skip in ['/login', '/logout', '/register', '/search', '/contact', '#']):
                continue
            
            links.append(clean_url)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        
        return unique_links[:20]  # Limit links per page
        
    except Exception as e:
        logger.warning(f"Error extracting links from {base_url}: {e}")
        return []