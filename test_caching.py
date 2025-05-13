"""
Test script to demonstrate document caching functionality.
This script loads a document twice to show the difference between
first-time processing and cached loading.
"""

import os
import time
import logging
from utils.document_processor import DocumentProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Use the Navigating Zoning document as it's known to work
    doc_path = "data/NYC Real Estate Development Curriculum - Navigating Land Use, Zoning, and Development Planning in New York City.docx"
    module_name = "Navigating Zoning, Land Use, and Development Planning"
    
    # First run - should process from scratch
    logger.info("=== FIRST RUN - SHOULD PROCESS FROM SCRATCH ===")
    start_time = time.time()
    processor = DocumentProcessor(doc_path, module_name)
    processor.load_document()
    processor.process_document()
    processor.build_embedding_index()
    first_run_time = time.time() - start_time
    logger.info(f"First run completed in {first_run_time:.2f} seconds")
    
    # Second run - should use cache
    logger.info("\n=== SECOND RUN - SHOULD USE CACHE ===")
    start_time = time.time()
    processor2 = DocumentProcessor(doc_path, module_name)
    processor2.load_document()
    processor2.process_document()
    processor2.build_embedding_index()
    second_run_time = time.time() - start_time
    logger.info(f"Second run completed in {second_run_time:.2f} seconds")
    
    # Calculate speedup
    if first_run_time > 0:
        speedup = first_run_time / second_run_time
        logger.info(f"Cache speedup: {speedup:.2f}x faster")
    
    # Check if cache files exist
    cache_dir = "cache"
    cache_files = os.listdir(cache_dir)
    logger.info(f"Cache files created: {cache_files}")

if __name__ == "__main__":
    main()
