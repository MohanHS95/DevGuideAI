"""
Test script for ChromaDB integration in DevGuide AI.

This script tests the ChromaDB integration by:
1. Loading a document
2. Processing it with ChromaDB enabled
3. Performing queries using ChromaDB
4. Comparing results with the original in-memory approach
"""

import os
import time
import logging
from utils.document_processor import DocumentProcessor
from utils.chroma_manager import ChromaManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test document path
DOC_PATH = "data/NYC Real Estate Development Curriculum - Navigating Land Use, Zoning, and Development Planning in New York City.docx"
MODULE_NAME = "Navigating Zoning, Land Use, and Development Planning"

# Test queries
TEST_QUERIES = [
    "What is FAR?",
    "Explain zoning regulations",
    "How do I get a building permit?",
    "What are the steps in the ULURP process?",
    "Explain the role of community boards"
]

def test_in_memory_retrieval():
    """Test the original in-memory retrieval approach."""
    logger.info("Testing in-memory retrieval...")
    
    # Create processor with in-memory retrieval
    processor = DocumentProcessor(DOC_PATH, MODULE_NAME)
    processor.load_document()
    processor.process_document()
    
    # Build embeddings with chunks but without ChromaDB
    start_time = time.time()
    processor.build_embedding_index(use_chunks=True, chunk_size=300, chunk_overlap=50, use_chroma=False)
    build_time = time.time() - start_time
    logger.info(f"Built in-memory embeddings in {build_time:.2f} seconds")
    
    # Test queries
    total_query_time = 0
    for query in TEST_QUERIES:
        # Time section retrieval
        start_time = time.time()
        sections = processor.get_relevant_sections_embedding(query, top_n=3)
        query_time = time.time() - start_time
        total_query_time += query_time
        
        logger.info(f"Query: '{query}'")
        logger.info(f"Retrieved sections: {sections}")
        logger.info(f"Query time: {query_time:.4f} seconds")
        logger.info("-" * 50)
    
    avg_query_time = total_query_time / len(TEST_QUERIES)
    logger.info(f"Average in-memory query time: {avg_query_time:.4f} seconds")
    return avg_query_time

def test_chromadb_retrieval():
    """Test the ChromaDB retrieval approach."""
    logger.info("Testing ChromaDB retrieval...")
    
    # Create processor with ChromaDB retrieval
    processor = DocumentProcessor(DOC_PATH, MODULE_NAME)
    processor.load_document()
    processor.process_document()
    
    # Build embeddings with chunks and ChromaDB
    start_time = time.time()
    processor.build_embedding_index(use_chunks=True, chunk_size=300, chunk_overlap=50, use_chroma=True)
    build_time = time.time() - start_time
    logger.info(f"Built ChromaDB embeddings in {build_time:.2f} seconds")
    
    # Test queries
    total_query_time = 0
    for query in TEST_QUERIES:
        # Time section retrieval
        start_time = time.time()
        sections = processor.get_relevant_sections_embedding(query, top_n=3)
        query_time = time.time() - start_time
        total_query_time += query_time
        
        logger.info(f"Query: '{query}'")
        logger.info(f"Retrieved sections: {sections}")
        logger.info(f"Query time: {query_time:.4f} seconds")
        logger.info("-" * 50)
    
    avg_query_time = total_query_time / len(TEST_QUERIES)
    logger.info(f"Average ChromaDB query time: {avg_query_time:.4f} seconds")
    return avg_query_time

def test_chunk_context_retrieval():
    """Test chunk context retrieval with ChromaDB."""
    logger.info("Testing chunk context retrieval with ChromaDB...")
    
    # Create processor with ChromaDB retrieval
    processor = DocumentProcessor(DOC_PATH, MODULE_NAME)
    processor.load_document()
    processor.process_document()
    processor.build_embedding_index(use_chunks=True, chunk_size=300, chunk_overlap=50, use_chroma=True)
    
    # Test queries
    for query in TEST_QUERIES:
        # Get chunk context
        context = processor.get_chunk_context(query, top_n=3)
        
        # Log the first 200 characters of the context
        preview = context[:200] + "..." if len(context) > 200 else context
        logger.info(f"Query: '{query}'")
        logger.info(f"Context preview: {preview}")
        logger.info("-" * 50)

def main():
    """Run all tests and compare results."""
    logger.info("Starting ChromaDB integration tests...")
    
    # Test in-memory retrieval
    in_memory_time = test_in_memory_retrieval()
    
    # Test ChromaDB retrieval
    chromadb_time = test_chromadb_retrieval()
    
    # Test chunk context retrieval
    test_chunk_context_retrieval()
    
    # Compare results
    logger.info("=" * 50)
    logger.info("Test Results Summary:")
    logger.info(f"In-memory average query time: {in_memory_time:.4f} seconds")
    logger.info(f"ChromaDB average query time: {chromadb_time:.4f} seconds")
    
    if chromadb_time < in_memory_time:
        improvement = (in_memory_time - chromadb_time) / in_memory_time * 100
        logger.info(f"ChromaDB is {improvement:.2f}% faster than in-memory retrieval")
    else:
        slowdown = (chromadb_time - in_memory_time) / in_memory_time * 100
        logger.info(f"ChromaDB is {slowdown:.2f}% slower than in-memory retrieval")
    
    logger.info("=" * 50)
    logger.info("ChromaDB integration tests completed.")

if __name__ == "__main__":
    main()
