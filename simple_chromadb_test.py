"""
Simple test script for ChromaDB integration in DevGuide AI.
"""

import logging
from utils.document_processor import DocumentProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test document path
DOC_PATH = "data/NYC Real Estate Development Curriculum - Navigating Land Use, Zoning, and Development Planning in New York City.docx"
MODULE_NAME = "Navigating Zoning, Land Use, and Development Planning"

# Test query
TEST_QUERY = "What is FAR?"

def main():
    """Run a simple test of ChromaDB integration."""
    logger.info("Starting simple ChromaDB test...")
    
    # Create processor with ChromaDB retrieval
    processor = DocumentProcessor(DOC_PATH, MODULE_NAME)
    processor.load_document()
    processor.process_document()
    
    # Build embeddings with chunks and ChromaDB
    logger.info("Building embeddings with ChromaDB...")
    processor.build_embedding_index(use_chunks=True, chunk_size=300, chunk_overlap=50, use_chroma=True)
    
    # Test query
    logger.info(f"Testing query: '{TEST_QUERY}'")
    sections = processor.get_relevant_sections_embedding(TEST_QUERY, top_n=3)
    
    logger.info(f"Retrieved sections: {sections}")
    
    # Get chunk context
    context = processor.get_chunk_context(TEST_QUERY, top_n=3)
    
    # Log the first 200 characters of the context
    preview = context[:200] + "..." if len(context) > 200 else context
    logger.info(f"Context preview: {preview}")
    
    logger.info("Simple ChromaDB test completed.")

if __name__ == "__main__":
    main()
