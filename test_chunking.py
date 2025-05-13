import os
import time
import logging
from utils.document_processor import DocumentProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_section_retrieval(processor, query, top_n=3):
    """Test section-based retrieval."""
    logger.info(f"Testing section-based retrieval for query: '{query}'")
    start_time = time.time()
    sections = processor.get_relevant_sections_embedding(query, top_n=top_n)
    elapsed = time.time() - start_time
    logger.info(f"Section retrieval took {elapsed:.4f} seconds")
    logger.info(f"Retrieved {len(sections)} sections: {sections}")
    return sections

def test_chunk_retrieval(processor, query, top_n=5):
    """Test chunk-based retrieval."""
    logger.info(f"Testing chunk-based retrieval for query: '{query}'")
    start_time = time.time()
    context = processor.get_chunk_context(query, top_n=top_n)
    elapsed = time.time() - start_time
    logger.info(f"Chunk retrieval took {elapsed:.4f} seconds")

    # Count chunks by looking for section headers
    chunk_count = context.count("===")
    logger.info(f"Retrieved context with approximately {chunk_count} chunks")

    # Print a preview of the context
    preview = context[:500] + "..." if len(context) > 500 else context
    logger.info(f"Context preview:\n{preview}")

    return context

def main():
    print("Starting chunking test...")
    # Use the Navigating Zoning document as it's known to work
    doc_path = "data/NYC Real Estate Development Curriculum - Navigating Land Use, Zoning, and Development Planning in New York City.docx"
    module_name = "Navigating Zoning, Land Use, and Development Planning"
    print(f"Using document: {doc_path}")

    # Test queries
    queries = [
        "What is FAR in zoning?",
        "How do I get a zoning variance?",
        "What are the steps in the ULURP process?",
        "What is the difference between as-of-right and discretionary approvals?"
    ]

    # First run - section-based retrieval
    logger.info("=== TESTING SECTION-BASED RETRIEVAL ===")
    processor = DocumentProcessor(doc_path, module_name)
    processor.load_document()
    processor.process_document()
    processor.build_embedding_index(use_chunks=False)

    for query in queries:
        test_section_retrieval(processor, query)

    # Second run - chunk-based retrieval
    logger.info("\n=== TESTING CHUNK-BASED RETRIEVAL ===")
    chunk_processor = DocumentProcessor(doc_path, module_name)
    chunk_processor.load_document()
    chunk_processor.process_document()
    chunk_processor.build_embedding_index(use_chunks=True, chunk_size=300, chunk_overlap=50)

    for query in queries:
        test_chunk_retrieval(chunk_processor, query)

    # Compare results for a specific query
    logger.info("\n=== COMPARING RETRIEVAL METHODS ===")
    comparison_query = "What is FAR in zoning?"

    logger.info(f"Query: '{comparison_query}'")
    sections = test_section_retrieval(processor, comparison_query)
    chunk_context = test_chunk_retrieval(chunk_processor, comparison_query)

    # Check if the chunk context contains the key sections
    for section in sections:
        if section in chunk_context:
            logger.info(f"Section '{section}' found in chunk context")
        else:
            logger.info(f"Section '{section}' NOT found in chunk context")

if __name__ == "__main__":
    main()
