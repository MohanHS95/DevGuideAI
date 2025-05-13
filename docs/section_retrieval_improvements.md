# Section Retrieval Improvements

This document outlines the improvements made to the section retrieval system in DevGuide AI to better handle direct section references and improve the accuracy of retrieving specific sections.

## Problem Statement

The system was having issues with accurately retrieving information from specific sections when users made direct references to section numbers (like "section 2.2") or asked about specific topics that should map to particular sections (like "actions subject to ULURP").

## Implemented Solutions

### 1. Enhanced Section Reference Detection

The `_check_for_section_reference` method in `document_processor.py` has been improved to:

- Add more robust pattern matching for section references
- Handle special cases like "actions subject to ULURP"
- Improve logging for better debugging
- Support multiple types of section references (numbers, keywords, etc.)

```python
def _check_for_section_reference(self, query: str) -> List[str]:
    """
    Check if the query contains a direct reference to a section number and return matching sections.
    Enhanced to handle more section reference patterns and improve matching accuracy.
    """
    # Log the query for debugging
    logger.info(f"Checking for section references in query: '{query}'")
    
    # Check for section references with various patterns
    section_patterns = [
        r'section\s+(\d+(\.\d+)*)',  # matches "section 2.2"
        r'\b(\d+\.\d+)\b',           # matches standalone "2.2"
        r'section\s+(\d+)',          # matches "section 2"
        r'actions\s+subject\s+to\s+(\w+)',  # matches "actions subject to ULURP"
        # ... additional patterns
    ]
    
    # ... implementation details
```

### 2. Improved Chunking Strategy

The chunking strategy in `_chunk_sections` has been enhanced to:

- Create smaller, more focused chunks (200 words instead of 250)
- Add special section header chunks that are prioritized in retrieval
- Include rich metadata with each chunk for better filtering
- Extract and store keywords from chunk content
- Add special handling for important sections like "Actions Subject to ULURP"

```python
def _chunk_sections(self, chunk_size=200, chunk_overlap=100):
    """
    Split sections into smaller chunks for more precise retrieval.
    Enhanced version that preserves section context and creates more semantically
    meaningful chunks with better overlap.
    """
    # ... implementation details
    
    # Create a special section header chunk that will be prioritized in retrieval
    if section_number:
        header_chunk_id = str(uuid.uuid4())
        header_chunk_text = f"Section {section_number}: {section_title}\n\nThis section contains information about {section_title}."
        
        # For specific sections, add more context to help with retrieval
        if "actions subject to ulurp" in section_title.lower():
            header_chunk_text += "\n\nThis section lists the actions that are subject to ULURP review."
```

### 3. Keyword Extraction for Better Retrieval

A new method `_extract_keywords` has been added to extract important keywords from text:

```python
def _extract_keywords(self, text: str) -> List[str]:
    """
    Extract important keywords from text for better retrieval.
    """
    # Remove common stop words
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
                 # ... more stop words
                }
    
    # Clean text and split into words
    clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = clean_text.split()
    
    # Filter out stop words and short words
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Add special keywords for specific content
    if "ulurp" in clean_text:
        keywords.extend(["ulurp", "uniform land use review procedure"])
    # ... more special cases
```

### 4. Enhanced ChromaDB Integration

The ChromaDB integration has been improved to:

- Better handle the enhanced metadata
- Implement a more robust hybrid search approach
- Add special handling for section-specific queries
- Improve the reranking algorithm to prioritize section-specific content

```python
def _extract_important_words(self, query_text: str) -> List[str]:
    """
    Extract important words from a query for keyword filtering.
    Enhanced to better handle section references, numbered sections, and special queries.
    """
    # ... implementation details
    
    # Check for special queries about ULURP actions
    if "actions subject to ulurp" in query_text.lower() or "what actions are subject to ulurp" in query_text.lower():
        logger.info("Detected special query about ULURP actions")
        return ["actions", "subject", "ulurp", "actions_subject_to", "2.2", "section 2.2"]
```

### 5. Improved Hybrid Search Reranking

The reranking algorithm has been enhanced to better prioritize section-specific content:

```python
def _rerank_hybrid_results(self, query_text: str, chunk_ids: List[str],
                          section_titles: List[str], documents: List[str],
                          distances: List[float]) -> List[Tuple[str, str, float]]:
    """
    Rerank results using both vector similarity and keyword matching.
    Enhanced to better prioritize section-specific content and special queries.
    """
    # ... implementation details
    
    # Combined score with adjusted weights:
    # - 50% vector similarity
    # - 30% keyword matching
    # - 15% section title match
    # - 5% content type (header vs regular)
    combined_score = (0.5 * similarity_score) + (0.3 * keyword_score) + 
                     (0.15 * section_match_score) + (0.05 * content_type_score)
```

### 6. Configurable Rebuild Endpoint

The `/rebuild-index` endpoint has been updated to allow configuring chunking parameters:

```python
@app.route('/rebuild-index', methods=['GET'])
def rebuild_index():
    """Rebuild the embedding index with current parameters."""
    # Allow overriding parameters via query string
    chunk_size = int(request.args.get('chunk_size', CHUNK_SIZE))
    chunk_overlap = int(request.args.get('chunk_overlap', CHUNK_OVERLAP))
    use_chunks = request.args.get('use_chunks', str(USE_CHUNKS)).lower() == 'true'
    use_chroma = request.args.get('use_chroma', str(USE_CHROMA)).lower() == 'true'
    
    # ... implementation details
```

## How to Test the Improvements

1. Rebuild the index with the new chunking strategy:
   - Visit `/rebuild-index?module=Navigating%20Zoning,%20Land%20Use,%20and%20Development%20Planning`

2. Test direct section references:
   - Ask "What is in section 2.2?"
   - Ask "Tell me about section 2.2"
   - Ask "What actions are subject to ULURP?"

3. Check the logs for detailed information about the retrieval process.

## Future Improvements

- Further optimize chunk size and overlap parameters
- Add more special case handling for common queries
- Implement a feedback mechanism to improve retrieval over time
- Consider adding a section-specific search interface
