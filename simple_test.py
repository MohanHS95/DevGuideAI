from utils.document_processor import DocumentProcessor

# Use the Navigating Zoning document
doc_path = "data/NYC Real Estate Development Curriculum - Navigating Land Use, Zoning, and Development Planning in New York City.docx"
module_name = "Navigating Zoning, Land Use, and Development Planning"

# Create processor
processor = DocumentProcessor(doc_path, module_name)
processor.load_document()
processor.process_document()

# Test chunking
print("Testing chunking...")
chunks = processor._chunk_sections(chunk_size=300, chunk_overlap=50)
print(f"Created {len(chunks)} chunks")

# Print a few chunks
for i, (chunk_id, chunk_text) in enumerate(list(chunks.items())[:3]):
    section = processor.chunk_to_section.get(chunk_id)
    print(f"\nChunk {i+1} from section '{section}':")
    preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
    print(preview)
