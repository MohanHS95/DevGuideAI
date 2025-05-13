# DevGuide AI

A web-based learning platform for navigating land use, zoning, and development planning in New York City.

## Project Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your environment variables:
```
FLASK_APP=app.py
FLASK_ENV=development

# Chunking configuration
USE_CHUNKS=true
CHUNK_SIZE=300
CHUNK_OVERLAP=50
CONTEXT_CHUNK_COUNT=5
CONTEXT_SECTION_COUNT=3
```

4. Run the application:
```bash
flask run
```

The application will be available at `http://localhost:5000`

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates
- `requirements.txt`: Project dependencies
- `documentation.md`: Project documentation
- `progress.md`: Development progress tracking
- `workplan.md`: Project roadmap and implementation plan

## Features

- Display structured learning content with hierarchical navigation
- Interactive chat interface with AI guide (Regular and Learning modes)
- RAG-powered knowledge base with embedding-based retrieval
- Multi-module support with document caching
- Glossary integration with term linking and hover effects
- Footnote processing and navigation

## Chunking Configuration

The application supports chunking for more precise RAG retrieval:

- `USE_CHUNKS`: Enable/disable chunking (true/false)
- `CHUNK_SIZE`: Size of chunks in words (default: 300)
- `CHUNK_OVERLAP`: Overlap between chunks in words (default: 50)
- `CONTEXT_CHUNK_COUNT`: Number of chunks to retrieve for context (default: 5)
- `CONTEXT_SECTION_COUNT`: Number of sections to retrieve when not using chunks (default: 3)

To test chunking functionality, run:
```bash
python test_chunking.py
```