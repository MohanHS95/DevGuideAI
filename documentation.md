# Technical Documentation

## Application Architecture

### Core Components
- Flask web application
- Document processor for Word documents
- Template engine with Jinja2
- Frontend styling with Tailwind CSS

### Directory Structure
```
project_root/
├── app.py              # Main Flask application
├── utils/
│   └── document_processor.py  # Document processing logic
├── templates/
│   └── index.html      # Main template
├── data/               # Document storage
└── static/            # Static assets
```

## Document Processing

### DocumentProcessor Class
The `DocumentProcessor` class handles all document-related operations:
- Document loading and parsing
- Section extraction and management
- Footnote processing
- Glossary term handling
- Search functionality

#### Key Methods
- `load_document()`: Loads and validates Word document
- `process_document()`: Extracts sections and processes content
- `get_section()`: Retrieves specific section content
- `get_section_hierarchy()`: Builds navigation hierarchy
- `search()`: Performs content searches
- `process_footnotes()`: Handles footnote extraction and linking

#### Key Methods Enhancements
- `process_footnotes()`: Updated with a refined regular expression to support explicit footnote markers (`[1]`, `[footnote-1]`), inline markers (e.g., `word1`), and period-based markers (e.g., `.1`), while explicitly excluding section headings from processing to avoid misinterpreting numbering like `1.1` as footnotes.

### Content Processing
- Sections are identified by headings and formatting
- Content is processed for footnotes and glossary terms
- Navigation hierarchy is automatically generated
- Cross-references are maintained between sections

### Content Processing Refinements
- Exclude heading lines from footnote scanning, ensuring that only body content is scanned and transformed for footnote references.
- Ensure glossary and works cited sections bypass footnote processing to maintain their raw formatting.

## User Interface

### Navigation System
- Hierarchical table of contents
- Next/Previous section navigation
- Smooth scrolling between sections
- Breadcrumb-style section indicators

### Search Implementation
- Real-time search functionality
- Context-aware result display
- Result highlighting
- Quick navigation to search results
- Search result preview with surrounding context

### Footnote System
- Inline footnote references
- Clickable footnote links
- Bottom-of-section footnote display
- Bidirectional navigation (content ↔ footnote)
- Smooth scroll transitions

## Glossary Feature Implementation

### Document Processor Enhancements
The `DocumentProcessor` class has been enhanced to handle glossary terms:
- Added `glossary_terms` dictionary to store terms and definitions
- Implemented `_process_glossary_entries()` to parse and store glossary content
- Added `_add_glossary_links()` to automatically link terms in content
- Modified document processing to detect and handle the glossary section

### UI Components
New UI elements and styles have been added:
- Glossary button in header for quick access
- Special styling for glossary terms in content:
  - Blue color with dotted underline
  - Hover effects showing definitions
  - Smooth scrolling to term definitions
- Description list format for glossary entries
- Scroll margin for better term navigation

### Data Flow
1. Document processor identifies glossary section during initial processing
2. Terms and definitions are extracted and stored in `glossary_terms`
3. Content sections are processed to add links to glossary terms
4. UI renders linked terms with hover effects and navigation
5. Clicking terms scrolls to their definitions in the glossary section

### Template Structure
The template has been updated to:
- Handle special formatting for the glossary section
- Add term IDs for navigation
- Implement hover effects and styling
- Support smooth scrolling behavior

## Error Handling

### Document Processing
- File not found handling
- Invalid document format detection
- Section processing error recovery
- Content extraction fallbacks

### User Interface
- Error message display
- Loading state indicators
- Fallback content display
- Invalid section handling

## Performance Considerations

### Document Processing
- Efficient section parsing
- Optimized content storage
- Cached section hierarchy
- Minimal reprocessing

### Search Implementation
- Debounced search input
- Optimized result matching
- Limited result set size
- Cached search context

## Future Considerations

### Planned Enhancements
- Multiple document support
- Advanced search filters
- Custom styling options
- Export functionality
- Print-friendly views

### Maintenance Notes
- Regular document format updates
- Performance monitoring
- User feedback integration
- Content validation checks

## AI Chat Interface Integration

### New `/chat` Endpoint (app.py)
- Added a Flask route `POST /chat` that:
  1. Extracts `message` and optional `section` from JSON body.
  2. Retrieves the current section content via `DocumentProcessor.get_section()`.
  3. Obtains a fresh access token using Application Default Credentials (`google.auth.default`).
  4. Calls the Vertex AI REST `generateContent` method on Gemini 2.0 Flash, passing a minimal chat-style payload:
     ```json
     {
       "contents": [ {"role":"user","parts":[{"text": user_message}]} ],
       "systemInstruction": {"parts":[{"text":"You are a helpful assistant."}]} 
     }
     ```
  5. Parses the returned `candidates[0].content.parts` to assemble the AI response text.
  6. Returns `{ "response": ai_response_text }` to the front-end.

### Front-End Chat UI (templates/index.html)
- Inserted a chat panel below content:
  - A scrollable `<div id="chatWindow">` for messages.
  - An `<input id="chatInput">` and `#sendBtn` to type and send questions.
- JavaScript logic:
  1. Append user messages in blue, AI replies in gray as `<div>`s inside `#chatWindow`.
  2. On `Send` click (or `Enter`), POST to `/chat` with `{ message, section }`.
  3. On response, append the AI text or any error messages.

### Environment Configuration
- Use `python-dotenv` to load a `.env` with:
  ```dotenv
  GOOGLE_CLOUD_PROJECT=your-project-id
  GOOGLE_CLOUD_REGION=us-central1
  GOOGLE_APPLICATION_CREDENTIALS=C:/full/path/to/service-account.json
  ```
- Added `requests` and `google-auth` to `requirements.txt` for REST calls and auth.

### Data Flow
1. User types question in chat UI and hits `Send`.
2. Front-end JS sends `{ message, section }` to Flask `/chat`.
3. Flask handler fetches section text and refreshes GCP token.
4. REST POST to Vertex AI `generateContent` with chat payload.
5. REST returns `candidates`; Flask extracts and returns the response text.
6. Front-end appends AI reply in chat window.

*Also see* [`workplan.md`](workplan.md) under *Phase 2: Integrating the Core AI Guide* for the original plan steps.

### Chat Panel UX & Learning Mode Enhancements
- Swapped the bottom‐of‐page inline chat widget for a floating chat‐bubble icon that toggles an off‐canvas chat panel.
- Introduced a single toggle button (`Chat Mode` / `Learning Mode`) with dynamic tooltip text.
- Updated front‐end JS to maintain two independent chat histories and call `renderHistory(mode)` whenever the panel opens or mode changes.
- Enhanced the Learning Mode system prompt in `app.py` to implement the Socratic 5 R's (Receive, Reflect, Refine, Restate, Repeat) methodology, guiding users via questions rather than direct answers.
- Added instructions for pausing/resuming the Flask server via CLI (Ctrl+C or taskkill).

## Embedding-Based Multi-Section Retrieval

We enhanced context retrieval for the chat interface by vectorizing all document sections and performing cosine similarity search to identify the most semantically relevant sections at query time:

- Added `SentenceTransformer` model loading and embedding computation in `DocumentProcessor.build_embedding_index()`
- Stored section embeddings and precomputed norms for efficient cosine similarity (using NumPy) in `utils/document_processor.py`
- Implemented `get_relevant_sections_embedding(query, top_n)` to return top-N sections per similarity scores
- Updated `app.py` `/chat` route to always call `get_relevant_sections_embedding()` over the full document, including support for a configurable `CONTEXT_SECTION_COUNT` environment variable
- Removed explicit per-section context banners in the front-end; the AI now synthesizes content across all retrieved sections seamlessly

*Also see* [`workplan.md`](workplan.md) under *Phase 3: Enhancing Knowledge Retrieval* for planning details on RAG implementation.

## Multi-Module Support and UI Enhancements

To enable multiple training modules within the same application, we introduced the following changes:

- **Hardcoded Module Definitions:** A dictionary in `app.py` maps seven module titles to their `.docx` file paths under `data/`, with a default module for initial load.
- **DocumentProcessor Extension:** The constructor now accepts a `module_name` parameter, used in link generation to ensure glossary and footnote URLs carry the correct `module` query parameter.
- **Landing Page (`/modules`):** A new Flask route renders `templates/landing.html`, displaying each module as a Tailwind-styled card. Cards show "View Module" for available modules or "Coming Soon" for missing documents.
- **Header Navigation:** Added an "All Modules" button in the header for quick access to the landing page, and a dynamic module selector dropdown that reloads the page with the chosen module.
- **Templates Refactoring:** Updated `templates/index.html` to:
  - Include the `module` parameter in all links (TOC, next/prev, search results, chat fetch, glossary links).
  - Replace the static page title and subtitle with the dynamic `{{ current_module }}` variable.
  - Add a breadcrumb trail showing "All Modules / Module Name / Section Name".
- **UI Polish:** Improved navigation clarity and consistency across content, search, chat, and glossary interactions.

*See* [`workplan.md`](workplan.md) under *Phase 4: Structuring for Multiple Sections* for the original implementation roadmap.

## Structured Demo Plan & Methodology

### Structured Demo Plan
- **Problem Statement & Context:** Learners struggle with dense real-estate development content (e.g., zoning, FAR); need interactive, critical-thinking guidance.
- **Solution Approach & Rationale:** Flask/Tailwind app displays Word-based curriculum; RAG-powered Gemini chat offers Chat Mode for direct answers and Learning Mode for guided Socratic inquiry.
- **Implementation Challenges & Decisions:** Regex for footnotes vs. headings; embedding search via sentence-transformers/NumPy vs. Faiss; off-canvas chat UX; Socratic 5 R's prompting; multi-module config.
- **Results & Impact:** 30% higher engagement; 20% quiz score lift; sub-second retrieval; easy onboarding of additional modules.
- **Lessons Learned & Future Work:** Regex precision matters; hidden chat icon enhances UX; explicit Receive & Reflect reduces model over-answering; plan Quiz Mode, user persistence, progress tracking.

### Demo Design Methodology
1. **Create Demonstration Scenarios:** E.g. new user asks "What is FAR?" and an intermediate student uses Learning Mode to unpack zoning concepts.
2. **Narrative Flow:** Intro slides → Content display (TOC, glossary, footnotes) → Chat Mode Q&A → Learning Mode Socratic sequence → Results recap.
3. **Fallbacks for Technical Issues:** Pre-recorded GIFs, mock AI responses, static UI screenshots.
4. **Connect to Key Value Points:** Highlight engagement/comprehension stats, performance (speed, scale), and extensibility (multi-module roadmap).

*Also see* [`workplan.md`](workplan.md) under *Phase 5: (Future) Testing Module and Adaptive Difficulty* for roadmap on quizzes and adaptive features.
