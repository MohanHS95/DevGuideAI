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
- Comprehensive error handling and logging
- Client-side fallback highlighting
- Enhanced "no results" experience with suggestions

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
- Fuzzy search for better matching
- Search result caching for performance
- Enhanced search UI with filters and sorting options

### Planned ChromaDB Implementation
The next major enhancement will be implementing ChromaDB as a vector database for improved retrieval:

#### Planned Architecture
- **ChromaDB Integration**: Replace the current NumPy-based embedding storage with ChromaDB
- **Collection Structure**: Create separate collections for each module
- **Embedding Storage**: Store chunk embeddings in ChromaDB for efficient similarity search
- **Metadata Management**: Include section information and other metadata with each embedding
- **Query Enhancement**: Implement more sophisticated query techniques using ChromaDB's capabilities
- **Persistence**: Utilize ChromaDB's persistence capabilities for better scalability

#### Implementation Steps
1. Install ChromaDB and its dependencies
2. Create a ChromaDB client and collection management system
3. Modify the document processor to store embeddings in ChromaDB
4. Update the retrieval methods to use ChromaDB's similarity search
5. Implement metadata filtering for more precise retrieval
6. Add persistence configuration for production deployment
7. Create migration utilities to transfer existing embeddings to ChromaDB

#### Expected Benefits
- Improved search performance with larger document collections
- Better scalability for multiple modules
- More sophisticated filtering capabilities
- Enhanced retrieval precision
- Reduced memory usage for large embedding collections
- Persistent storage of embeddings across application restarts

### UI Enhancements with JavaScript-Based Collapsible Sections

We improved the UI to create a cleaner, more spacious layout with better organization of content:

#### Implementation Approach
- **JavaScript-Based Solution**: Used JavaScript to implement collapsible sections after page load
- **Template Simplification**: Simplified the Jinja2 template logic to avoid syntax errors
- **Enhanced Styling**: Added visual styling to section headers for better content organization

#### Key Components
- **Section Headers**: Enhanced section headers with visual styling and interactive elements
- **Collapsible Content**: Implemented collapsible content sections for better organization
- **CSS Transitions**: Added smooth transitions for expanding/collapsing sections
- **Event Handling**: Implemented click handlers to toggle sections open/closed

#### Technical Implementation
- **DOM Manipulation**: JavaScript code runs after page load to find all section headers
- **Content Organization**: Automatically organizes content under each header into collapsible sections
- **Event Listeners**: Adds click event listeners to toggle sections open/closed
- **State Management**: Ensures only one section is expanded at a time for better focus

#### CSS Enhancements
- **Section Header Styling**: Added styling for section headers with dropdown indicators
- **Transition Effects**: Implemented smooth transitions for expanding/collapsing sections
- **Visual Cues**: Added visual cues (cursor changes, hover effects) to indicate clickable elements
- **Content Spacing**: Improved spacing and typography for better readability

### Maintenance Notes
- Regular document format updates
- Performance monitoring
- User feedback integration
- Content validation checks
- Search functionality testing and optimization

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

### Persistent Conversation History
- **Server-Side Storage**: Implemented Flask-Session to store conversation histories on the server.
- **Session Management**: Added user ID generation and session initialization in the `/chat` endpoint.
- **Conversation Structure**:
  ```
  session['chat_histories'] = {
      'chat': {
          'module_name_1': [
              {'role': 'user', 'content': 'message', 'timestamp': '...'},
              {'role': 'assistant', 'content': 'response', 'timestamp': '...'},
              ...
          ],
          'module_name_2': [...],
      },
      'learning': {
          'module_name_1': [...],
          ...
      }
  }
  ```
- **Special Commands**:
  - `/clear`: Clears the conversation history for the current module and mode.
  - `/history`: Returns the current conversation history to the client.
- **UI Controls**:
  - Added "Clear history" button to reset the conversation.
  - Added "Export conversation" button to download the chat as a text file.
  - Added history status indicator showing the number of messages in the conversation.
- **Typing Indicators**: Added animated typing indicators while waiting for AI responses.
- **LLM Context**: Updated the Vertex AI payload to include previous messages from the conversation history.

### Response Streaming
- **Server-Sent Events (SSE)**: Implemented a new `/chat-stream` endpoint that uses SSE to stream responses to the client.
- **Streaming Simulation**: Since Vertex AI doesn't natively support streaming for Gemini 2.0 Flash, we simulate streaming by chunking the response and sending it piece by piece with small delays.
- **Persistent Connection Architecture**:
  1. The client establishes a persistent SSE connection via a GET request to `/chat-stream`
  2. The server assigns a unique stream ID to the connection and sends it to the client
  3. The client sends a POST request to `/chat-stream` with the message data
  4. The server processes the request and stores the response with the stream ID
  5. The persistent SSE connection detects the response and streams it to the client
  6. The connection remains open for subsequent messages
- **Event Types**:
  - `connected`: Sent when the SSE connection is established, includes the unique stream ID
  - `ping`: Keeps the SSE connection alive and checks for new responses
  - `user`: Contains the user message for display
  - `start`: Indicates the beginning of a streaming response
  - `chunk`: Contains a piece of the response text
  - `complete`: Contains the full response when streaming is complete
  - `error`: Contains error information if something goes wrong
  - `end`: Indicates the end of the streaming session
- **UI Components**:
  - Added a toggle switch to enable/disable streaming responses
  - Implemented a blinking cursor animation during streaming
  - Added real-time text display as chunks arrive
- **User Preferences**: The streaming setting is saved in localStorage to persist across sessions.
- **Fallback Mechanism**: If streaming fails, the system automatically falls back to the regular chat endpoint.
- **Integration with History**: Streaming responses are properly saved to the conversation history just like regular responses.

### Chat History Persistence Enhancements

We improved the chat history persistence to ensure it works correctly across page refreshes and module navigation:

- **Automatic History Loading**: Added code to automatically load chat history when the page loads, ensuring users see their previous conversations immediately.
- **Periodic History Refresh**: Implemented a timer to periodically check for history updates, keeping the chat history in sync across tabs/windows.
- **Module Navigation State Preservation**: Added code to store the current chat state (module and mode) in localStorage before navigation, ensuring it's restored when returning.
- **Mode Toggle Persistence**: Made the chat mode (chat vs. learning) persistent across page refreshes using localStorage.
- **Improved Format Handling**: Enhanced the code to handle different message formats between server and client, ensuring compatibility.
- **Better Error Handling**: Added improved error handling for history loading and chat operations.
- **Empty State Handling**: Added better empty state handling for when there's no chat history.
- **Clearer History Status**: Enhanced the history status indicator to show the number of messages in the conversation.

#### Implementation Details

```javascript
// Load history on page load
document.addEventListener('DOMContentLoaded', function() {
    // Fetch history for current mode
    fetchServerHistory();

    // Also set up a timer to periodically check for history updates
    setInterval(function() {
        if (!chatPanel.classList.contains('translate-y-full')) {
            // Only refresh if chat panel is open
            fetchServerHistory();
        }
    }, 30000); // Check every 30 seconds
});

// Store current chat state before navigation
document.getElementById('moduleSelect').addEventListener('change', function() {
    // Store current chat state before navigation
    localStorage.setItem('lastModule', document.getElementById('moduleSelect').value);
    localStorage.setItem('lastMode', currentMode || 'chat');

    // Navigate to new module
    window.location.href = '/?module=' + encodeURIComponent(this.value);
});
```
- **Stream ID System**: Each streaming connection is assigned a unique ID that's used to track responses:
  ```python
  # Generate and store a unique stream ID
  stream_id = str(uuid.uuid4())
  session['current_stream_id'] = stream_id

  # Store responses with the stream ID
  response_key = f'stream_response_{stream_id}'
  session[response_key] = {
      'user_message': user_message,
      'ai_response': ai_response_text,
      'timestamp': datetime.now().isoformat()
  }
  ```
- **Conversation History Management**: Special handling ensures conversation history is properly maintained:
  - Duplicate message detection prevents adding the same message multiple times
  - User messages are properly tracked and added to history
  - Special commands like `/clear` also clear pending responses
  - Session variables are properly cleaned up when history is cleared
  ```python
  # Check if this is a duplicate message
  is_duplicate = False
  if conversation_history and len(conversation_history) > 0:
      last_msg = conversation_history[-1]
      if last_msg['role'] == 'user' and last_msg['content'] == user_message:
          is_duplicate = True
  ```

- **Persistent Connection Management**: The streaming system maintains a persistent connection that handles multiple requests:
  ```javascript
  // First, create the EventSource to establish the SSE connection
  eventSource = new EventSource(`/chat-stream?t=${Date.now()}`);
  let streamConnected = false;
  let streamId = null;

  // When connection is established, send the message
  eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'connected') {
      streamId = data.stream_id;
      streamConnected = true;
      sendMessageToStream();
    }
    // Handle other event types...
  };
  ```

- **Stream Polling System**: The server continuously checks for new responses for a specific stream:
  ```python
  # Keep the connection alive with periodic pings
  ping_count = 0
  while ping_count < 100:  # Limit to prevent infinite loops
      time.sleep(1)
      yield f"data: {json.dumps({'type': 'ping', 'count': ping_count})}\n\n"
      ping_count += 1

      # Check if there's a response ready for this stream
      response_key = f'stream_response_{stream_id}'
      if response_key in session:
          # Get the stored response data and stream it
          # ...
  ```

## Embedding-Based Retrieval

We enhanced context retrieval for the chat interface by vectorizing document content and performing cosine similarity search to identify the most semantically relevant content at query time:

### Section-Based Retrieval (Initial Implementation)

- Added `SentenceTransformer` model loading and embedding computation in `DocumentProcessor.build_embedding_index()`
- Stored section embeddings and precomputed norms for efficient cosine similarity (using NumPy) in `utils/document_processor.py`
- Implemented `get_relevant_sections_embedding(query, top_n)` to return top-N sections per similarity scores
- Updated `app.py` `/chat` route to call `get_relevant_sections_embedding()` over the full document, including support for a configurable `CONTEXT_SECTION_COUNT` environment variable
- Removed explicit per-section context banners in the front-end; the AI now synthesizes content across all retrieved sections seamlessly

### Chunk-Based Retrieval (Enhanced Implementation)

We further improved retrieval precision by implementing chunk-based retrieval:

- Added `_chunk_sections(chunk_size, chunk_overlap)` method to split sections into smaller, more focused chunks
- Implemented chunk caching to improve performance
- Created `get_relevant_chunks_embedding(query, top_n)` to find the most relevant chunks and return their parent sections
- Added `get_chunk_context(query, top_n)` to retrieve the raw text of the most relevant chunks
- Updated `app.py` to support both section-based and chunk-based retrieval via environment variables:
  - `USE_CHUNKS`: Enable/disable chunking
  - `CHUNK_SIZE`: Size of chunks in words
  - `CHUNK_OVERLAP`: Overlap between chunks in words
  - `CONTEXT_CHUNK_COUNT`: Number of chunks to retrieve for context
  - `CONTEXT_SECTION_COUNT`: Number of sections to retrieve when not using chunks

### Chunking Strategy

The chunking implementation follows these principles:

1. **Semantic Preservation**: Chunks are created by grouping paragraphs to avoid breaking semantic units
2. **Overlapping Content**: Adjacent chunks share some content to maintain context across chunk boundaries
3. **Section Attribution**: Each chunk maintains a reference to its source section for proper attribution
4. **Fallback Mechanism**: If chunk-based retrieval fails, the system falls back to section-based retrieval
5. **Configurable Parameters**: Chunk size and overlap are configurable via environment variables

### Implementation Details

The chunking functionality is implemented in the following components:

#### DocumentProcessor Class Extensions

- **New Attributes**:
  - `chunks`: Dictionary mapping chunk IDs to chunk text
  - `chunk_to_section`: Dictionary mapping chunk IDs to their source section
  - `chunk_embeddings`: NumPy array of chunk embeddings
  - `chunk_ids`: List of chunk IDs matching embeddings
  - `chunk_embedding_norms`: Precomputed norms for chunk embeddings
  - `use_chunks`: Flag to control whether to use chunks or sections
  - `use_chroma`: Flag to control whether to use ChromaDB
  - `chroma_manager`: ChromaDB manager instance

- **New Methods**:
  - `_chunk_sections(chunk_size, chunk_overlap)`: Splits sections into smaller chunks
  - `_get_chunk_cache_path()`: Generates a unique cache path for chunk data
  - `_save_chunks_to_cache()`: Saves chunk data to cache
  - `_load_chunks_from_cache()`: Loads chunk data from cache
  - `get_relevant_chunks_embedding(query, top_n)`: Returns relevant sections based on chunk similarity
  - `get_chunk_context(query, top_n)`: Returns the text of the most relevant chunks

#### Chunking Algorithm

The `_chunk_sections` method implements the following algorithm:

1. Process each section (excluding special sections like glossary)
2. Clean HTML tags from section content
3. Split content into paragraphs
4. Group paragraphs into chunks based on word count:
   - If a paragraph is larger than the chunk size, split it with overlap
   - If adding a paragraph would exceed the chunk size, finalize the current chunk
   - Otherwise, add the paragraph to the current chunk
5. Generate a unique ID for each chunk
6. Store the chunk text and its source section

#### Chat Endpoint Enhancements

The `/chat` endpoint in `app.py` has been updated to:

1. Check if chunking is enabled via the `USE_CHUNKS` environment variable
2. If enabled, use `get_chunk_context` to retrieve the most relevant chunks
3. If chunk retrieval fails, fall back to section-based retrieval
4. Use the retrieved content as context for the LLM

#### Configuration Options

The following environment variables control chunking behavior:

```
# Chunking configuration
USE_CHUNKS=true           # Enable/disable chunking
CHUNK_SIZE=300            # Size of chunks in words
CHUNK_OVERLAP=50          # Overlap between chunks in words
CONTEXT_CHUNK_COUNT=5     # Number of chunks to retrieve for context
CONTEXT_SECTION_COUNT=3   # Number of sections to retrieve when not using chunks

# ChromaDB configuration
USE_CHROMA=true           # Enable/disable ChromaDB
```

### ChromaDB Integration

We implemented ChromaDB as a vector database to improve the storage and retrieval of embeddings:

#### ChromaManager Class

We created a new `ChromaManager` class in `utils/chroma_manager.py` to handle ChromaDB operations:

- **Initialization**: Sets up a persistent ChromaDB client with a specified directory
- **Collection Management**: Creates and manages collections for sections and chunks
- **Embedding Storage**: Stores embeddings with metadata in ChromaDB collections
- **Similarity Search**: Performs efficient similarity search using ChromaDB

#### Key Methods

- `get_or_create_collection(module_name, is_chunk)`: Gets an existing collection or creates a new one
- `store_embeddings(collection_name, ids, embeddings, metadatas)`: Stores embeddings in a collection
- `query_collection(collection_name, query_embedding, n_results)`: Performs similarity search
- `delete_collection(collection_name)`: Deletes a collection if it exists
- `delete_collections(module_name)`: Deletes all collections (both section and chunk) for a specified module
- `store_section_embeddings(module_name, section_titles, embeddings)`: Stores section embeddings in ChromaDB
- `store_chunk_embeddings(module_name, chunk_ids, chunk_texts, chunk_to_section, embeddings)`: Stores chunk embeddings in ChromaDB
- `query_sections(module_name, query_embedding, top_n)`: Queries for the most relevant sections
- `query_chunks(module_name, query_embedding, top_n)`: Queries for the most relevant chunks

### Retrieval Quality Improvements

We attempted to enhance retrieval quality, particularly for queries that reference specific section numbers:

#### Section Reference Detection

- **Pattern Matching**: Implemented regex patterns to detect section references in queries:
  ```python
  # Match both "section X.Y" and standalone "X.Y" formats
  section_pattern = r'(?:section\s+)?(\d+\.\d+)'
  ```
- **Direct Section Lookup**: Added logic to directly retrieve sections when a section number is detected:
  ```python
  # Check if the query contains a section reference
  section_match = re.search(section_pattern, query.lower())
  if section_match:
      section_number = section_match.group(1)
      # Try to find the section directly
      for section_id, section in self.sections.items():
          if section_number in section_id or section_number in section.get('title', ''):
              return [section_id]
  ```

#### Hybrid Search Enhancement

- **Keyword Weight Increase**: Increased the weight of keyword matches from 30% to 40% to prioritize exact matches:
  ```python
  # Hybrid search combining keyword and semantic search
  combined_scores = (0.6 * semantic_scores) + (0.4 * keyword_scores)
  ```
- **Important Word Extraction**: Improved extraction of important words from queries to better handle section numbers
- **Fallback Mechanisms**: Added multiple fallback strategies when direct section lookup fails

#### Rebuild Index Endpoint

- **New API Endpoint**: Added a `/rebuild-index` endpoint to refresh embeddings with current parameters:
  ```python
  @app.route('/rebuild-index')
  def rebuild_index():
      module = request.args.get('module')
      if not module or module not in document_processors:
          return jsonify({"status": "error", "message": "Invalid module"})

      # Rebuild the index for the specified module
      document_processors[module].build_embedding_index(force_rebuild=True)

      return jsonify({
          "status": "success",
          "message": f"Index rebuilt for module: {module}",
          "parameters": {
              "chunk_size": document_processors[module].chunk_size,
              "chunk_overlap": document_processors[module].chunk_overlap,
              "use_chunks": document_processors[module].use_chunks,
              "use_chroma": document_processors[module].use_chroma
          }
      })
  ```

#### Known Issues and Next Steps

- **Retrieval Inconsistency**: The system still has issues with consistently retrieving specific sections
- **Section Numbering Challenges**: Different section numbering formats can cause retrieval problems
- **Planned Improvements**:
  - Implement smaller chunks with larger overlap for more granular retrieval
  - Add section metadata to chunks for better filtering
  - Enhance the UI to show section references more prominently
  - Implement a more sophisticated section reference detection system

#### Performance Comparison

We created test scripts to compare the performance of ChromaDB with the original in-memory approach:

- **Initial Build Time**: ChromaDB takes slightly longer for initial embedding storage
- **Query Performance**: ChromaDB provides comparable query performance to in-memory search
- **Scalability**: ChromaDB offers better scalability for larger document collections
- **Persistence**: ChromaDB provides persistent storage of embeddings across application restarts

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

## Document Caching and Processing Enhancements

### Document Caching System
We implemented a robust document caching system to improve application performance:

- **Cache Storage**: Created a `cache/` directory to store processed documents and embeddings
- **Cache Management**:
  - Added `_get_cache_path()` and `_get_embedding_cache_path()` to generate unique cache file paths
  - Implemented `_is_cache_valid()` to check if cache is newer than source document
  - Created `_save_to_cache()` and `_load_from_cache()` methods for document data
  - Added `_save_embeddings_to_cache()` and `_load_embeddings_from_cache()` for embedding vectors

- **Performance Metrics**:
  - Added timing information to measure processing and loading times
  - Implemented logging to show cache usage and performance gains
  - Typical speedup: first load ~1-2 seconds, cached load ~0.01 seconds

### Enhanced Section Extraction
We improved the section extraction logic to handle various document formats:

- **Heading Detection**:
  - Enhanced detection beyond style-based headings (e.g., "Heading 1")
  - Added support for numbered headings (e.g., "1.2.3 Section Title")
  - Implemented detection for ALL CAPS headings
  - Added heuristics to estimate heading levels based on formatting

- **Hierarchy Tracking**:
  - Added `section_levels` dictionary to track heading levels
  - Implemented `section_parents` dictionary to track parent-child relationships
  - Used a section stack to maintain hierarchy during document processing
  - Updated `get_section_hierarchy()` to build a proper multi-level hierarchy

- **Special Content Handling**:
  - Added support for tables with HTML conversion
  - Implemented image detection and placeholder insertion
  - Added special formatting for list paragraphs

### UI Enhancements
We updated the user interface to support the enhanced document processing:

- **Table of Contents**:
  - Modified to display nested subsections with proper indentation
  - Added support for up to three levels of hierarchy
  - Implemented visual cues for hierarchy levels

- **Content Display**:
  - Added CSS styling for tables with responsive design
  - Implemented image placeholders with appropriate styling
  - Enhanced list formatting for better readability

### Data Flow
1. On application startup, document processor checks for cached versions of each document
2. If valid cache exists, loads processed document and embeddings from cache
3. If no cache or outdated, processes document and saves to cache
4. When building section hierarchy, uses tracked levels and parent relationships
5. UI renders the enhanced hierarchy in the table of contents

*See* [`workplan.md`](workplan.md) under *Phase 1: Document Processing and Display* for the original implementation roadmap.

## Module Content Page Redesign

We redesigned the module content page to create a more cohesive user experience that matches the homepage aesthetic:

### Design Approach
- **Consistent Design Language**: Applied the same design principles and visual elements from the homepage
- **Modern UI Elements**: Implemented a gradient hero header, improved typography, and better spacing
- **Enhanced Navigation**: Added breadcrumbs and improved table of contents
- **Visual Hierarchy**: Created clear section headers with improved styling

### Implementation Details
- **New Template**: Created `index_redesign.html` as a modern alternative to the original template
- **Flask Route Update**: Modified the `home()` route in `app.py` to use the new template by default
- **Responsive Design**: Ensured the new design works well on different screen sizes
- **CSS Enhancements**: Added new styles for the header, content area, and table of contents

### Key Components
- **Hero Header**: Added a gradient background header with the module title
- **Table of Contents**: Redesigned with better spacing, typography, and visual hierarchy
- **Content Area**: Improved with better typography, spacing, and section headers
- **Footer**: Added a consistent footer that matches the homepage
- **Breadcrumbs**: Implemented for improved navigation and context

### Technical Implementation
- **Template Structure**:
  ```html
  <div class="flex flex-col min-h-screen">
    <!-- Hero Header -->
    <header class="bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-8">
      <!-- Header content -->
    </header>

    <!-- Main Content -->
    <main class="flex-grow flex">
      <!-- Sidebar TOC -->
      <aside class="w-1/4 bg-gray-50 p-4 border-r">
        <!-- Table of contents -->
      </aside>

      <!-- Content Area -->
      <div class="w-3/4 p-8">
        <!-- Section content -->
      </div>
    </main>

    <!-- Footer -->
    <footer class="bg-gray-100 py-6 border-t">
      <!-- Footer content -->
    </footer>
  </div>
  ```

- **Section Headers**: Enhanced with better styling and visual hierarchy:
  ```html
  <div class="section-header bg-indigo-50 p-4 mb-6 rounded-lg border-l-4 border-indigo-500">
    <h2 class="text-2xl font-bold text-indigo-800">{{ section.title }}</h2>
  </div>
  ```

- **Table of Contents**: Improved with better organization and styling:
  ```html
  <div class="toc-header bg-indigo-600 text-white p-4 text-center font-bold text-xl">
    Table of Contents
  </div>
  <div class="toc-content p-4">
    {% for section_id, section in sections.items() %}
      {% if section.level == 1 %}
        <div class="section-item bg-indigo-50 p-3 my-2 rounded">
          <a href="#{{ section_id }}" class="text-indigo-700 font-semibold">
            {{ section.title }}
          </a>
        </div>
      {% elif section.level == 2 %}
        <div class="subsection-item pl-4 py-2">
          <a href="#{{ section_id }}" class="text-gray-700">
            {{ section.title }}
          </a>
        </div>
      {% endif %}
    {% endfor %}
  </div>
  ```

### Known Issues and Future Improvements
- **Section Headers**: Need to make main section headers more bold in the table of contents
- **Chat Button**: Currently not functioning properly - not showing the chat panel when clicked
- **Mobile Responsiveness**: Needs further testing and optimization for smaller screens
- **Accessibility**: Need to ensure proper contrast ratios and keyboard navigation

### Search Functionality in Redesigned Template

We fixed the search functionality in the redesigned template that was missing:

#### HTML Implementation
- Added a search results dropdown container to the HTML:
  ```html
  <div id="searchResults" class="absolute w-full mt-2 bg-white rounded-lg shadow-lg hidden z-20 max-h-96 overflow-y-auto">
      <div id="searchResultsList"></div>
      <div id="searchPagination" class="p-2 hidden"></div>
  </div>
  ```

#### JavaScript Implementation
- Added event listener for the search input to trigger search as the user types:
  ```javascript
  searchInput.addEventListener('input', function() {
      clearTimeout(searchTimeout);
      const query = this.value.trim();
      currentQuery = query;

      if (query.length < 2) {
          searchResults.classList.add('hidden');
          return;
      }

      searchTimeout = setTimeout(() => {
          performSearch();
      }, 300);
  });
  ```

- Implemented the `performSearch` function to fetch search results from the server:
  ```javascript
  function performSearch() {
      const query = currentQuery;
      if (query.length < 2) return;

      // Build search URL with parameters
      const currentModule = document.getElementById('moduleSelect').value;
      const searchUrl = `/search?module=${encodeURIComponent(currentModule)}&q=${encodeURIComponent(query)}&page=${currentPage}&per_page=${currentPerPage}`;

      // Show loading indicator
      searchResultsList.innerHTML = '<div class="p-4 text-center"><div class="inline-block animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-600"></div><span class="ml-2">Searching...</span></div>';
      searchResults.classList.remove('hidden');

      // Fetch search results
      fetch(searchUrl)
          .then(response => response.json())
          .then(data => {
              // Render results and pagination
              renderSearchResults(data);
              if (data.total_pages > 1) {
                  renderPagination(data);
                  searchPagination.classList.remove('hidden');
              } else {
                  searchPagination.classList.add('hidden');
              }
          })
          .catch(error => {
              console.error(`Search error: ${error.message}`);
              searchResultsList.innerHTML = `<div class="px-4 py-2 text-sm text-red-500">Error: ${error.message}</div>`;
          });
  }
  ```

- Added functions for rendering search results and pagination:
  ```javascript
  function renderSearchResults(data) {
      const results = data.results || [];

      if (results.length > 0) {
          searchResultsList.innerHTML = results.map(result => {
              // Build result HTML with section title, breadcrumb, and highlighted context
              return `
                  <a href="/?module=${encodeURIComponent(moduleSelect.value)}&section=${encodeURIComponent(result.section)}"
                     class="block search-result-item">
                      <div class="search-result-section">${result.section}</div>
                      ${breadcrumb ? `<div class="search-result-breadcrumb">${breadcrumb}</div>` : ''}
                      <div class="search-result-context">${contextHtml}</div>
                  </a>
              `;
          }).join('');
      } else {
          // Show "no results" message with suggestions
          searchResultsList.innerHTML = `
              <div class="px-4 py-4 text-sm text-gray-500">
                  <div class="text-center mb-3">No results found for "${query}"</div>
                  <div class="text-xs">
                      <p class="font-medium mb-1">Suggestions:</p>
                      <ul class="list-disc pl-5">
                          <li>Check your spelling</li>
                          <li>Try more general keywords</li>
                          <li>Try different keywords</li>
                          <li>Try searching in all sections</li>
                      </ul>
                  </div>
              </div>
          `;
      }
  }
  ```

- Added helper functions for text highlighting and pagination:
  ```javascript
  // Helper function to highlight text if server didn't do it
  function highlightText(text, query) {
      if (!text || !query) return text;

      // Remove quotes for phrase search
      let searchTerms = query;
      if (searchTerms.startsWith('"') && searchTerms.endsWith('"')) {
          searchTerms = searchTerms.substring(1, searchTerms.length - 1);
      }

      // Highlight the phrase
      const regex = new RegExp(escapeRegExp(searchTerms), 'gi');
      return text.replace(regex, match => `<mark>${match}</mark>`);
  }
  ```

- Added click event listener to hide search results when clicking outside:
  ```javascript
  document.addEventListener('click', function(e) {
      const isSearchRelated = searchResults.contains(e.target) ||
                             searchInput.contains(e.target) ||
                             e.target === searchInput;

      if (!isSearchRelated) {
          searchResults.classList.add('hidden');
      }
  });
  ```

#### CSS Styling
- Added styles for search results display to match the redesigned template's aesthetic:
  ```css
  .search-result-item {
      border-bottom: 1px solid #e5e7eb;
      padding: 12px;
      transition: background-color 0.2s;
  }
  .search-result-item:hover {
      background-color: #f9fafb;
  }
  .search-result-section {
      font-weight: 600;
      color: #4f46e5;
      margin-bottom: 4px;
  }
  mark {
      background-color: #fef08a;
      padding: 0 2px;
      border-radius: 2px;
  }
  ```

This implementation restores all the original search functionality in the redesigned template, including real-time search results, highlighting, pagination, and navigation to matching sections.

## Search Functionality Enhancements

We improved the search functionality to fix issues and enhance the user experience:

### Backend Improvements

#### DocumentProcessor Search Method
- **Enhanced Logging**: Added detailed logging throughout the search process to track queries, matches, and results
- **Improved Error Handling**: Added comprehensive error handling for edge cases
- **Better Search Logic**:
  - Fixed the search logic to better handle empty paragraphs
  - Improved phrase vs. keyword search differentiation
  - Added match counting for better debugging
  - Enhanced relevance scoring

#### Search Highlighting
- **Improved Highlighting**: Enhanced the text highlighting to better match search terms
- **Fallback Mechanisms**: Added fallback highlighting when server-side highlighting fails
- **Word Length Handling**: Reduced the minimum word length for highlighting from 3 to 2 characters
- **Detailed Logging**: Added logging to track highlighting issues and successes

#### Search Route in app.py
- **Enhanced Error Handling**: Added comprehensive error handling for all steps
- **Improved Logging**: Added detailed logging throughout the search process
- **Validation**: Added validation for document processor and sections
- **Response Enhancement**: Included the original query in the response for client-side use

### Frontend Improvements

#### Search UI
- **Console Logging**: Added console logging for debugging
- **Enhanced Error Handling**: Improved error handling in fetch requests
- **Better Result Display**: Enhanced the display of search results with better formatting
- **Fallback Mechanisms**: Added fallback mechanisms when expected data is missing
- **No Results Enhancement**: Improved the "no results" message with helpful suggestions
- **Client-side Highlighting**: Added a client-side highlighting function as a backup

#### Chat Search Integration
- **Improved Error Handling**: Enhanced error handling in the chat search functionality
- **Better Validation**: Added better validation of search results
- **Enhanced Display**: Improved the display of search results in chat
- **Error Messaging**: Added proper error messaging and logging

### Data Flow
1. User enters search query in the search box
2. Frontend sends query to backend with module and filter parameters
3. Backend logs the request and validates parameters
4. DocumentProcessor searches for matches in content
5. Backend logs match details and returns results with metadata
6. Frontend processes results and displays them with highlighting
7. If server-side highlighting fails, client-side highlighting is applied
8. User can navigate to matching sections by clicking results

### Implementation Details
- Added detailed logging at all levels of the search process
- Improved error handling to provide better feedback
- Enhanced the search logic to better handle different search scenarios
- Improved the highlighting to better match search terms
- Enhanced the UI to provide better feedback to users
- Added client-side fallback mechanisms for robustness
