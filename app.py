from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
import os
import uuid
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from utils.document_processor import DocumentProcessor
import logging
from google.cloud import aiplatform
import requests
from google.auth import default
from google.auth.transport.requests import Request as GoogleAuthRequest
from flask_session import Session

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Vertex AI
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
region = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
if not project_id:
    logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
else:
    aiplatform.init(project=project_id, location=region)

# Default chunking parameters - updated for better section reference retrieval
DEFAULT_CHUNK_SIZE = 150  # Smaller chunks for more precise retrieval
DEFAULT_CHUNK_OVERLAP = 75  # Proportionally larger overlap for better context preservation

# Chunking configuration
USE_CHUNKS = os.getenv("USE_CHUNKS", "true").lower() == "true"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))  # Smaller chunks for more precise retrieval
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP)))  # Larger overlap to ensure context is preserved
CONTEXT_CHUNK_COUNT = int(os.getenv("CONTEXT_CHUNK_COUNT", "7"))  # More chunks for better coverage
CONTEXT_SECTION_COUNT = int(os.getenv("CONTEXT_SECTION_COUNT", "3"))

# ChromaDB configuration
USE_CHROMA = os.getenv("USE_CHROMA", "true").lower() == "true"
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"

# Streaming configuration
ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
STREAM_CHUNK_SIZE = 20  # Number of characters to send in each stream chunk

# Configure Flask app
app = Flask(__name__)
app.config["SESSION_PERMANENT"] = True
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", os.urandom(24).hex())
app.config["SESSION_FILE_DIR"] = os.path.join(os.getcwd(), "flask_session")
app.config["PERMANENT_SESSION_LIFETIME"] = 86400  # 24 hours in seconds

# Initialize Flask-Session
Session(app)

# Create session directory if it doesn't exist
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)

# Module definitions (hardcoded for now)
modules = {
    "Construction Essentials for Real Estate Developers": "data/Construction Essentials for Real Estate Developers.docx",
    "Financing Real Estate Development Projects": "data/Financing Real Estate Development Projects.docx",
    "Multifamily Residential Development": "data/Multifamily Residential Development.docx",
    "Navigating Zoning, Land Use, and Development Planning": "data/NYC Real Estate Development Curriculum - Navigating Land Use, Zoning, and Development Planning in New York City.docx",
    "Real Estate Acquisition and Ownership": "data/Real Estate Acquisition and Ownership.docx",
    "The Real Estate Development Process": "data/The Real Estate Development Process.docx",
    "Real Estate Development Market Analysis and Feasibility": "data/Real Estate Development Market Analysis and Feasibility.docx",
}
# Default module
default_module = "Navigating Zoning, Land Use, and Development Planning"

# Initialize processors
doc_processors = {}
for module_name, path in modules.items():
    # Pass module name to DocumentProcessor for proper link generation
    processor = DocumentProcessor(path, module_name)
    try:
        processor.load_document()
        processor.process_document()
        # Use chunking and ChromaDB if enabled
        processor.build_embedding_index(
            use_chunks=USE_CHUNKS,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            use_chroma=USE_CHROMA
        )
        logger.info(f"Loaded module '{module_name}' from {path}")
        if USE_CHUNKS:
            logger.info(f"Using chunks for module '{module_name}' with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
        if USE_CHROMA:
            logger.info(f"Using ChromaDB for module '{module_name}' vector storage")
    except Exception as e:
        logger.error(f"Error loading module {module_name}: {e}")
    doc_processors[module_name] = processor

@app.route('/modules')
def modules_landing():
    # Landing page listing all modules with availability info
    module_list = []
    for name, _ in modules.items():
        proc = doc_processors.get(name)
        available = bool(proc and proc.sections)
        module_list.append({'name': name, 'available': available})
    return render_template('landing.html', module_list=module_list)

@app.route('/')
def home():
    # Determine selected module
    current_module = request.args.get('module', default_module)
    try:
        # Get the processor for this module
        doc_processor = doc_processors.get(current_module)
        # Get all sections and hierarchy
        sections = doc_processor.get_section_list()
        hierarchy = doc_processor.get_section_hierarchy()
        # Get the requested section or default
        current_section = request.args.get('section', sections[0] if sections else None)
        section_content, section_footnotes = doc_processor.get_section(current_section) if current_section else ("", [])
        # Get next and previous sections
        next_section = doc_processor.get_next_section(current_section)
        prev_section = doc_processor.get_previous_section(current_section)

        # Use the redesigned template
        template = 'index_redesign.html' if request.args.get('redesign', 'true').lower() == 'true' else 'index.html'

        # Render template with module information
        return render_template(template,
                               modules=modules.keys(),
                               current_module=current_module,
                               sections=sections,
                               hierarchy=hierarchy,
                               current_section=current_section,
                               section_content=section_content,
                               section_footnotes=section_footnotes,
                               next_section=next_section,
                               prev_section=prev_section,
                               error=None)
    except Exception as e:
        logger.error(f"Error in home route: {str(e)}")
        template = 'index_redesign.html' if request.args.get('redesign', 'true').lower() == 'true' else 'index.html'
        return render_template(template,
                               modules=modules.keys(),
                               current_module=current_module,
                               sections=[],
                               hierarchy=[],
                               current_section=None,
                               section_content=None,
                               section_footnotes=[],
                               next_section=None,
                               prev_section=None,
                               error=f"Error loading content: {str(e)}")

@app.route('/search')
def search():
    try:
        # Determine selected module
        current_module = request.args.get('module', default_module)
        doc_processor = doc_processors.get(current_module)

        # Log search request
        logger.info(f"Search request received. Module: {current_module}")
        logger.info(f"Request args: {dict(request.args)}")

        # Check if document processor exists
        if not doc_processor:
            logger.error(f"Document processor not found for module: {current_module}")
            return jsonify({
                'error': f"Module '{current_module}' not found",
                'results': [],
                'total': 0
            }), 404

        # Get search parameters
        query = request.args.get('q', '').strip()
        if not query:
            logger.warning("Empty search query received")
            return jsonify({
                'results': [],
                'total': 0,
                'page': 1,
                'per_page': 10,
                'total_pages': 0
            })

        # Get optional filters
        section_filter = request.args.get('section', None)
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))

        # Validate pagination parameters
        page = max(1, page)
        per_page = max(1, min(50, per_page))

        logger.info(f"Processing search: query='{query}', section_filter='{section_filter}', page={page}, per_page={per_page}")

        # Get all results (we'll paginate in memory)
        max_results = 100  # Reasonable limit for performance

        # Check if document processor has sections loaded
        if not hasattr(doc_processor, 'sections') or not doc_processor.sections:
            logger.error("Document processor has no sections loaded")
            return jsonify({
                'error': "Document content not available. Please try again later.",
                'results': [],
                'total': 0
            }), 500

        # Perform the search
        results = doc_processor.search(query, section_filter, max_results)

        # Log search results summary
        logger.info(f"Search for '{query}' returned {len(results)} results")

        # Calculate pagination
        total = len(results)
        total_pages = (total + per_page - 1) // per_page
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total)

        # Get page of results
        page_results = results[start_idx:end_idx]

        # Log pagination details
        logger.info(f"Pagination: page {page} of {total_pages}, showing results {start_idx+1}-{end_idx} of {total}")

        # Get section hierarchy for breadcrumb navigation
        hierarchy = doc_processor.get_section_hierarchy()
        section_hierarchy = {}

        # Flatten hierarchy for easy lookup
        def flatten_hierarchy(items, parent=None):
            for item in items:
                section_hierarchy[item['title']] = {
                    'parent': parent,
                    'level': item.get('level', 0)
                }
                if 'subsections' in item:
                    flatten_hierarchy(item['subsections'], item['title'])

        flatten_hierarchy(hierarchy)

        # Log section hierarchy size
        logger.info(f"Section hierarchy contains {len(section_hierarchy)} sections")

        # Return paginated results with metadata
        return jsonify({
            'results': page_results,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': total_pages,
            'section_hierarchy': section_hierarchy,
            'query': query  # Include the original query in the response
        })
    except Exception as e:
        logger.error(f"Error in search: {str(e)}", exc_info=True)  # Include full traceback
        return jsonify({
            'error': f"Search error: {str(e)}",
            'results': [],
            'total': 0
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Determine selected module
        current_module = data.get('module', default_module)
        doc_processor = doc_processors.get(current_module)

        # Determine chat mode
        mode = data.get('mode', 'chat')

        # Initialize session if needed
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
            logger.info(f"Created new session with user_id: {session['user_id']}")

        # Initialize chat histories in session if needed
        if 'chat_histories' not in session:
            session['chat_histories'] = {
                'chat': {},
                'learning': {}
            }

        # Initialize module-specific chat history if needed
        if current_module not in session['chat_histories'][mode]:
            session['chat_histories'][mode][current_module] = []

        # Get conversation history for this module and mode
        conversation_history = session['chat_histories'][mode][current_module]

        # Check for special commands
        if user_message.lower() == '/clear':
            session['chat_histories'][mode][current_module] = []
            return jsonify({'response': 'Conversation history cleared.', 'history_cleared': True})

        # Handle history request command
        if user_message.lower() == '/history':
            # Return the conversation history for this module and mode
            return jsonify({
                'response': f'Retrieved {len(conversation_history)} messages from history.',
                'history': conversation_history,
                'has_history': len(conversation_history) > 0
            })

        # Add user message to history
        conversation_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })

        # Check if the module has any sections
        section_list = doc_processor.get_section_list()
        if not section_list:
            logger.warning(f"No sections found in module '{current_module}'")
            section_content = f"No content is available for the '{current_module}' module yet."
        else:
            # Determine if we should use chunk-based or section-based retrieval
            if USE_CHUNKS and doc_processor.use_chunks:
                # Use chunk-based retrieval with hybrid search
                logger.info(f"Using chunk-based retrieval with {CONTEXT_CHUNK_COUNT} chunks and hybrid search={USE_HYBRID_SEARCH}")
                # Get context directly from chunks using hybrid search
                section_content = doc_processor.get_chunk_context(
                    query=user_message,
                    top_n=CONTEXT_CHUNK_COUNT,
                    hybrid_search=USE_HYBRID_SEARCH
                )
                if not section_content:
                    # Fall back to section-based retrieval if chunk retrieval fails
                    logger.warning("Chunk retrieval returned no content, falling back to section-based retrieval")
                    sections_to_use = doc_processor.get_relevant_sections_embedding(
                        query=user_message,
                        top_n=CONTEXT_SECTION_COUNT,
                        hybrid_search=USE_HYBRID_SEARCH
                    )
                    logger.info(f"Selected sections for context: {sections_to_use}")
                    section_content = ''
                    for sec in sections_to_use:
                        content, _ = doc_processor.get_section(sec)
                        section_content += f"\n=== {sec} ===\n" + content
            else:
                # Use section-based retrieval with hybrid search
                logger.info(f"Using section-based retrieval with {CONTEXT_SECTION_COUNT} sections and hybrid search={USE_HYBRID_SEARCH}")
                sections_to_use = doc_processor.get_relevant_sections_embedding(
                    query=user_message,
                    top_n=CONTEXT_SECTION_COUNT,
                    hybrid_search=USE_HYBRID_SEARCH
                )
                logger.info(f"Selected sections for context: {sections_to_use}")
                section_content = ''
                for sec in sections_to_use:
                    content, _ = doc_processor.get_section(sec)
                    section_content += f"\n=== {sec} ===\n" + content

        # Prepare system text based on mode
        if mode == 'learning':
            # Count the number of exchanges to determine where we are in the learning journey
            exchange_count = len(conversation_history) // 2
            conversation_stage = "beginning" if exchange_count <= 1 else "intermediate" if exchange_count <= 3 else "advanced"

            # Enhanced learning mode system prompt with more sophisticated strategies
            system_text = f"""You are a Real-Estate Development Learning Guide using the Socratic Method (the 5 R's: Receive, Reflect, Refine, Restate, Repeat).
A short summary of relevant document excerpts follows. Your task is to guide the student through critical thinking by asking questions—never simply hand them the answer.

Current conversation stage: {conversation_stage} (Exchange count: {exchange_count})
Adapt your approach accordingly - use simpler questions for beginning stages, and more complex analytical questions for advanced stages.

1.  Receive & Reflect
   • Begin by briefly acknowledging or paraphrasing the student's question to show you've listened.
   • Identify the key concepts in their question that relate to real estate development.
   • If this is their first question, warmly welcome them to the learning journey.

2.  Refine (Choose the most appropriate strategy based on the conversation stage)
   • FOUNDATIONAL QUESTIONS:
     - Ask the student where in the provided text they might find the relevant information.
     - Prompt them to quote or summarize a passage that could answer their question.
     - Ask: "What specific terms or concepts in the text seem most relevant to your question?"

   • ANALYTICAL QUESTIONS:
     - Challenge assumptions: "Why do you think the text describes it that way?"
     - Encourage connecting multiple sections: "How might Section X explain what you read in Section Y?"
     - Ask: "What might be the implications of this concept for real estate developers?"

   • SCENARIO-BASED QUESTIONS:
     - Offer a scenario based on the text: "Imagine you're a developer facing zoning limits—what questions would you ask to clarify FAR rules?"
     - Ask: "How would you apply this concept in a real-world development situation?"
     - Prompt: "If you were explaining this concept to a colleague, what key points would you emphasize?"

   • COMPARATIVE QUESTIONS:
     - Ask: "How does this concept compare to others we've discussed?"
     - Prompt: "What similarities and differences do you see between these related ideas?"
     - Ask: "How might different stakeholders view this concept differently?"

3.  Restate
   • Once they offer an answer or insight, ask them to restate it in their own words, citing the text.
   • Prompt: "That's an interesting perspective. Can you elaborate on how you arrived at that understanding?"
   • Ask: "How would you summarize what you've learned about this concept so far?"

4.  Repeat
   • If their restatement is incomplete, continue by asking another targeted question to deepen their reasoning.
   • Acknowledge progress: "You've made a good observation about X. Now let's explore Y..."
   • Build complexity gradually: Start with foundational questions, then move to analytical and scenario-based questions as the conversation progresses.

Constraints:
• Base every question solely on the RAG-provided text.
• Don't introduce new external facts—only use very brief, everyday analogies to illustrate text-based concepts.
• Your ultimate goal is to help the user arrive at understanding through guided inquiry.
• Adapt your approach based on the user's responses—if they're struggling, use simpler questions; if they're advancing quickly, increase complexity.
• Track their progress through the conversation and acknowledge their growth in understanding.

Now, begin by acknowledging their question.

A short summary of relevant document excerpts follows:
{section_content}
"""
        else:
            system_text = f"You are a helpful assistant. Use the following document section as context:\n{section_content}"

        # Authenticate and call Vertex AI
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        auth_req = GoogleAuthRequest()
        credentials.refresh(auth_req)
        token = credentials.token
        endpoint = (
            f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/"
            f"locations/{region}/publishers/google/models/gemini-2.0-flash:generateContent"
        )
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        # Prepare conversation history for the API
        contents = []

        # Include up to 10 most recent messages from history (to avoid context length issues)
        recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history

        # Add conversation history to contents
        for msg in recent_history[:-1]:  # Exclude the most recent user message
            contents.append({
                "role": msg["role"],
                "parts": [{"text": msg["content"]}]
            })

        # Add the current user message
        contents.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })

        # Prepare the payload with conversation history
        payload = {
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": contents
        }

        # Log the number of messages in the conversation
        logger.info(f"Sending {len(contents)} messages to Vertex AI")

        # Call Vertex AI
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        resp = response.json()
        candidates = resp.get("candidates", [])

        if candidates:
            part_list = candidates[0].get("content", {}).get("parts", [])
            ai_response_text = "".join([p.get("text", "") for p in part_list])

            # Add AI response to history
            conversation_history.append({
                'role': 'assistant',
                'content': ai_response_text,
                'timestamp': datetime.now().isoformat()
            })

            # Save updated history to session
            session['chat_histories'][mode][current_module] = conversation_history
            session.modified = True

            # Return response with history info
            return jsonify({
                'response': ai_response_text,
                'history_length': len(conversation_history),
                'has_history': True
            })
        else:
            ai_response_text = "I couldn't generate a response. Please try again."
            return jsonify({'response': ai_response_text, 'has_history': True})

    except Exception as e:
        logger.error(f"Error in chat route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/rebuild-index', methods=['POST', 'GET'])
def rebuild_index():
    """
    Rebuild the embedding index for a module with updated chunking parameters.
    This is useful for applying new chunking strategies without restarting the server.

    Supports both GET and POST methods:
    - GET: Parameters are passed as query string parameters
    - POST: Parameters are passed as JSON in the request body
    """
    try:
        # Handle both GET and POST methods
        if request.method == 'POST':
            # Get parameters from JSON body
            data = request.get_json() or {}
            module_name = data.get('module', default_module)
            chunk_size = int(data.get('chunk_size', DEFAULT_CHUNK_SIZE))
            chunk_overlap = int(data.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP))
            use_chunks = data.get('use_chunks', USE_CHUNKS)
            use_chroma = data.get('use_chroma', USE_CHROMA)
            force_rebuild = data.get('force_rebuild', True)
        else:  # GET method
            # Get parameters from query string
            module_name = request.args.get('module', default_module)
            chunk_size = int(request.args.get('chunk_size', DEFAULT_CHUNK_SIZE))
            chunk_overlap = int(request.args.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP))
            use_chunks = request.args.get('use_chunks', str(USE_CHUNKS)).lower() == 'true'
            use_chroma = request.args.get('use_chroma', str(USE_CHROMA)).lower() == 'true'
            force_rebuild = request.args.get('force_rebuild', 'true').lower() == 'true'

        # Log the rebuild request
        logger.info(f"Rebuilding index for module '{module_name}' with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, force_rebuild={force_rebuild}")

        # Check if we're rebuilding a specific module or all modules
        if module_name == 'all':
            # Rebuild all modules
            rebuilt_modules = []
            for name, processor in doc_processors.items():
                logger.info(f"Rebuilding index for module: {name}")
                processor.build_embedding_index(
                    use_chunks=use_chunks,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    use_chroma=use_chroma,
                    force_rebuild=force_rebuild
                )
                rebuilt_modules.append(name)

            return jsonify({
                'success': True,
                'message': f"Successfully rebuilt indices for {len(rebuilt_modules)} modules",
                'modules': rebuilt_modules,
                'parameters': {
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'use_chunks': use_chunks,
                    'use_chroma': use_chroma,
                    'force_rebuild': force_rebuild
                }
            })
        else:
            # Get the document processor for this module
            doc_processor = doc_processors.get(module_name)
            if not doc_processor:
                return jsonify({
                    'error': f"Module '{module_name}' not found",
                    'success': False
                }), 404

            # Rebuild the embedding index with the new parameters
            doc_processor.build_embedding_index(
                use_chunks=use_chunks,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                use_chroma=use_chroma,
                force_rebuild=force_rebuild
            )

            # Return success response
            return jsonify({
                'success': True,
                'message': f"Successfully rebuilt index for module '{module_name}'",
                'module': module_name,
                'parameters': {
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'use_chunks': use_chunks,
                    'use_chroma': use_chroma,
                    'force_rebuild': force_rebuild
                }
            })

    except Exception as e:
        logger.error(f"Error rebuilding index: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/chat-stream', methods=['POST', 'GET'])
def chat_stream():
    """Streaming version of the chat endpoint that uses Server-Sent Events (SSE)"""
    # Handle GET requests for SSE connection
    if request.method == 'GET':
        # Create a unique stream ID for this connection
        stream_id = str(uuid.uuid4())

        # Store the stream ID in the session
        session['current_stream_id'] = stream_id
        session.modified = True

        # Function to generate SSE events
        def generate_stream():
            try:
                # Send initial connection event
                yield f"data: {json.dumps({'type': 'connected', 'stream_id': stream_id})}\n\n"

                # Keep the connection alive with periodic pings
                ping_count = 0
                while ping_count < 100:  # Limit to prevent infinite loops
                    time.sleep(1)
                    yield f"data: {json.dumps({'type': 'ping', 'count': ping_count})}\n\n"
                    ping_count += 1

                    # Check if there's a response ready for this stream
                    response_key = f'stream_response_{stream_id}'
                    if response_key in session:
                        try:
                            # Get the stored response data
                            response_data = session.pop(response_key)
                            user_message = response_data.get('user_message', '')
                            ai_response_text = response_data.get('ai_response', '')
                            error_message = response_data.get('error', None)

                            # If there's an error message, send it and end the stream
                            if error_message:
                                yield f"data: {json.dumps({'type': 'error', 'content': error_message})}\n\n"
                                yield f"data: {json.dumps({'type': 'end'})}\n\n"
                                continue

                            # Send user message event
                            yield f"data: {json.dumps({'type': 'user', 'content': user_message})}\n\n"

                            # Send start event
                            yield f"data: {json.dumps({'type': 'start'})}\n\n"

                            # Stream the response in chunks
                            for i in range(0, len(ai_response_text), STREAM_CHUNK_SIZE):
                                chunk = ai_response_text[i:i+STREAM_CHUNK_SIZE]
                                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                                time.sleep(0.05)  # Small delay to simulate typing

                            # Send complete event
                            yield f"data: {json.dumps({'type': 'complete', 'content': ai_response_text})}\n\n"

                            # Send end event
                            yield f"data: {json.dumps({'type': 'end'})}\n\n"

                            # Reset ping count to keep connection alive for next potential response
                            ping_count = 0
                        except Exception as e:
                            logger.error(f"Error processing stream response: {str(e)}")
                            yield f"data: {json.dumps({'type': 'error', 'content': f'Error processing response: {str(e)}'})}\n\n"
                            yield f"data: {json.dumps({'type': 'end'})}\n\n"
            except Exception as e:
                logger.error(f"Error in stream generator: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'content': f'Stream error: {str(e)}'})}\n\n"
                yield f"data: {json.dumps({'type': 'end'})}\n\n"

        return Response(stream_with_context(generate_stream()), mimetype="text/event-stream")
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Determine selected module
        current_module = data.get('module', default_module)
        doc_processor = doc_processors.get(current_module)

        # Determine chat mode
        mode = data.get('mode', 'chat')

        # Initialize session if needed
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
            logger.info(f"Created new session with user_id: {session['user_id']}")

        # Initialize chat histories in session if needed
        if 'chat_histories' not in session:
            session['chat_histories'] = {
                'chat': {},
                'learning': {}
            }

        # Initialize module-specific chat history if needed
        if current_module not in session['chat_histories'][mode]:
            session['chat_histories'][mode][current_module] = []

        # Get conversation history for this module and mode
        conversation_history = session['chat_histories'][mode][current_module]

        # Check for special commands
        if user_message.lower() == '/clear':
            session['chat_histories'][mode][current_module] = []

            # Clear any stream-related data to prevent old context from being used
            stream_id = session.get('current_stream_id')
            if stream_id:
                response_key = f'stream_response_{stream_id}'
                if response_key in session:
                    session.pop(response_key)

            # Remove any other stream response keys that might be lingering
            keys_to_remove = []
            for key in session:
                if key.startswith('stream_response_'):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                session.pop(key)

            session.modified = True

            # For streaming, we'll send a special response
            stream_id = session.get('current_stream_id')
            if stream_id:
                response_key = f'stream_response_{stream_id}'
                session[response_key] = {
                    'user_message': user_message,
                    'ai_response': 'Conversation history cleared.',
                    'timestamp': datetime.now().isoformat(),
                    'history_cleared': True
                }
                session.modified = True

            return jsonify({'response': 'Conversation history cleared.', 'history_cleared': True})

        # Handle history request command
        if user_message.lower() == '/history':
            # Return the conversation history for this module and mode
            return jsonify({
                'response': f'Retrieved {len(conversation_history)} messages from history.',
                'history': conversation_history,
                'has_history': len(conversation_history) > 0
            })

        # Add user message to history
        # First check if this is a duplicate message (could happen with streaming)
        is_duplicate = False
        if conversation_history and len(conversation_history) > 0:
            last_msg = conversation_history[-1]
            if last_msg['role'] == 'user' and last_msg['content'] == user_message:
                is_duplicate = True

        if not is_duplicate:
            conversation_history.append({
                'role': 'user',
                'content': user_message,
                'timestamp': datetime.now().isoformat()
            })

        # Check if the module has any sections
        section_list = doc_processor.get_section_list()
        if not section_list:
            logger.warning(f"No sections found in module '{current_module}'")
            section_content = f"No content is available for the '{current_module}' module yet."
        else:
            # Determine if we should use chunk-based or section-based retrieval
            if USE_CHUNKS and doc_processor.use_chunks:
                # Use chunk-based retrieval with hybrid search
                logger.info(f"Using chunk-based retrieval with {CONTEXT_CHUNK_COUNT} chunks and hybrid search={USE_HYBRID_SEARCH}")
                # Get context directly from chunks using hybrid search
                section_content = doc_processor.get_chunk_context(
                    query=user_message,
                    top_n=CONTEXT_CHUNK_COUNT,
                    hybrid_search=USE_HYBRID_SEARCH
                )
                if not section_content:
                    # Fall back to section-based retrieval if chunk retrieval fails
                    logger.warning("Chunk retrieval returned no content, falling back to section-based retrieval")
                    sections_to_use = doc_processor.get_relevant_sections_embedding(
                        query=user_message,
                        top_n=CONTEXT_SECTION_COUNT,
                        hybrid_search=USE_HYBRID_SEARCH
                    )
                    logger.info(f"Selected sections for context: {sections_to_use}")
                    section_content = ''
                    for sec in sections_to_use:
                        content, _ = doc_processor.get_section(sec)
                        section_content += f"\n=== {sec} ===\n" + content
            else:
                # Use section-based retrieval with hybrid search
                logger.info(f"Using section-based retrieval with {CONTEXT_SECTION_COUNT} sections and hybrid search={USE_HYBRID_SEARCH}")
                sections_to_use = doc_processor.get_relevant_sections_embedding(
                    query=user_message,
                    top_n=CONTEXT_SECTION_COUNT,
                    hybrid_search=USE_HYBRID_SEARCH
                )
                logger.info(f"Selected sections for context: {sections_to_use}")
                section_content = ''
                for sec in sections_to_use:
                    content, _ = doc_processor.get_section(sec)
                    section_content += f"\n=== {sec} ===\n" + content

        # Prepare system text based on mode
        if mode == 'learning':
            # Count the number of exchanges to determine where we are in the learning journey
            exchange_count = len(conversation_history) // 2
            conversation_stage = "beginning" if exchange_count <= 1 else "intermediate" if exchange_count <= 3 else "advanced"

            # Enhanced learning mode system prompt with more sophisticated strategies
            system_text = f"""You are a Real-Estate Development Learning Guide using the Socratic Method (the 5 R's: Receive, Reflect, Refine, Restate, Repeat).
A short summary of relevant document excerpts follows. Your task is to guide the student through critical thinking by asking questions—never simply hand them the answer.

Current conversation stage: {conversation_stage} (Exchange count: {exchange_count})
Adapt your approach accordingly - use simpler questions for beginning stages, and more complex analytical questions for advanced stages.

1.  Receive & Reflect
   • Begin by briefly acknowledging or paraphrasing the student's question to show you've listened.
   • Identify the key concepts in their question that relate to real estate development.
   • If this is their first question, warmly welcome them to the learning journey.

2.  Refine (Choose the most appropriate strategy based on the conversation stage)
   • FOUNDATIONAL QUESTIONS:
     - Ask the student where in the provided text they might find the relevant information.
     - Prompt them to quote or summarize a passage that could answer their question.
     - Ask: "What specific terms or concepts in the text seem most relevant to your question?"

   • ANALYTICAL QUESTIONS:
     - Challenge assumptions: "Why do you think the text describes it that way?"
     - Encourage connecting multiple sections: "How might Section X explain what you read in Section Y?"
     - Ask: "What might be the implications of this concept for real estate developers?"

   • SCENARIO-BASED QUESTIONS:
     - Offer a scenario based on the text: "Imagine you're a developer facing zoning limits—what questions would you ask to clarify FAR rules?"
     - Ask: "How would you apply this concept in a real-world development situation?"
     - Prompt: "If you were explaining this concept to a colleague, what key points would you emphasize?"

   • COMPARATIVE QUESTIONS:
     - Ask: "How does this concept compare to others we've discussed?"
     - Prompt: "What similarities and differences do you see between these related ideas?"
     - Ask: "How might different stakeholders view this concept differently?"

3.  Restate
   • Once they offer an answer or insight, ask them to restate it in their own words, citing the text.
   • Prompt: "That's an interesting perspective. Can you elaborate on how you arrived at that understanding?"
   • Ask: "How would you summarize what you've learned about this concept so far?"

4.  Repeat
   • If their restatement is incomplete, continue by asking another targeted question to deepen their reasoning.
   • Acknowledge progress: "You've made a good observation about X. Now let's explore Y..."
   • Build complexity gradually: Start with foundational questions, then move to analytical and scenario-based questions as the conversation progresses.

Constraints:
• Base every question solely on the RAG-provided text.
• Don't introduce new external facts—only use very brief, everyday analogies to illustrate text-based concepts.
• Your ultimate goal is to help the user arrive at understanding through guided inquiry.
• Adapt your approach based on the user's responses—if they're struggling, use simpler questions; if they're advancing quickly, increase complexity.
• Track their progress through the conversation and acknowledge their growth in understanding.

Now, begin by acknowledging their question.

A short summary of relevant document excerpts follows:
{section_content}
"""
        else:
            system_text = f"You are a helpful assistant. Use the following document section as context:\n{section_content}"

        # Check network connectivity before making API calls
        try:
            # Try to connect to Google's servers to check connectivity
            import socket
            socket.create_connection(("oauth2.googleapis.com", 443), timeout=5)
            logger.info("Network connectivity check passed")
        except (socket.timeout, socket.error) as e:
            logger.error(f"Network connectivity check failed: {str(e)}")
            error_message = "Network connectivity issue detected. Please check your internet connection and try again."

            # Store the error in the session for the stream to pick up
            stream_id = session.get('current_stream_id')
            if stream_id:
                response_key = f'stream_response_{stream_id}'
                session[response_key] = {
                    'user_message': user_message,
                    'error': error_message,
                    'timestamp': datetime.now().isoformat()
                }
                session.modified = True

            return jsonify({'status': 'error', 'message': error_message})

        try:
            # Authenticate and call Vertex AI
            credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            auth_req = GoogleAuthRequest()
            credentials.refresh(auth_req)
            token = credentials.token
            endpoint = (
                f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/"
                f"locations/{region}/publishers/google/models/gemini-2.0-flash:generateContent"
            )
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

            # Prepare conversation history for the API
            contents = []

            # Include up to 10 most recent messages from history (to avoid context length issues)
            recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history

            # Add conversation history to contents
            for msg in recent_history[:-1]:  # Exclude the most recent user message
                contents.append({
                    "role": msg["role"],
                    "parts": [{"text": msg["content"]}]
                })

            # Add the current user message
            contents.append({
                "role": "user",
                "parts": [{"text": user_message}]
            })

            # Prepare the payload with conversation history
            payload = {
                "systemInstruction": {"parts": [{"text": system_text}]},
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.7,
                    "topP": 0.95,
                    "topK": 40
                }
            }

            # Log the number of messages in the conversation
            logger.info(f"Sending {len(contents)} messages to Vertex AI for streaming")

            # Call Vertex AI with timeout
            response = requests.post(endpoint, headers=headers, json=payload, stream=False, timeout=30)
            response.raise_for_status()
            resp = response.json()
            candidates = resp.get("candidates", [])

            if candidates:
                part_list = candidates[0].get("content", {}).get("parts", [])
                ai_response_text = "".join([p.get("text", "") for p in part_list])

                # Add user message to history if not already there
                user_msg_in_history = False
                for msg in conversation_history:
                    if msg['role'] == 'user' and msg['content'] == user_message:
                        user_msg_in_history = True
                        break

                if not user_msg_in_history:
                    conversation_history.append({
                        'role': 'user',
                        'content': user_message,
                        'timestamp': datetime.now().isoformat()
                    })

                # Add AI response to history
                conversation_history.append({
                    'role': 'assistant',
                    'content': ai_response_text,
                    'timestamp': datetime.now().isoformat()
                })

                # Save updated history to session
                session['chat_histories'][mode][current_module] = conversation_history

                # Get the current stream ID from the session
                stream_id = session.get('current_stream_id')
                if stream_id:
                    # Store the response for the stream to pick up
                    response_key = f'stream_response_{stream_id}'
                    session[response_key] = {
                        'user_message': user_message,
                        'ai_response': ai_response_text,
                        'timestamp': datetime.now().isoformat()
                    }

                session.modified = True

                # Return success response
                return jsonify({'status': 'success', 'message': 'Response ready for streaming'})
            else:
                error_message = "I couldn't generate a response. Please try again."

                # Store the error in the session for the stream to pick up
                stream_id = session.get('current_stream_id')
                if stream_id:
                    response_key = f'stream_response_{stream_id}'
                    session[response_key] = {
                        'user_message': user_message,
                        'error': error_message,
                        'timestamp': datetime.now().isoformat()
                    }
                    session.modified = True

                return jsonify({'status': 'error', 'message': error_message})

        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {str(e)}")
            error_message = f"Error communicating with AI service: {str(e)}"

            # Store the error in the session for the stream to pick up
            stream_id = session.get('current_stream_id')
            if stream_id:
                response_key = f'stream_response_{stream_id}'
                session[response_key] = {
                    'user_message': user_message,
                    'error': error_message,
                    'timestamp': datetime.now().isoformat()
                }
                session.modified = True

            return jsonify({'status': 'error', 'message': error_message})

    except Exception as e:
        logger.error(f"Error in chat-stream route: {str(e)}")
        if request.method == 'GET':
            return Response(
                f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n",
                mimetype="text/event-stream"
            )
        else:
            # Store the error in the session for the stream to pick up
            stream_id = session.get('current_stream_id')
            if stream_id:
                response_key = f'stream_response_{stream_id}'
                session[response_key] = {
                    'user_message': user_message if 'user_message' in locals() else "Unknown message",
                    'error': f"An unexpected error occurred: {str(e)}",
                    'timestamp': datetime.now().isoformat()
                }
                session.modified = True

            return jsonify({'status': 'error', 'message': str(e)})

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear the conversation history for a specific mode and module"""
    try:
        data = request.get_json()
        mode = data.get('mode', 'chat')
        module = data.get('module', default_module)

        # Initialize chat_histories if it doesn't exist
        if 'chat_histories' not in session:
            session['chat_histories'] = {}

        # Initialize mode if it doesn't exist
        if mode not in session['chat_histories']:
            session['chat_histories'][mode] = {}

        # Clear history for the specified module
        session['chat_histories'][mode][module] = []

        # Clear any stream-related data to prevent old context from being used
        stream_id = session.get('current_stream_id')
        if stream_id:
            response_key = f'stream_response_{stream_id}'
            if response_key in session:
                session.pop(response_key)

        # Remove any other stream response keys that might be lingering
        keys_to_remove = []
        for key in session:
            if key.startswith('stream_response_'):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            session.pop(key)

        session.modified = True

        return jsonify({'status': 'success', 'message': 'Conversation history cleared'})
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

# The rebuild-index route is now defined above with POST method

if __name__ == '__main__':
    app.run(debug=True)