from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from utils.document_processor import DocumentProcessor
import logging
from google.cloud import aiplatform
import requests
from google.auth import default
from google.auth.transport.requests import Request as GoogleAuthRequest

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

app = Flask(__name__)

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
        processor.build_embedding_index()
        logger.info(f"Loaded module '{module_name}' from {path}")
    except Exception as e:
        logger.error(f"Error loading module {module_name}: {e}")
    doc_processors[module_name] = processor

@app.route('/modules')
def modules_landing():
    # Landing page listing all modules with availability info
    module_list = []
    for name, path in modules.items():
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
        # Render template with module information
        return render_template('index.html',
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
        return render_template('index.html',
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
        query = request.args.get('q', '').strip()
        if not query:
            return jsonify([])
        results = doc_processor.search(query)
        return jsonify([{
            'section': section,
            'context': context,
            'full_text': full_text
        } for section, full_text, context in results])
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
        # Always use embedding-based retrieval
        env_count = os.getenv('CONTEXT_SECTION_COUNT')
        if env_count and env_count.isdigit():
            top_n = int(env_count)
        else:
            top_n = len(doc_processor.get_section_list())
        sections_to_use = doc_processor.get_relevant_sections_embedding(user_message, top_n=top_n)
        logger.info(f"Selected sections for context: {sections_to_use}")
        section_content = ''
        for sec in sections_to_use:
            content, _ = doc_processor.get_section(sec)
            section_content += f"\n=== {sec} ===\n" + content
        # Determine chat mode
        mode = data.get('mode', 'chat')
        if mode == 'learning':
            system_text = f"""You are a Real-Estate Development Learning Guide using the Socratic Method (the 5 R's: Receive, Reflect, Refine, Restate, Repeat).
A short summary of relevant document excerpts follows. Your task is to guide the student through critical thinking by asking questions—never simply hand them the answer.

1.  Receive & Reflect
   • Begin by briefly acknowledging or paraphrasing the student's question to show you've listened.
2.  Refine
   • Ask the student where in the provided text they might find the relevant information.
   • Prompt them to quote or summarize a passage that could answer their question.
   • Challenge assumptions: "Why do you think the text describes it that way?"
   • Encourage connecting multiple sections: "How might Section 3 explain what you read in Section 1?"
   • Offer a tiny scenario based on the text: "Imagine you're a developer facing zoning limits—what questions would you ask to clarify FAR rules?"
3.  Restate
   • Once they offer an answer or insight, ask them to restate it in their own words, citing the text.
4.  Repeat
   • If their restatement is incomplete, continue by asking another targeted question to deepen their reasoning.

Constraints:
•  Base every question solely on the RAG-provided text.
•  Don't introduce new external facts—only use very brief, everyday analogies to illustrate text-based concepts.
•  Your ultimate goal is to help the user arrive at understanding through guided inquiry.

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
        payload = {
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": [
                {"role": "user", "parts": [{"text": user_message}]}  
            ]
        }
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        resp = response.json()
        candidates = resp.get("candidates", [])
        if candidates:
            part_list = candidates[0].get("content", {}).get("parts", [])
            ai_response_text = "".join([p.get("text", "") for p in part_list])
        else:
            ai_response_text = ""
        return jsonify({'response': ai_response_text})
    except Exception as e:
        logger.error(f"Error in chat route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 