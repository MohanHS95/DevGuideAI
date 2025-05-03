# DevGuide AI (Web Platform): MVP Implementation Plan - Zoning Module + Core Features

## Project Goal
To build a Minimum Viable Product (MVP) of the "DevGuide AI" as a web platform. The MVP will initially focus on the "Navigating Land Use, Zoning, and Development Planning in New York City" content. Key features include displaying structured learning content, providing a persistent RAG-powered Socratic AI guide, and laying the groundwork for future sections and testing.

## Core Technical Capabilities
* Web application structure (Front-end and potentially a Python back-end).
* Displaying structured text content in a web browser.
* Reading text content from a document file on the back-end.
* Sending document content (context) and user queries to an LLM API from the back-end.
* Crafting LLM prompts for guided, Socratic responses.
* Implementing a conversational chat interface in the web UI.
* (Future) Handling user progress and test logic.
* (Future) AI logic for adaptive test difficulty.

## Knowledge Base
The primary knowledge base for the initial module is the uploaded document: "NYC Real Estate Development Curriculum - Navigating Land Use, Zoning, and Development Planning in New York City.docx". The system will be designed to accommodate additional documents for other curriculum sections later.

## Phase 1: Web Application Structure and Content Display

**Goal:** Set up a basic web application structure and display the content of the Zoning document in a user-friendly format in a web browser.

* **Step 1.1: Project Setup and Basic Web Server**
    * **Action:** Create a new project in Cursor. Set up a minimal web server using a Python framework like Flask (recommended for simplicity for MVP). Create basic HTML templates.
    * **Using Cursor:** Ask Cursor to generate a basic Flask web application structure with one route that renders an HTML template. Ask for guidance on setting up a virtual environment.
    * **Output:** A running basic web server displaying a simple page.

* **Step 1.2: Reading and Preparing Content for Web Display**
    * **Goal:** Read the Zoning document content on the back-end and prepare it for display in HTML, preserving structure (like headings and paragraphs).
    * **Action:** Integrate the document reading logic (similar to Phase 1.2 in the previous plan, potentially using `python-docx`) into your Flask application. Process the text to convert headings and paragraphs into HTML-friendly format (e.g., using Markdown conversion if your source is Markdown, or simple string processing).
    * **Using Cursor:** Ask Cursor how to read a `.docx` file in Python using `python-docx`. Ask how to format plain text with headings and paragraphs into basic HTML `<p>` and `<h1>`/`<h2>` tags.
    * **Output:** Python function that reads the document and returns formatted text/HTML.

* **Step 1.3: Displaying Content in HTML Template**
    * **Goal:** Render the processed content in your web page.
    * **Action:** Modify your Flask route to pass the processed content to the HTML template. Update the HTML template to display this content.
    * **Using Cursor:** Ask Cursor how to pass a variable from a Flask route to an HTML template and how to display raw HTML content within a Jinja (Flask's default) template.
    * **Output:** A web page displaying the content of your Zoning document with basic formatting.

## Phase 2: Integrating the Core AI Guide (Chat Interface)

**Goal:** Add a chat interface to the web page and connect it to the LLM for the RAG-powered Socratic guide.

* **Step 2.1: Set Up LLM API Access (Back-end)**
    * **Goal:** Ensure LLM API access is configured securely on your back-end (Flask application).
    * **Action:** Transfer the LLM API key setup (environment variables) and library installation (from Phase 2.1 of the previous plan) to your Flask project environment.
    * **Using Cursor:** Ask for guidance on securely managing API keys in a Flask application using environment variables.
    * **Output:** LLM library installed and API key configured for the Flask app.

* **Step 2.2: Implement Chat Interface (Front-end HTML/JavaScript)**
    * **Goal:** Add a simple chat box, send button, and message display area to your HTML page using basic HTML and JavaScript.
    * **Action:** Write HTML for input field and display area. Write JavaScript to send the user's input to your Flask back-end when the button is clicked and display responses received.
    * **Using Cursor:** Ask Cursor to generate basic HTML for a chat interface (input box, button, message area). Ask for basic JavaScript to send text from an input field to a specific URL endpoint when a button is clicked.
    * **Output:** A functional but basic chat interface on your web page.

* **Step 2.3: Create API Endpoint for AI Interaction (Back-end Flask)**
    * **Goal:** Create a Flask route that receives chat messages from the front-end, processes them with the LLM, and sends the response back.
    * **Action:** Create a new Flask route (e.g., `/chat`). This route should receive the user's message. Inside this route, retrieve the relevant knowledge base content (or a summary/embedding - see Phase 3). Call the LLM API with the user's message, the Socratic prompt, and the knowledge base context. Return the LLM's response.
    * **Using Cursor:** Ask Cursor to create a Flask route that accepts a POST request with JSON data (the user message). Ask how to call your chosen LLM API function from within this route and return a JSON response.

* **Step 2.4: Integrate Socratic Prompt and Context (Back-end)**
    * **Goal:** Refine the back-end logic to include the Socratic prompt and the knowledge base content in the LLM call, and manage conversation history.
    * **Action:** Use the refined Socratic prompt (from Phase 3.1 of the previous plan). Modify the LLM call in the `/chat` route to include this prompt and the document content. Implement basic storage for conversation history for a single session (e.g., in a server-side list or session object) and include it in prompts.
    * **Using Cursor:** Ask Cursor how to include a system message (for the Socratic role), user messages (with history), and context (document content) in your LLM API calls using your chosen library. Ask how to manage a simple conversation history list in a Flask application.
    * **Output:** A working chat interface where the AI responds based on the document using a Socratic style.

## Phase 3: Enhancing Knowledge Retrieval (RAG)

**Goal:** Implement a basic RAG system to efficiently provide relevant context from the knowledge base to the LLM, especially as the knowledge base grows.

* **Step 3.1: (Simple) Section Retrieval**
    * **Goal:** Instead of sending the *entire* document, identify the most relevant section(s) based on the user's query.
    * **Action:** Structure your knowledge base document internally (e.g., as a dictionary mapping section titles to text). When a user asks a question, perform a simple keyword search or prompt the LLM to identify the relevant section title from your document structure. Send only that section's text to the LLM along with the query.
    * **Using Cursor:** Ask Cursor how to store text data in a Python dictionary with keys as section titles. Ask for Python code to perform a keyword search within the keys/values of the dictionary based on a user query.

* **Step 3.2: (More Advanced) Embedding-Based Retrieval**
    * **Goal:** Use vector embeddings to find semantically similar chunks of text from the knowledge base.
    * **Action:** This step involves creating embeddings for chunks of your document, potentially storing them in a simple in-memory structure or a lightweight vector database, and using embedding similarity search to find relevant chunks at query time. *Note: This adds significant complexity. Consider this a V1.1 feature.*
    * **Using Cursor:** Ask Cursor about libraries for creating text embeddings and performing similarity search in Python (e.g., using libraries like `sentence-transformers` and `faiss` or integrated LLM embedding models).

## Phase 4: Structuring for Multiple Sections

**Goal:** Modify the application structure to easily incorporate other curriculum sections.

* **Step 4.1: Modularize Knowledge Loading**
    * **Goal:** Make it easy to load different knowledge base documents or sections based on user selection (future feature).
    * **Action:** Refactor your code so that the document reading and processing logic is reusable and can be easily switched to load content for "Financing," "Construction," etc., from different files.
    * **Using Cursor:** Ask Cursor how to write a Python function that takes a filename as input and returns the processed text content.

* **Step 4.2: Basic Section Navigation (Front-end)**
    * **Goal:** Add simple links or buttons on the web page to represent different curriculum sections (even if only Zoning is active initially).
    * **Action:** Add basic HTML to list the curriculum sections (Zoning, Financing, etc.). These links won't fully work yet, but they create the structure.
    * **Using Cursor:** Ask Cursor to generate an HTML unordered list (`<ul>`) with list items (`<li>`) for each of your curriculum section titles.

## Phase 5: (Future) Testing Module and Adaptive Difficulty

**Goal:** Implement quizzes for each section and add AI logic to adjust difficulty.

* **Step 5.1: Create Quiz Content Structure**
    * **Goal:** Define a format for storing quiz questions and answers related to each section.
    * **Action:** Create a separate file or structure (e.g., a JSON file or Python dictionary) to hold questions (multiple choice, short answer) for the Zoning section.
    * **Manual Effort:** Writing the actual quiz questions based on the Zoning document.
    * **Output:** A structured set of quiz questions for the Zoning section.

* **Step 5.2: Implement Quiz Display and Grading (Front-end & Back-end)**
    * **Goal:** Add a page or section to the web app to display quizzes and check user answers.
    * **Action:** Create Flask routes and HTML templates to display quiz questions. Write JavaScript to collect user answers. Write back-end Python code to receive answers, compare them to correct answers, and calculate a score.
    * **Using Cursor:** Ask Cursor to generate HTML for a simple multiple-choice quiz form. Ask for a Flask route to receive form data and process it in Python.

* **Step 5.3: Implement AI Adaptive Difficulty**
    * **Goal:** Use AI to analyze user performance and adjust the difficulty or type of subsequent questions or learning interactions.
    * **Action:** Based on quiz scores or how the user interacts with the AI guide (e.g., needing lots of hints), prompt the LLM to suggest the user review certain topics again, or generate follow-up questions that are harder or simpler. *Note: This is a complex AI integration point.*
    * **Using Cursor:** Ask Cursor for ideas on prompting an LLM to suggest next learning steps based on a user's performance on a quiz covering specific topics from the knowledge base.

## MVP Definition (Revised)

The MVP is a web-based Python application (using Flask) that displays the content of the "Navigating Zoning, Land Use, and Development Planning" document and provides a persistent RAG-powered, Socratic chat interface allowing the user to ask questions about the document content and receive guiding responses. The application structure will be set up to allow for the addition of other curriculum sections and future testing/adaptive features.

This revised plan gives us a roadmap for building the web platform version of your DevGuide AI. It requires more steps involving both front-end and back-end code, but it aligns with your vision for a more interactive and structured learning experience.

Which phase and step would you like to start with? I recommend starting with **Phase 1: Setting Up and Reading the Knowledge Base** to get the core content loaded, likely starting with **Step 1.1** (Project Setup and Basic Web Server).

