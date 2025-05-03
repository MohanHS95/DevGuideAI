from docx import Document
import os
import logging
import re
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from urllib.parse import quote_plus

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example references for demonstration purposes
FOOTNOTE_REFERENCES = {
    '1': 'Reference to comprehensive land use planning guidelines.',
    '2': 'Details on the Comprehensive Plan or Master Plan.',
    '3': 'Information on zoning regulations and their impact.',
    '4': 'Study on sustainable development practices.',
    '5': 'Analysis of community engagement in planning.',
    # Add more references as needed
}

class DocumentProcessor:
    def __init__(self, doc_path, module_name=None):
        self.doc_path = doc_path
        self.module_name = module_name
        self.document = None
        self.sections = {}
        self.section_order = []
        self.section_footnotes = {}  # Store footnotes for each section
        self.glossary_terms = {}  # Store glossary terms and their definitions
        # Embedding-based retrieval attributes
        self.emb_model = None
        self.section_embeddings = None  # numpy array of section embeddings
        self.section_titles = []        # list of section titles matching embeddings
        self.section_embedding_norms = None  # precomputed norms for cosine similarity

    def load_document(self):
        """Load the Word document."""
        if not os.path.exists(self.doc_path):
            raise FileNotFoundError(f"Document not found at {self.doc_path}")
        
        logger.info(f"Loading document from {self.doc_path}")
        self.document = Document(self.doc_path)
        return self

    def process_footnotes(self, text: str, section: str) -> Tuple[str, List[Dict]]:
        """
        Process footnotes in text and return the processed text and footnotes.
        Targets explicit markers, numbers directly following words, 
        or numbers directly following a period.
        """
        footnotes = []
        # Restore the pattern including period+number: |(\.)(\d+)(?=[\s.,:;!?]|$) 
        footnote_pattern = r'(?<!>)(?:(\[(\d+)\])|(\[footnote-(\d+)\])|(\b\w+)(\d+)(?=[\s.,:;!?]|$)|(\.)(\d+)(?=[\s.,:;!?]|$))'

        processed_parts = []
        last_end = 0

        for match in re.finditer(footnote_pattern, text):
            start, end = match.span()
            # Append text before the match
            processed_parts.append(text[last_end:start])

            footnote_num = None
            original_content = ""
            
            # Determine which pattern matched
            if match.group(2): # [1]
                footnote_num = match.group(2)
            elif match.group(4): # [footnote-1]
                footnote_num = match.group(4)
            elif match.group(6): # word1
                original_content = match.group(5)
                footnote_num = match.group(6)
            elif match.group(8): # .1 -> Restored match group
                original_content = match.group(7) # The period
                footnote_num = match.group(8)

            if footnote_num:
                # Create footnote data
                footnote_id = f"{section}-footnote-{footnote_num}"
                ref_id = f"{section}-ref-{footnote_num}"
                footnotes.append({
                    'id': footnote_id,
                    'number': footnote_num,
                    'ref_id': ref_id,
                })
                # Append original content (word or period) + the footnote link
                processed_parts.append(f'{original_content}<sup><a href="#{footnote_id}" id="{ref_id}" class="footnote-link">{footnote_num}</a></sup>')
            else:
                # If somehow no number was extracted, append the matched text as is
                processed_parts.append(match.group(0))

            last_end = end
        
        # Append any remaining text after the last match
        processed_parts.append(text[last_end:])
        
        processed_text = "".join(processed_parts)
        return processed_text, footnotes

    def _process_glossary_entries(self, content_lines):
        """Process glossary entries and store them."""
        # Make sure this logic correctly handles raw lines from the doc
        self.glossary_terms = {}
        current_term = None
        current_def = []
        for line in content_lines:
            if ':' in line and len(line.split(':', 1)[0]) < 50: # Basic check for term: definition
                # Finalize previous term if any
                if current_term and current_def:
                    self.glossary_terms[current_term] = ' '.join(current_def).strip()
                    logger.info(f"Stored glossary term: {current_term}")
                
                term, definition_part = line.split(':', 1)
                current_term = term.strip()
                current_def = [definition_part.strip()]
            elif current_term:
                 # Append to current definition if it seems like a continuation
                 current_def.append(line)
        # Store the last term
        if current_term and current_def:
             self.glossary_terms[current_term] = ' '.join(current_def).strip()
             logger.info(f"Stored glossary term: {current_term}")
        logger.info(f"Processed {len(self.glossary_terms)} glossary terms.")

    def process_document(self):
        """Process the document and extract sections with footnotes (NO CLEANING)."""
        if not self.document:
            raise ValueError("Document not loaded. Call load_document() first.")

        # Reset state
        self.sections = {}
        self.section_order = []
        self.section_footnotes = {}
        self.glossary_terms = {}
        
        current_section = None # Start with no section
        current_content_lines = []
        in_glossary_section = False
        is_works_cited_section = False

        logger.info("Processing document paragraphs (NO CLEANING)...")

        for paragraph in self.document.paragraphs:
            raw_text = paragraph.text.strip()
            # Use raw_text for all logic now
            text_to_process = raw_text 
            
            # Style-based heading detection (more reliable than bold)
            style_name = paragraph.style.name
            is_heading = style_name.startswith('Heading') or style_name == 'Title'
            heading_level = 0
            if is_heading:
                try:
                    heading_level = int(style_name.split(' ')[-1])
                except:
                    heading_level = 1 # Default for Title or unnumbered Heading

            # Explicit section name checks for special handling
            is_glossary_heading = is_heading and raw_text == "Glossary of Terms"
            is_works_cited_heading = is_heading and raw_text == "Works cited"

            # --- Section Transition Logic --- 
            if is_heading:
                # Process the *previous* section's content before starting a new one
                if current_section is not None and current_content_lines:
                    content = '\n'.join(current_content_lines)
                    if is_works_cited_section: 
                        self.sections[current_section] = content
                        self.section_footnotes[current_section] = []
                    elif in_glossary_section:
                        # Process glossary content using the dedicated method
                        self._process_glossary_entries(current_content_lines)
                        self.sections[current_section] = content # Store raw text for glossary display
                        self.section_footnotes[current_section] = []
                    else:
                        # Process regular section content for footnotes
                        processed_content, footnotes = self.process_footnotes(content, current_section)
                        self.sections[current_section] = processed_content
                        self.section_footnotes[current_section] = footnotes
                    # Add previous section to order only after processing
                    if current_section not in self.section_order: 
                       self.section_order.append(current_section)
                
                # Start the new section
                current_section = raw_text # Use the actual heading text
                current_content_lines = [] # Reset content for the new section
                in_glossary_section = is_glossary_heading
                is_works_cited_section = is_works_cited_heading
                
                # Don't add the heading itself to the content lines
                continue 
            
            # --- Content Accumulation --- 
            # Add non-empty paragraph text to the current section's content lines
            if current_section is not None and text_to_process:
                current_content_lines.append(text_to_process)

        # --- Process the VERY LAST section --- 
        if current_section is not None and current_content_lines:
            content = '\n'.join(current_content_lines)
            if is_works_cited_section:
                self.sections[current_section] = content
                self.section_footnotes[current_section] = []
            elif in_glossary_section:
                self._process_glossary_entries(current_content_lines)
                self.sections[current_section] = content
                self.section_footnotes[current_section] = []
            else:
                processed_content, footnotes = self.process_footnotes(content, current_section)
                self.sections[current_section] = processed_content
                self.section_footnotes[current_section] = footnotes
             # Add the last section to the order
            if current_section not in self.section_order: 
                self.section_order.append(current_section)

        # --- Post-Processing --- 
        # Add glossary links (check exclusion logic again)
        if self.glossary_terms:
            self._add_glossary_links()

        logger.info(f"Processing complete. Found {len(self.sections)} sections: {self.section_order}")
        logger.info(f"Glossary terms found: {len(self.glossary_terms)}")
        return self.sections

    def _add_glossary_links(self):
        """Add links to glossary terms in all sections except the glossary itself."""
        for section_name, content in self.sections.items():
            if section_name != "Glossary of Terms":
                for term in sorted(self.glossary_terms.keys(), key=len, reverse=True):
                    # Create a regex pattern that matches the term as a whole word
                    pattern = r'\b' + re.escape(term) + r'\b'
                    # Generate URL parameters
                    mod = quote_plus(self.module_name) if self.module_name else ''
                    sec = quote_plus("Glossary of Terms")
                    term_quoted = quote_plus(term)
                    replacement = (
                        f'<a href="/?module={mod}&section={sec}#{term_quoted}" '
                        f'class="glossary-link" title="{self.glossary_terms[term]}">{term}</a>'
                    )
                    content = re.sub(pattern, replacement, content)
                self.sections[section_name] = content

    def get_section(self, section_name):
        """Get content and footnotes for a specific section."""
        content = self.sections.get(section_name, "")
        footnotes = self.section_footnotes.get(section_name, [])
        return content, footnotes

    def get_all_sections(self):
        """Get all sections and their content."""
        return self.sections

    def get_section_list(self):
        """Get a list of all section titles."""
        return self.section_order

    def get_section_hierarchy(self) -> List[Dict]:
        """Get the section hierarchy for the table of contents."""
        hierarchy = []
        current_main_section = None

        for section in self.section_order:
            if section.startswith('Section'):
                current_main_section = {
                    'title': section,
                    'subsections': []
                }
                hierarchy.append(current_main_section)
            elif current_main_section is not None:
                current_main_section['subsections'].append(section)

        return hierarchy

    def search(self, query: str) -> List[Tuple[str, str, str]]:
        """Search for text in all sections."""
        results = []
        query = query.lower()
        
        for section_title, content in self.sections.items():
            # Remove HTML tags for searching
            clean_content = re.sub(r'<[^>]+>', '', content)
            paragraphs = clean_content.split('\n')
            
            for paragraph in paragraphs:
                if query in paragraph.lower():
                    start = max(0, paragraph.lower().find(query) - 50)
                    end = min(len(paragraph), paragraph.lower().find(query) + len(query) + 50)
                    context = f"...{paragraph[start:end]}..."
                    results.append((section_title, paragraph, context))
        
        return results

    def get_next_section(self, current_section: str) -> str:
        """Get the next section title."""
        try:
            current_index = self.section_order.index(current_section)
            if current_index < len(self.section_order) - 1:
                return self.section_order[current_index + 1]
        except ValueError:
            pass
        return None

    def get_previous_section(self, current_section: str) -> str:
        """Get the previous section title."""
        try:
            current_index = self.section_order.index(current_section)
            if current_index > 0:
                return self.section_order[current_index - 1]
        except ValueError:
            pass
        return None

    def get_relevant_sections(self, query: str, top_n: int = 1) -> List[str]:
        """Return the top_n section titles whose content best matches the query using simple keyword count."""
        q = query.lower()
        scores = []
        for title, content in self.sections.items():
            clean_content = re.sub(r'<[^>]+>', '', content).lower()
            count = clean_content.count(q)
            scores.append((count, title))
        # Sort by descending count
        scores.sort(reverse=True)
        # Select titles with at least one match
        relevant = [title for count, title in scores if count > 0]
        # Fallback to first section if no matches found
        if not relevant and self.section_order:
            return [self.section_order[0]]
        return relevant[:top_n]

    def build_embedding_index(self, model_name: str = "all-MiniLM-L6-v2"):
        """Build embeddings for each section using a SentenceTransformer."""
        # Clean section texts from HTML
        texts = [re.sub(r'<[^>]+>', '', content) for content in self.sections.values()]
        # Store section titles in same order as texts
        self.section_titles = list(self.sections.keys())
        # Load embedding model
        self.emb_model = SentenceTransformer(model_name)
        # Compute embeddings
        embeddings = self.emb_model.encode(texts, show_progress_bar=False)
        # Store embeddings and precompute norms for cosine similarity
        self.section_embeddings = np.array(embeddings)
        self.section_embedding_norms = np.linalg.norm(self.section_embeddings, axis=1)

    def get_relevant_sections_embedding(self, query: str, top_n: int = 1) -> List[str]:
        """Return the top_n section titles most similar to the query via cosine similarity."""
        if self.section_embeddings is None or self.emb_model is None:
            raise ValueError("Embedding index not built. Call build_embedding_index() first.")
        # Compute query embedding
        q_emb = self.emb_model.encode([query], show_progress_bar=False)[0]
        # Compute cosine similarities
        q_norm = np.linalg.norm(q_emb)
        sims = np.dot(self.section_embeddings, q_emb) / (self.section_embedding_norms * q_norm + 1e-10)
        # Get top indices by descending similarity
        top_indices = np.argsort(sims)[::-1][:top_n]
        return [self.section_titles[i] for i in top_indices] 