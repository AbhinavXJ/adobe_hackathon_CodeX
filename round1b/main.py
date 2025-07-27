import json
import datetime
import os
import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import time
from collections import Counter
import statistics
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

warnings.filterwarnings("ignore")

class PersonaDrivenAnalyzer:
    def __init__(self):
        print("Loading Persona-Driven Document Intelligence...")
        start_time = time.time()
        
        try:
            # Force CPU-only mode
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            self.stop_words = set(stopwords.words('english'))
            
            load_time = time.time() - start_time
            print(f"Intelligence loaded in {load_time:.2f}s")
            
        except Exception as e:
            print(f"Error loading system: {e}")
            raise e

    def extract_text_with_formatting(self, pdf_path):
        """Extract text with formatting information - optimized for speed"""
        pages_data = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Limit pages for performance (60s constraint)
                max_pages = min(20, len(pdf.pages))
                
                for page_num in range(max_pages):
                    page = pdf.pages[page_num]
                    
                    # Fast text extraction
                    page_text = page.extract_text()
                    if not page_text or len(page_text) < 30:
                        continue
                    
                    # Simple line splitting instead of complex char analysis
                    lines = [line.strip() for line in page_text.split('\n') if line.strip()]
                    
                    if lines:
                        pages_data[page_num + 1] = {
                            'lines': [{'text': line} for line in lines],
                            'full_text': page_text
                        }
                        
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return {"pages": {}}
        
        return {"pages": pages_data}

    def extract_section_title(self, page_data):
        """Extract section title - fast heuristic approach"""
        if 'lines' not in page_data:
            text = page_data.get('full_text', '')
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            return lines[0] if lines else "Content"
        
        lines_data = page_data['lines']
        
        if not lines_data:
            return "Content"
        
        # Simple heuristic: first non-empty line under 100 chars
        for line in lines_data[:5]:
            text = line['text'].strip()
            if 5 <= len(text) <= 100 and len(text.split()) <= 15:
                return text
        
        return lines_data[0]['text'][:100] if lines_data else "Content"

    def generate_analysis_queries(self, persona, job_description):
        """Generate targeted queries - simplified for speed"""
        try:
            # Primary query
            queries = [f"{persona} {job_description}"]
            
            # Extract key terms quickly
            combined_text = f"{persona} {job_description}".lower()
            tokens = word_tokenize(combined_text)
            
            # Simple keyword extraction
            important_words = [word for word in tokens 
                             if len(word) > 3 and word not in self.stop_words][:6]
            
            # Create additional queries
            for i in range(0, min(len(important_words), 4), 2):
                if i + 1 < len(important_words):
                    queries.append(f"{important_words[i]} {important_words[i+1]}")
            
            return queries[:4]  # Limit for speed
            
        except Exception as e:
            return [f"{persona} {job_description}"]

    def calculate_semantic_relevance(self, page_content, queries, persona, job_description):
        """Calculate semantic relevance - optimized for speed"""
        try:
            # Truncate content for speed
            content_sample = page_content[:1000]  # First 1000 chars only
            
            # Primary semantic matching
            job_text = f"{persona} {job_description}"
            
            # Single embedding calculation
            embeddings = self.model.encode([job_text, content_sample])
            job_embedding = embeddings[0:1]
            content_embedding = embeddings[1:2]
            
            similarity = cosine_similarity(job_embedding, content_embedding)[0][0]
            
            # Simple keyword matching boost
            job_keywords = set(job_text.lower().split())
            content_keywords = set(content_sample.lower().split())
            keyword_overlap = len(job_keywords.intersection(content_keywords))
            keyword_boost = min(keyword_overlap * 0.05, 0.3)
            
            final_score = (similarity + keyword_boost) * 100
            return min(final_score, 100)
            
        except Exception as e:
            print(f"Error in semantic relevance: {e}")
            return 0

    def create_refined_text(self, page_content, queries):
        """Create refined text - simplified for speed"""
        try:
            # Simple sentence-based refinement
            sentences = sent_tokenize(page_content)[:10]  # First 10 sentences only
            
            if not sentences:
                return page_content[:500]
            
            # Score sentences based on query terms
            query_terms = set()
            for query in queries[:2]:  # Only use first 2 queries
                query_terms.update(query.lower().split())
            
            scored_sentences = []
            for sentence in sentences:
                sentence_terms = set(sentence.lower().split())
                overlap = len(query_terms.intersection(sentence_terms))
                scored_sentences.append((sentence, overlap))
            
            # Sort by relevance and take top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            
            refined_text = ""
            for sentence, score in scored_sentences:
                if len(refined_text) + len(sentence) + 1 <= 500:
                    refined_text += sentence + " "
                else:
                    break
            
            return refined_text.strip()
            
        except Exception as e:
            return page_content[:500]

    def analyze_document_collection(self, document_name, content, persona, job_to_be_done):
        """Analyze document collection - optimized for speed"""
        extracted_sections = []
        sub_section_analysis = []
        
        queries = self.generate_analysis_queries(persona, job_to_be_done)
        
        # Process maximum 15 pages per document for speed
        processed_pages = 0
        for page_num, page_data in content["pages"].items():
            processed_pages += 1
            if processed_pages > 15:  # Speed limit
                break
            
            page_content = page_data.get('full_text', '')
            if not page_content or len(page_content) < 30:
                continue
            
            relevance_score = self.calculate_semantic_relevance(
                page_content, queries, persona, job_to_be_done
            )
            
            # Lower threshold for more results
            if relevance_score > 10:
                section_title = self.extract_section_title(page_data)
                
                extracted_sections.append({
                    "document": document_name,
                    "section_title": section_title,
                    "importance_score": relevance_score,
                    "page_number": page_num
                })
                
                refined_text = self.create_refined_text(page_content, queries)
                
                sub_section_analysis.append({
                    "document": document_name,
                    "refined_text": refined_text,
                    "page_number": page_num
                })
        
        # Sort and rank
        extracted_sections.sort(key=lambda x: x["importance_score"], reverse=True)
        
        for i, section in enumerate(extracted_sections):
            section["importance_rank"] = i + 1
            del section["importance_score"]
        
        return extracted_sections, sub_section_analysis

def read_input_json(input_path):
    """Read and parse input JSON"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        persona = data.get("persona", "")
        job_to_be_done = data.get("job_to_be_done", "")
        
        if isinstance(persona, dict):
            persona = persona.get('role', '') or list(persona.values())[0] if persona else ""
        if isinstance(job_to_be_done, dict):
            job_to_be_done = job_to_be_done.get('task', '') or list(job_to_be_done.values())[0] if job_to_be_done else ""
        
        return str(persona), str(job_to_be_done)
        
    except Exception as e:
        print(f"Error reading input file: {e}")
        return None, None

def process_collection(input_dir, output_dir):
    """Process document collection with timing constraints"""
    print("Adobe Hackathon Round 1B - Persona-Driven Document Intelligence")
    print("=" * 70)
    
    overall_start = time.time()
    
    try:
        analyzer = PersonaDrivenAnalyzer()
    except Exception as e:
        print(f"Failed to initialize analyzer: {e}")
        return False
    
    input_json_path = os.path.join(input_dir, "input.json")
    if not os.path.exists(input_json_path):
        print(f"input.json not found in {input_dir}")
        return False
    
    persona, job_to_be_done = read_input_json(input_json_path)
    if not persona or not job_to_be_done:
        print("Failed to read persona and job_to_be_done")
        return False
    
    print(f"Persona: {persona}")
    print(f"Job: {job_to_be_done}")
    
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found")
        return False
    
    # Limit to 5 documents for 60s constraint
    pdf_files = pdf_files[:5]
    print(f"Processing {len(pdf_files)} PDF files")
    
    all_extracted_sections = []
    all_sub_section_analysis = []
    input_doc_names = []
    
    for filename in pdf_files:
        # Check time constraint
        elapsed = time.time() - overall_start
        if elapsed > 50:  # Leave 10s buffer
            print(f"Time limit approaching, stopping at {filename}")
            break
        
        pdf_path = os.path.join(input_dir, filename)
        print(f"Analyzing: {filename}")
        
        doc_start = time.time()
        content = analyzer.extract_text_with_formatting(pdf_path)
        
        if content["pages"]:
            input_doc_names.append(filename)
            sections, sub_sections = analyzer.analyze_document_collection(
                filename, content, persona, job_to_be_done
            )
            
            all_extracted_sections.extend(sections)
            all_sub_section_analysis.extend(sub_sections)
            
            doc_time = time.time() - doc_start
            print(f"   Found {len(sections)} relevant sections ({doc_time:.1f}s)")
        else:
            print(f"   No extractable content")
    
    if not all_extracted_sections:
        print("No relevant content found")
        return False
    
    # Get top 5 sections
    all_extracted_sections.sort(key=lambda x: x["importance_rank"])
    top_5_sections = all_extracted_sections[:5]
    
    for i, section in enumerate(top_5_sections):
        section["importance_rank"] = i + 1
    
    top_5_keys = {(s["document"], s["page_number"]) for s in top_5_sections}
    top_5_subsections = [s for s in all_sub_section_analysis 
                        if (s["document"], s["page_number"]) in top_5_keys]
    
    output_json = {
        "metadata": {
            "input_documents": input_doc_names,
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": top_5_sections,
        "subsection_analysis": top_5_subsections
    }
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output.json")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)
        
        total_time = time.time() - overall_start
        print(f"Analysis complete! Output saved to: {output_path}")
        print(f"Total processing time: {total_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"Error saving output: {e}")
        return False

def main():
    """Main function for Docker execution"""
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return
    
    success = process_collection(input_dir, output_dir)
    
    if success:
        print("Round 1B processing completed successfully!")
    else:
        print("Round 1B processing failed!")

if __name__ == "__main__":
    main()
