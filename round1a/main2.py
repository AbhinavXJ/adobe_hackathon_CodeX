import os
import json
import collections
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import statistics

import pdfplumber
from pdfplumber.page import Page

from utils import extract_lines_from_pdf
from train_supervised import SupervisedHeadingClassifier

# --- Configuration ---
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
MODEL_FOLDER = "models"

def identify_recurring_headers_footers(pdf: pdfplumber.PDF, recurrence_threshold: float = 0.3) -> Set[str]:
    """Identifies headers and footers"""
    line_counts = collections.defaultdict(int)
    margin_height = 0.15

    if len(pdf.pages) < 3:
        return set()

    for page in pdf.pages:
        header_boundary = page.height * margin_height
        footer_boundary = page.height * (1 - margin_height)

        words_in_margins = [
            w for w in page.extract_words(x_tolerance=2, y_tolerance=2, extra_attrs=["size", "x0", "x1", "top"])
            if w['top'] < header_boundary or w['top'] > footer_boundary
        ]
        
        for word in words_in_margins:
            word['page_number'] = page.page_number
        
        if not words_in_margins:
            continue
        
        from utils import group_words_into_lines
        lines_in_margins = group_words_into_lines(words_in_margins)

        for line in lines_in_margins:
            text = line.get("text", "").strip()
            if text and len(text) > 2:
                line_counts[text] += 1
    
    recurring_elements = {
        text for text, count in line_counts.items()
        if (count / len(pdf.pages)) >= recurrence_threshold
    }
    
    return recurring_elements

def get_document_title(pdf: pdfplumber.PDF, max_pages_to_check: int = 2) -> str:
    """Get document title"""
    title = "Untitled Document"
    max_font_size = 0
    
    for i, page in enumerate(pdf.pages):
        if i >= max_pages_to_check:
            break
            
        words = page.extract_words(extra_attrs=["size", "x0", "x1", "top"])
        if not words:
            continue

        for word in words:
            word['page_number'] = page.page_number

        from utils import group_words_into_lines
        lines = group_words_into_lines(words)
        if not lines:
            continue

        current_page_max_size = max((line.get('size', 0) for line in lines), default=0)

        if current_page_max_size > max_font_size:
            max_font_size = current_page_max_size
            title_lines = [line for line in lines if line.get('size') == max_font_size]
            title = ' '.join(line['text'] for line in title_lines)

    return title.strip()

def extract_structure_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """Main extraction function with supervised ML"""
    print(f"Processing: {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            classifier = SupervisedHeadingClassifier()
            
            # Load supervised model
            script_dir = Path(__file__).parent
            model_dir = script_dir / MODEL_FOLDER
            model_path = model_dir / "supervised_heading_classifier.pkl"
            
            model_loaded = classifier.load_model(str(model_path))
            if not model_loaded:
                print("  - No trained model found. Please run train_supervised.py first!")
                return {"title": "No Model Available", "outline": []}
            
            print("  - Loaded supervised ML model")
            
            headers_footers_to_ignore = identify_recurring_headers_footers(pdf)
            if headers_footers_to_ignore:
                print(f"  - Identified recurring headers/footers to ignore: {headers_footers_to_ignore}")

            title = get_document_title(pdf)
            
            # Extract all lines
            all_lines, avg_font_size, page_height = extract_lines_from_pdf(pdf_path)
            
            if not all_lines:
                return {"title": title, "outline": []}

            # Predict headings using supervised model
            predictions = classifier.predict(all_lines, avg_font_size, page_height)
            
            print(f"  - ML model made {len(predictions)} predictions")

            # Build outline
            outline = []
            for idx, (line, (is_heading, confidence, level)) in enumerate(zip(all_lines, predictions)):
                if is_heading and confidence > 0.5:  # Confidence threshold
                    text = line.get("text", "").strip()
                    page_num = line.get("page_number", 1)
                    
                    if (text not in headers_footers_to_ignore and 
                        len(text) > 3 and 
                        level is not None):
                        outline.append({
                            "level": level,
                            "text": text,
                            "page": page_num,
                            "font_size": line.get("size", 0),
                            "confidence": round(confidence, 3),
                            "is_bold": line.get("is_bold", False),
                            "is_italic": line.get("is_italic", False)
                        })
            
            print(f"  - Final outline contains {len(outline)} headings")
            return {"title": title, "outline": outline}

    except Exception as e:
        print(f"  - Error processing {pdf_path}: {e}")
        return {"title": f"Error processing {os.path.basename(pdf_path)}", "outline": []}

def main():
    """Main function"""
    script_dir = Path(__file__).parent
    input_dir = script_dir / INPUT_FOLDER
    output_dir = script_dir / OUTPUT_FOLDER

    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    print(f"Input folder: {input_dir.resolve()}")
    print(f"Output folder: {output_dir.resolve()}")
    print("-" * 50)

    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in the '{INPUT_FOLDER}' directory.")
        return

    for pdf_path in pdf_files:
        structured_data = extract_structure_from_pdf(pdf_path)
        
        output_filename = pdf_path.stem + "_structure.json"
        output_path = output_dir / output_filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=4, ensure_ascii=False)
            print(f"  -> Successfully saved structure to {output_path}")
        except Exception as e:
            print(f"  - Error saving JSON for {pdf_path.name}: {e}")
        
        print("-" * 50)

    print("Processing complete.")

if __name__ == '__main__':
    main()
