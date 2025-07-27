import os
import json
import glob
from pathlib import Path
import pdfplumber
import collections

def identify_recurring_headers_footers(pdf, recurrence_threshold=0.3):
    """Identifies headers and footers to ignore"""
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
        
        try:
            from utils import group_words_into_lines
            lines_in_margins = group_words_into_lines(words_in_margins)
            
            for line in lines_in_margins:
                text = line.get("text", "").strip()
                if text and len(text) > 2:
                    line_counts[text] += 1
        except Exception:
            continue
    
    recurring_elements = {
        text for text, count in line_counts.items()
        if (count / len(pdf.pages)) >= recurrence_threshold
    }
    
    return recurring_elements

def get_document_title(pdf, max_pages_to_check=2):
    """Extract document title"""
    title = "Untitled Document"
    max_font_size = 0
    
    try:
        for i, page in enumerate(pdf.pages):
            if i >= max_pages_to_check:
                break
            
            words = page.extract_words(extra_attrs=["size", "x0", "x1", "top"])
            if not words:
                continue
            
            for word in words:
                word['page_number'] = page.page_number
            
            try:
                from utils import group_words_into_lines
                lines = group_words_into_lines(words)
                
                if not lines:
                    continue
                
                current_page_max_size = max((line.get('size', 0) for line in lines), default=0)
                
                if current_page_max_size > max_font_size:
                    max_font_size = current_page_max_size
                    title_lines = [line for line in lines if line.get('size') == max_font_size]
                    title = ' '.join(line['text'] for line in title_lines)
            except Exception:
                continue
    except Exception:
        pass
    
    return title.strip()

def load_model_silently():
    """Load model without debug output"""
    model_path = "/app/models/supervised_heading_classifier.pkl"
    
    try:
        import joblib
        model_data = joblib.load(model_path)
        return True, model_data
    except Exception:
        return False, None

def process_pdf(pdf_path):
    """Process a single PDF and return structured data"""
    print(f" Processing: {os.path.basename(pdf_path)}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            try:
                from train_supervised import SupervisedHeadingClassifier
                classifier = SupervisedHeadingClassifier()
                
                # Load model silently
                model_loaded_directly, model_data = load_model_silently()
                
                if model_loaded_directly:
                    classifier.model = model_data
                    classifier.heading_classifier = model_data['heading_classifier']
                    classifier.level_classifier = model_data['level_classifier']
                    classifier.scaler = model_data['scaler']
                    classifier.level_encoder = model_data['level_encoder']
                    classifier.is_trained = model_data.get('is_trained', True)
                else:
                    return {"title": get_document_title(pdf), "outline": []}
                
                # Process document
                headers_footers_to_ignore = identify_recurring_headers_footers(pdf)
                title = get_document_title(pdf)
                
                from utils import extract_lines_from_pdf
                all_lines, avg_font_size, page_height = extract_lines_from_pdf(pdf_path)
                
                if not all_lines:
                    return {"title": title, "outline": []}
                
                # Get predictions
                predictions = classifier.predict(all_lines, avg_font_size, page_height)
                
                if not predictions:
                    return {"title": title, "outline": []}
                
                # Process results
                outline = []
                for line, (is_heading, confidence, level) in zip(all_lines, predictions):
                    text = line.get("text", "").strip()
                    
                    if (is_heading and confidence > 0.3 and 
                        text not in headers_footers_to_ignore and 
                        len(text) > 2 and text.strip()):
                        
                        page_num = line.get("page_number", 1)
                        final_level = level if level is not None else "H3"
                        
                        outline.append({
                            "level": final_level,
                            "text": text,
                            "page": page_num
                        })
                
                print(f" Output saved: {os.path.basename(pdf_path).replace('.pdf', '.json')}")
                return {"title": title, "outline": outline}
                
            except Exception:
                return {"title": get_document_title(pdf), "outline": []}
                
    except Exception:
        return {"title": f"Error processing {os.path.basename(pdf_path)}", "outline": []}

def main():
    """Main function for Docker execution"""
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    print(" Adobe Hackathon Round 1A - PDF Heading Detection")
    print("="*60)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all PDF files in input directory
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    
    if not pdf_files:
        print(" No PDF files found in input directory!")
        return
    
    print(f" Found {len(pdf_files)} PDF files to process\n")
    
    for pdf_path in pdf_files:
        pdf_filename = Path(pdf_path).stem
        output_path = os.path.join(output_dir, f"{pdf_filename}.json")
        
        # Process PDF
        result = process_pdf(pdf_path)
        
        # Save result as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n Processing complete! All results saved to /app/output")

if __name__ == "__main__":
    main()
