import os
import json
import glob
from pathlib import Path
import pdfplumber
import collections
import traceback

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
        except Exception as e:
            print(f"DEBUG: Error in header/footer detection: {e}")
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
            except Exception as e:
                print(f"DEBUG: Error in title extraction: {e}")
                continue
    except Exception as e:
        print(f"DEBUG: Error in get_document_title: {e}")
    
    return title.strip()

def debug_model_loading():
    """Comprehensive model loading debug"""
    print("\n" + "="*50)
    print("🔍 MODEL LOADING DEBUG")
    print("="*50)
    
    model_path = "./models/supervised_heading_classifier.pkl"
    
    # 1. Check file system
    print(f"1. File path: {model_path}")
    print(f"   File exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        print(f"   File size: {os.path.getsize(model_path)} bytes")
        print(f"   File readable: {os.access(model_path, os.R_OK)}")
        print(f"   File permissions: {oct(os.stat(model_path).st_mode)[-3:]}")
    
    # 2. Check directory contents
    models_dir = "./models"
    if os.path.exists(models_dir):
        print(f"2. Models directory contents:")
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            size = os.path.getsize(item_path) if os.path.isfile(item_path) else "DIR"
            print(f"   - {item} ({size} bytes)")
    
    # 3. Test direct joblib loading
    print("3. Testing direct joblib loading:")
    try:
        import joblib
        print(f"   joblib version: {joblib.__version__}")
        model_data = joblib.load(model_path)
        print(f"   ✅ Direct joblib load SUCCESS")
        print(f"   Model type: {type(model_data)}")
        if hasattr(model_data, 'keys') and callable(model_data.keys):
            print(f"   Model keys: {list(model_data.keys())}")
        return True, model_data
    except Exception as e:
        print(f"   ❌ Direct joblib load FAILED: {e}")
        print(f"   Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False, None

def process_pdf(pdf_path):
    """Process a single PDF and return structured data"""
    print(f"\n📖 Processing: {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Initialize classifier for ALL files
            try:
                from train_supervised import SupervisedHeadingClassifier
                classifier = SupervisedHeadingClassifier()
                
                # Load model for ALL files
                model_path = "./models/supervised_heading_classifier.pkl"
                
                # Try direct loading first
                model_loaded_directly, model_data = debug_model_loading()
                
                if model_loaded_directly:
                    print("✅ Model loaded successfully!")
                    classifier.model = model_data
                    classifier.heading_classifier = model_data['heading_classifier']
                    classifier.level_classifier = model_data['level_classifier']
                    classifier.scaler = model_data['scaler']
                    classifier.level_encoder = model_data['level_encoder']
                    classifier.is_trained = model_data.get('is_trained', True)
                else:
                    # Fallback to standard loading
                    model_loaded = classifier.load_model(model_path)
                    if not model_loaded:
                        print(f"❌ Failed to load model for {pdf_path}")
                        return {"title": get_document_title(pdf), "outline": []}
                
                # Process ALL files with full pipeline
                headers_footers_to_ignore = identify_recurring_headers_footers(pdf)
                title = get_document_title(pdf)
                
                from utils import extract_lines_from_pdf
                all_lines, avg_font_size, page_height = extract_lines_from_pdf(pdf_path)
                
                if not all_lines:
                    print(f"⚠️ No lines extracted from {pdf_path}")
                    return {"title": title, "outline": []}
                
                print(f"📊 Extracted {len(all_lines)} lines for analysis")
                
                # Get predictions
                predictions = classifier.predict(all_lines, avg_font_size, page_height)
                
                if not predictions:
                    print(f"⚠️ No predictions generated for {pdf_path}")
                    return {"title": title, "outline": []}
                
                # Debug predictions
                heading_candidates = 0
                low_confidence_headings = 0
                
                outline = []
                for idx, (line, (is_heading, confidence, level)) in enumerate(zip(all_lines, predictions)):
                    text = line.get("text", "").strip()
                    
                    # Debug output for first few lines
                    if idx < 5:
                        print(f"DEBUG Line {idx}: '{text[:50]}...' | Heading: {is_heading} | Conf: {confidence:.3f} | Level: {level}")
                    
                    # Lower confidence threshold and add more debugging
                    if is_heading and confidence > 0.3:  # Lowered from 0.5 to 0.3
                        heading_candidates += 1
                        page_num = line.get("page_number", 1)
                        
                        # More lenient filtering
                        if (text not in headers_footers_to_ignore and 
                            len(text) > 2 and  # Reduced from 3 to 2
                            text.strip()):     # Just ensure non-empty
                            
                            # Use default level if None
                            final_level = level if level is not None else "H3"
                            
                            outline.append({
                                "level": final_level,
                                "text": text,
                                "page": page_num
                            })
                            
                            print(f"✅ Added heading: '{text[:60]}...' (Level: {final_level}, Conf: {confidence:.3f})")
                    
                    elif is_heading and confidence <= 0.3:
                        low_confidence_headings += 1
                
                print(f"📈 Analysis Results:")
                print(f"   - Total lines: {len(all_lines)}")
                print(f"   - Heading candidates (>0.3 conf): {heading_candidates}")
                print(f"   - Low confidence headings (≤0.3): {low_confidence_headings}")
                print(f"   - Final headings: {len(outline)}")
                print(f"   - Headers/footers ignored: {len(headers_footers_to_ignore)}")
                
                return {"title": title, "outline": outline}
                
            except Exception as classifier_error:
                print(f"❌ Classifier error for {pdf_path}: {classifier_error}")
                import traceback
                traceback.print_exc()
                return {"title": get_document_title(pdf), "outline": []}
                
    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
        import traceback
        traceback.print_exc()
        return {"title": f"Error processing {os.path.basename(pdf_path)}", "outline": []}

def main():
    # """Main function for Docker execution"""
    # input_dir = "/app/input"
    # output_dir = "/app/output"
    # With these:
    input_dir = "./input"         # Local input directory
    output_dir = "./output"       # Local output directory
    
    print("🚀 Adobe Hackathon Round 1A - PDF Heading Detection")
    print("="*60)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all PDF files in input directory
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    
    print(f"📚 Found {len(pdf_files)} PDF files to process")
    
    if not pdf_files:
        print("❌ No PDF files found in input directory!")
        return
    
    for pdf_path in pdf_files:
        pdf_filename = Path(pdf_path).stem
        output_path = os.path.join(output_dir, f"{pdf_filename}.json")
        
        # Process PDF
        result = process_pdf(pdf_path)
        
        # Save result as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Output saved: {output_path}")

if __name__ == "__main__":
    main()
