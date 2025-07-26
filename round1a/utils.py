import json
import re
import numpy as np
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path
import pdfplumber
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# --- Configuration ---
LINE_TOLERANCE = 2
CHAR_TOLERANCE = 1.5

def group_words_into_lines(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Groups word objects into lines"""
    if not words:
        return []

    words.sort(key=lambda w: (w['top'], w['x0']))
    
    lines = []
    current_line_words = [words[0]]

    for word in words[1:]:
        last_word = current_line_words[-1]
        
        if abs(word['top'] - last_word['top']) > LINE_TOLERANCE:
            lines.append(build_line_from_words(current_line_words))
            current_line_words = [word]
        else:
            current_line_words.append(word)

    if current_line_words:
        lines.append(build_line_from_words(current_line_words))
    
    return lines

def build_line_from_words(line_words: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Builds line from words with formatting information"""
    if not line_words:
        return {}

    text = line_words[0]['text']
    for i in range(1, len(line_words)):
        prev_word = line_words[i-1]
        curr_word = line_words[i]
        
        if curr_word['x0'] - prev_word['x1'] > CHAR_TOLERANCE:
            text += ' '
        text += curr_word['text']

    font_name = line_words[0].get('fontname', '')
    is_bold = 'bold' in font_name.lower() or 'black' in font_name.lower()
    is_italic = 'italic' in font_name.lower() or 'oblique' in font_name.lower()

    return {
        "text": text,
        "size": line_words[0]["size"],
        "page_number": line_words[0].get("page_number", 1),
        "fontname": font_name,
        "is_bold": is_bold,
        "is_italic": is_italic,
        "x0": line_words[0].get("x0", 0),
        "top": line_words[0].get("top", 0)
    }

def extract_lines_from_pdf(pdf_path: str) -> Tuple[List[Dict], float, float]:
    """Extract all lines from PDF with metadata"""
    all_lines = []
    all_font_sizes = []
    page_height = 800
    
    with pdfplumber.open(pdf_path) as pdf:
        if pdf.pages:
            page_height = pdf.pages[0].height
            
        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["size", "top", "x0", "x1", "fontname"])
            for word in words:
                word['page_number'] = page.page_number
                all_font_sizes.append(word.get("size", 0))
            
            if words:
                page_lines = group_words_into_lines(words)
                all_lines.extend(page_lines)
    
    avg_font_size = np.mean(all_font_sizes) if all_font_sizes else 12.0
    return all_lines, avg_font_size, page_height

def load_expected_outputs(expected_dir: Path) -> Dict[str, List[Dict]]:
    """Load all expected output files"""
    expected_data = {}
    
    for file_path in expected_dir.glob("e-file*.json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Extract file number (e.g., "04" from "e-file04.json")
            file_num = file_path.stem.replace('e-file', '')
            expected_data[file_num] = data.get('outline', [])
    
    return expected_data

def text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity"""
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    # Exact match
    if text1 == text2:
        return 1.0
    
    # Substring match
    if text1 in text2 or text2 in text1:
        return 0.8
    
    # Word overlap
    words1 = set(text1.split())
    words2 = set(text2.split())
    overlap = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return overlap / union if union > 0 else 0.0

def find_best_matching_expected_heading(line_text: str, expected_headings: List[Dict]) -> Tuple[str, float]:
    """Find the best matching expected heading"""
    best_match = None
    best_score = 0.0
    
    for heading in expected_headings:
        expected_text = heading.get('text', '')
        similarity = text_similarity(line_text, expected_text)
        
        if similarity > best_score:
            best_score = similarity
            best_match = heading.get('level', 'H3')
    
    return best_match, best_score
