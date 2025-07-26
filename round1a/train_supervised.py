import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from typing import List, Dict, Tuple
import statistics

from utils import extract_lines_from_pdf, load_expected_outputs, find_best_matching_expected_heading

INPUT_FOLDER = "input"
EXPECTED_FOLDER = "expected"  
MODEL_FOLDER = "models"

class SupervisedHeadingClassifier:
    def __init__(self):
        self.heading_classifier = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15)
        self.level_classifier = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.level_encoder = LabelEncoder()
        self.is_trained = False
        
    def extract_features(self, line: Dict, all_lines: List[Dict], line_idx: int, avg_font_size: float, page_height: float) -> np.ndarray:
        """Extract comprehensive features for a line"""
        text = line.get("text", "").strip()
        font_size = line.get("size", 0)
        
        # Basic features
        font_size_ratio = font_size / avg_font_size if avg_font_size > 0 else 1.0
        text_length = len(text)
        word_count = len(text.split())
        
        # Pattern features
        ends_with_colon = 1 if text.endswith(':') else 0
        is_title_case = 1 if text.istitle() else 0
        is_uppercase = 1 if text.isupper() and len(text) > 2 else 0
        starts_with_number = 1 if text and text[0].isdigit() else 0
        contains_appendix = 1 if 'appendix' in text.lower() else 0
        
        # Formatting features
        is_bold = 1 if line.get("is_bold", False) else 0
        is_italic = 1 if line.get("is_italic", False) else 0
        
        # Position features
        position_y_normalized = line.get("top", 0) / page_height if page_height > 0 else 0
        
        # Context features
        next_line_size_ratio = 1.0
        prev_line_size_ratio = 1.0
        line_spacing_above = 0
        line_spacing_below = 0
        
        if line_idx > 0:
            prev_line = all_lines[line_idx - 1]
            prev_font_size = prev_line.get("size", font_size)
            prev_line_size_ratio = font_size / prev_font_size if prev_font_size > 0 else 1.0
            line_spacing_above = abs(line.get("top", 0) - prev_line.get("top", 0))
        
        if line_idx < len(all_lines) - 1:
            next_line = all_lines[line_idx + 1]
            next_font_size = next_line.get("size", font_size)
            next_line_size_ratio = font_size / next_font_size if next_font_size > 0 else 1.0
            line_spacing_below = abs(next_line.get("top", 0) - line.get("top", 0))
        
        # Advanced pattern features
        has_punctuation = 1 if any(char in text for char in '.,;!?()[]{}') else 0
        is_short_line = 1 if len(text) < 80 else 0
        font_size_percentile = self.get_font_size_percentile(font_size, all_lines)
        
        return np.array([
            font_size_ratio, text_length, word_count, ends_with_colon, is_title_case,
            is_uppercase, starts_with_number, contains_appendix, is_bold, is_italic,
            position_y_normalized, next_line_size_ratio, prev_line_size_ratio,
            line_spacing_above, line_spacing_below, has_punctuation, is_short_line,
            font_size_percentile
        ])
    
    def get_font_size_percentile(self, font_size: float, all_lines: List[Dict]) -> float:
        """Get font size percentile within document"""
        font_sizes = [line.get('size', 0) for line in all_lines]
        if not font_sizes:
            return 0.5
        
        smaller_count = sum(1 for fs in font_sizes if fs < font_size)
        return smaller_count / len(font_sizes)
    
    def prepare_training_data(self):
        """Prepare training data from expected outputs"""
        script_dir = Path(__file__).parent
        input_dir = script_dir / INPUT_FOLDER
        expected_dir = script_dir / EXPECTED_FOLDER
        
        # Load expected outputs
        expected_data = load_expected_outputs(expected_dir)
        
        X_heading = []
        y_heading = []
        X_level = []
        y_level = []
        
        print("Preparing training data from expected outputs...")
        
        for file_num, expected_headings in expected_data.items():
            pdf_path = input_dir / f"file{file_num}.pdf"
            
            if not pdf_path.exists():
                print(f"Warning: PDF file {pdf_path} not found")
                continue
                
            print(f"Processing file{file_num}.pdf...")
            
            # Extract lines from PDF
            all_lines, avg_font_size, page_height = extract_lines_from_pdf(str(pdf_path))
            
            if not all_lines:
                continue
            
            # Create expected headings mapping
            expected_texts = {h['text']: h['level'] for h in expected_headings}
            
            # Process each line
            for idx, line in enumerate(all_lines):
                text = line.get('text', '').strip()
                if not text or len(text) < 3:
                    continue
                
                features = self.extract_features(line, all_lines, idx, avg_font_size, page_height)
                X_heading.append(features)
                
                # Find best matching expected heading
                best_level, match_score = find_best_matching_expected_heading(text, expected_headings)
                
                # Label as heading if good match found
                if match_score >= 0.7:  # Similarity threshold
                    y_heading.append(1)  # Is heading
                    X_level.append(features)
                    y_level.append(best_level)
                else:
                    y_heading.append(0)  # Not heading
        
        return np.array(X_heading), np.array(y_heading), np.array(X_level), np.array(y_level)
    
    def train(self):
        """Train the supervised model"""
        X_heading, y_heading, X_level, y_level = self.prepare_training_data()
        
        if len(X_heading) == 0:
            print("No training data available!")
            return False
        
        print(f"Training with {len(X_heading)} samples")
        print(f"Positive heading examples: {sum(y_heading)}")
        print(f"Level examples: {len(X_level)}")
        
        # Scale features
        X_heading_scaled = self.scaler.fit_transform(X_heading)
        
        # Train heading classifier
        self.heading_classifier.fit(X_heading_scaled, y_heading)
        
        # Train level classifier if we have level data
        if len(X_level) > 0:
            X_level_scaled = self.scaler.transform(X_level)
            y_level_encoded = self.level_encoder.fit_transform(y_level)
            self.level_classifier.fit(X_level_scaled, y_level_encoded)
        
        self.is_trained = True
        
        # Print feature importance
        feature_names = [
            'font_size_ratio', 'text_length', 'word_count', 'ends_with_colon', 'is_title_case',
            'is_uppercase', 'starts_with_number', 'contains_appendix', 'is_bold', 'is_italic',
            'position_y_normalized', 'next_line_size_ratio', 'prev_line_size_ratio',
            'line_spacing_above', 'line_spacing_below', 'has_punctuation', 'is_short_line',
            'font_size_percentile'
        ]
        
        importances = self.heading_classifier.feature_importances_
        for name, importance in zip(feature_names, importances):
            print(f"{name}: {importance:.4f}")
        
        return True
    
    def predict(self, all_lines: List[Dict], avg_font_size: float, page_height: float) -> List[Tuple[int, float, str]]:
        """Predict headings and their levels"""
        if not self.is_trained:
            return []
        
        predictions = []
        
        for idx, line in enumerate(all_lines):
            text = line.get("text", "").strip()
            if not text or len(text) < 3:
                predictions.append((0, 0.0, None))
                continue
            
            features = self.extract_features(line, all_lines, idx, avg_font_size, page_height)
            features_scaled = self.scaler.transform([features])
            
            # Predict if it's a heading
            is_heading = self.heading_classifier.predict(features_scaled)[0]
            heading_prob = self.heading_classifier.predict_proba(features_scaled)[0][1]
            
            level = None
            if is_heading and len(self.level_encoder.classes_) > 0:
                level_encoded = self.level_classifier.predict(features_scaled)[0]
                level = self.level_encoder.inverse_transform([level_encoded])[0]
            
            predictions.append((is_heading, heading_prob, level))
        
        return predictions
    
    def save_model(self, model_path: str):
        """Save trained model"""
        if self.is_trained:
            joblib.dump({
                'heading_classifier': self.heading_classifier,
                'level_classifier': self.level_classifier,
                'scaler': self.scaler,
                'level_encoder': self.level_encoder,
                'is_trained': True
            }, model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load trained model"""
        try:
            data = joblib.load(model_path)
            self.heading_classifier = data['heading_classifier']
            self.level_classifier = data['level_classifier']
            self.scaler = data['scaler']
            self.level_encoder = data['level_encoder']
            self.is_trained = data.get('is_trained', True)
            return True
        except:
            return False

def main():
    """Train the supervised model"""
    script_dir = Path(__file__).parent
    model_dir = script_dir / MODEL_FOLDER
    model_dir.mkdir(exist_ok=True)
    
    classifier = SupervisedHeadingClassifier()
    
    print("Training supervised heading classifier...")
    success = classifier.train()
    
    if success:
        model_path = model_dir / "supervised_heading_classifier.pkl"
        classifier.save_model(str(model_path))
        print(f"Model saved to {model_path}")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()
