import json
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import load_expected_outputs, text_similarity

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output" 
EXPECTED_FOLDER = "expected"

def evaluate_model_performance():
    """Evaluate model performance against expected outputs"""
    script_dir = Path(__file__).parent
    output_dir = script_dir / OUTPUT_FOLDER
    expected_dir = script_dir / EXPECTED_FOLDER
    
    expected_data = load_expected_outputs(expected_dir)
    
    overall_metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'hierarchy_accuracy': []
    }
    
    print("Evaluating model performance...")
    print("-" * 60)
    
    for file_num, expected_headings in expected_data.items():
        output_file = output_dir / f"file{file_num}_structure.json"
        
        if not output_file.exists():
            print(f"Output file {output_file} not found")
            continue
        
        # Load predicted output
        with open(output_file, 'r', encoding='utf-8') as f:
            predicted_data = json.load(f)
        
        predicted_headings = predicted_data.get('outline', [])
        
        # Calculate metrics
        precision, recall, f1, hierarchy_acc = calculate_metrics(expected_headings, predicted_headings)
        
        overall_metrics['precision'].append(precision)
        overall_metrics['recall'].append(recall)
        overall_metrics['f1'].append(f1)
        overall_metrics['hierarchy_accuracy'].append(hierarchy_acc)
        
        print(f"File {file_num}:")
        print(f"  Expected headings: {len(expected_headings)}")
        print(f"  Predicted headings: {len(predicted_headings)}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  Hierarchy Accuracy: {hierarchy_acc:.3f}")
        print("-" * 40)
    
    # Overall performance
    if overall_metrics['precision']:
        avg_precision = sum(overall_metrics['precision']) / len(overall_metrics['precision'])
        avg_recall = sum(overall_metrics['recall']) / len(overall_metrics['recall'])
        avg_f1 = sum(overall_metrics['f1']) / len(overall_metrics['f1'])
        avg_hierarchy = sum(overall_metrics['hierarchy_accuracy']) / len(overall_metrics['hierarchy_accuracy'])
        
        print("OVERALL PERFORMANCE:")
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"Average F1-Score: {avg_f1:.3f}")
        print(f"Average Hierarchy Accuracy: {avg_hierarchy:.3f}")

def calculate_metrics(expected: List[Dict], predicted: List[Dict]) -> Tuple[float, float, float, float]:
    """Calculate precision, recall, F1, and hierarchy accuracy"""
    if not expected:
        return 0.0, 0.0, 0.0, 0.0
    
    # Create text mappings
    expected_texts = [h['text'] for h in expected]
    predicted_texts = [h['text'] for h in predicted]
    
    # Find matches based on text similarity
    matched_expected = set()
    matched_predicted = set()
    correct_hierarchy = 0
    
    for i, pred_heading in enumerate(predicted):
        pred_text = pred_heading['text']
        pred_level = pred_heading.get('level', 'H3')
        
        best_match_idx = -1
        best_similarity = 0.0
        
        for j, exp_heading in enumerate(expected):
            if j in matched_expected:
                continue
                
            exp_text = exp_heading['text']
            similarity = text_similarity(pred_text, exp_text)
            
            if similarity > best_similarity and similarity >= 0.7:
                best_similarity = similarity
                best_match_idx = j
        
        if best_match_idx >= 0:
            matched_expected.add(best_match_idx)
            matched_predicted.add(i)
            
            # Check hierarchy accuracy
            exp_level = expected[best_match_idx]['level']
            if pred_level == exp_level:
                correct_hierarchy += 1
    
    # Calculate metrics
    true_positives = len(matched_predicted)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    hierarchy_accuracy = correct_hierarchy / true_positives if true_positives > 0 else 0.0
    
    return precision, recall, f1, hierarchy_accuracy

if __name__ == "__main__":
    evaluate_model_performance()
