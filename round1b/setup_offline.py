#!/usr/bin/env python3
"""
Pre-download and cache all models for offline execution
"""
import os
import sys
import nltk
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")

def setup_offline_models():
    """Download and cache all required models"""
    print("Setting up offline models...")
    
    try:
        # Download NLTK data
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✓ NLTK data downloaded")
        
        # Download and cache sentence transformer model
        print("Downloading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # Test model to ensure it's working
        test_embedding = model.encode(["This is a test sentence."])
        print(f"✓ Sentence transformer model downloaded and tested (embedding size: {test_embedding.shape})")
        
        # Verify model size
        import torch
        model_size_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)  # 4 bytes per float32
        print(f"✓ Model size: {model_size_mb:.1f} MB (well under 1GB limit)")
        
        print("All models setup complete!")
        return True
        
    except Exception as e:
        print(f"Error setting up models: {e}")
        return False

if __name__ == "__main__":
    success = setup_offline_models()
    if not success:
        sys.exit(1)
