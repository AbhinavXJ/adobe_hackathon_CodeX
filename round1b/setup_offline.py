import os
import time
from sentence_transformers import SentenceTransformer
import nltk

def setup_complete_offline():
    """Complete offline setup including all required data"""
    print("🚀 Adobe Hackathon - Complete Advanced Setup")
    print("="*60)
    print("📥 Downloading all models and data for complete offline execution...")
    
    start_time = time.time()
    
    try:
        # 1. Download sentence transformers model
        print("⬇️  Downloading sentence-transformers model (~90MB)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence-transformers model cached!")
        
        # 2. Download all NLTK data
        print("⬇️  Downloading NLTK data packages...")
        nltk_packages = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
        
        for package in nltk_packages:
            print(f"  📦 Downloading {package}...")
            nltk.download(package, quiet=True)
            print(f"  ✅ {package} cached!")
        
        # 3. Test complete setup
        print("🧪 Testing complete offline setup...")
        
        # Test sentence transformers
        test_embedding = model.encode(["Advanced offline test"])
        
        # Test NLTK
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk import pos_tag
        
        test_tokens = word_tokenize("This is an advanced test.")
        test_sentences = sent_tokenize("Advanced test sentence one. Advanced test sentence two.")
        test_pos = pos_tag(test_tokens)
        test_stopwords = stopwords.words('english')
        
        setup_time = time.time() - start_time
        
        print(f"✅ Complete advanced setup successful in {setup_time:.2f} seconds!")
        print("="*60)
        print("🎉 EVERYTHING CACHED! Advanced system now works 100% OFFLINE!")
        print("📊 Total cached: ~110MB | Enhanced ML capabilities ready!")
        print("🚫 No more internet downloads needed!")
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return False
    
    return True

if __name__ == "__main__":
    setup_complete_offline()
