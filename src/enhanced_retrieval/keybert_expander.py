# KeyBERT-based query expansion for e6data documentation
import os
import pickle
import logging
from typing import List, Tuple, Dict, Any
from training.keybert_trainer import KeyBERTTrainer

class KeyBERTExpander:
    def __init__(self, model_file='data/keybert_model.pkl'):
        self.model_file = model_file
        self.keybert_trainer = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize KeyBERT trainer
        self.initialize_keybert()
    
    def initialize_keybert(self):
        """Initialize KeyBERT trainer"""
        try:
            self.keybert_trainer = KeyBERTTrainer(model_file=self.model_file)
            
            # Try to load existing model
            if not self.keybert_trainer.load_model():
                self.logger.warning("KeyBERT model not found, will need to be trained")
                self.logger.info("Please run the KeyBERT trainer first")
            
        except Exception as e:
            self.logger.error(f"Error initializing KeyBERT: {e}")
    
    def expand(self, user_input: str, top_k: int = 8, similarity_threshold: float = 0.7) -> List[str]:
        """Extract keywords from e6data documentation similar to the query"""
        if not self.keybert_trainer:
            self.logger.warning("KeyBERT trainer not initialized, returning empty list")
            return []
        
        try:
            self.logger.info(f"KeyBERT expansion for query: '{user_input}'")
            
            # Extract keywords using KeyBERT
            keywords_with_scores = self.keybert_trainer.extract_keywords_from_query(
                user_input, 
                top_k=top_k, 
                similarity_threshold=similarity_threshold
            )
            
            # Extract just the keywords (without scores)
            keywords = [keyword for keyword, score in keywords_with_scores]
            
            # Filter out generic terms that don't add value
            filtered_keywords = self.filter_generic_terms(keywords)
            
            self.logger.info(f"KeyBERT extracted {len(filtered_keywords)} keywords: {filtered_keywords}")
            return filtered_keywords
            
        except Exception as e:
            self.logger.error(f"Error in KeyBERT expansion: {e}")
            return []
    
    def filter_generic_terms(self, keywords: List[str]) -> List[str]:
        """Filter out generic terms that don't add search value"""
        # Terms that are too generic for search
        generic_terms = {
            'documentation', 'examples', 'usage', 'information', 'details',
            'description', 'overview', 'introduction', 'summary', 'notes',
            'reference', 'guide', 'manual', 'help', 'support', 'tutorial',
            'the', 'and', 'or', 'with', 'for', 'from', 'that', 'this', 'these',
            'function', 'functions', 'syntax', 'parameter', 'parameters',
            'return', 'returns', 'value', 'values', 'result', 'results'
        }
        
        # Filter out generic terms
        filtered = []
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if (keyword_lower not in generic_terms and 
                len(keyword_lower) > 2 and  # Avoid very short terms
                not keyword_lower.isdigit()):  # Avoid pure numbers
                filtered.append(keyword)
        
        return filtered
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Get statistics about the KeyBERT model"""
        if not self.keybert_trainer:
            return {
                'model_loaded': False,
                'chunks_count': 0,
                'embeddings_computed': False
            }
        
        return self.keybert_trainer.get_model_stats()
    
    def test_keyword_extraction(self, test_queries: List[str] = None):
        """Test keyword extraction with sample queries"""
        if not test_queries:
            test_queries = [
                "how does e6data datediff work",
                "SQL syntax for window functions", 
                "aggregate functions in e6data",
                "date time functions examples"
            ]
        
        print("Testing KeyBERT keyword extraction:")
        print("=" * 60)
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            keywords = self.expand(query, top_k=8)
            print(f"Keywords: {keywords}")
        
        print("\nKeyBERT testing completed!") 