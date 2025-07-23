# Simple co-occurrence based query expansion
import pickle
import re
import logging

class SimpleWordSimilarityExpander:
    def __init__(self, model_path='data/word_similarity_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.logger = logging.getLogger(__name__)
        self._load_model()
    
    def _load_model(self):
        """Load trained similarity model"""
        self.logger.info(f"Loading similarity model from {self.model_path}")
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.logger.info(f"Similarity model loaded successfully with {len(self.model['word_counts'])} words")
        except Exception as e:
            self.logger.error(f"Error loading similarity model: {e}")
            raise
    
    def expand(self, query, topn=5):
        """Expand query using word similarities"""
        self.logger.info(f"Similarity expansion for query: '{query}'")
        
        # Extract potential function names from query
        words = re.findall(r'\b\w+\b', query.lower())
        
        expanded_terms = []
        for word in words:
            if len(word) > 2:  # Filter short words
                try:
                    similar_words = self.get_similar_words(word, topn)
                    expanded_terms.extend(similar_words)
                    self.logger.info(f"Similarity expansion for '{word}': {similar_words}")
                except Exception as e:
                    self.logger.info(f"Word '{word}' not found in vocabulary: {e}")
                    continue
        
        unique_terms = list(set(expanded_terms))  # Remove duplicates
        self.logger.info(f"Final similarity terms: {unique_terms}")
        
        return unique_terms
    
    def get_similar_words(self, word, topn=10):
        """Get similar words for a specific word"""
        self.logger.info(f"Getting similar words for '{word}'")
        
        cooccurrence = self.model['cooccurrence']
        
        if word not in cooccurrence:
            self.logger.warning(f"Word '{word}' not found in vocabulary")
            return []
        
        # Get co-occurring words and their counts
        similar_words = cooccurrence[word].most_common(topn)
        result = [word for word, count in similar_words]
        
        self.logger.info(f"Similar words to '{word}': {result}")
        return result
    
    def get_vocabulary_stats(self):
        """Get statistics about the similarity model vocabulary"""
        return {
            'vocabulary_size': len(self.model['word_counts']),
            'model_loaded': self.model is not None
        } 