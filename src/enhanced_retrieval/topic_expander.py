# Domain-specific topic expander for e6data chunks
import os
import pickle
import logging
from typing import List, Dict, Any

class DomainSpecificTopicExpander:
    def __init__(self, model_file='data/topic_model.pkl'):
        self.model_file = model_file
        self.lda_model = None
        self.vectorizer = None
        self.feature_names = None
        self.n_topics = 10
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load the domain-specific topic model
        self.load_model()
    
    def load_model(self):
        """Load the domain-specific topic model"""
        if not os.path.exists(self.model_file):
            self.logger.warning(f"Topic model file not found: {self.model_file}")
            self.logger.info("Please run the domain-specific topic model trainer first")
            return False
        
        try:
            with open(self.model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.lda_model = model_data['lda_model']
            self.vectorizer = model_data['vectorizer']
            self.feature_names = model_data['feature_names']
            self.n_topics = model_data['n_topics']
            
            self.logger.info(f"Domain-specific topic model loaded successfully")
            self.logger.info(f"Number of topics: {self.n_topics}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading topic model: {e}")
            return False
    
    def analyze_docs(self, bm25_results: List[Dict[str, Any]]) -> List[str]:
        """Analyze BM25 results and extract topic-based terms"""
        if not self.lda_model or not self.vectorizer:
            self.logger.warning("Topic model not loaded, returning empty list")
            return []
        
        self.logger.info(f"Analyzing {len(bm25_results)} BM25 results for topic expansion")
        
        # Extract text from BM25 results
        doc_texts = []
        for result in bm25_results:
            chunk_text = result.get('chunk', '')
            if chunk_text:
                # Preprocess text (same as training)
                chunk_text = chunk_text.lower()
                chunk_text = ' '.join(chunk_text.split())
                if len(chunk_text.split()) > 5:  # At least 5 words
                    doc_texts.append(chunk_text)
        
        if not doc_texts:
            self.logger.warning("No valid document texts found in BM25 results")
            return []
        
        # Transform documents using the trained vectorizer
        try:
            doc_vectors = self.vectorizer.transform(doc_texts)
            
            # Get topic distributions for the documents
            topic_distributions = self.lda_model.transform(doc_vectors)
            
            # Find the most prominent topics
            avg_topic_dist = topic_distributions.mean(axis=0)
            top_topic_indices = avg_topic_dist.argsort()[-3:][::-1]  # Top 3 topics
            
            # Extract terms from the most prominent topics
            topic_terms = []
            for topic_idx in top_topic_indices:
                topic = self.lda_model.components_[topic_idx]
                top_terms = [self.feature_names[i] for i in topic.argsort()[-8:][::-1]]
                topic_terms.extend(top_terms)
            
            # Remove duplicates and limit to reasonable number
            unique_terms = list(set(topic_terms))[:15]
            
            self.logger.info(f"Extracted {len(unique_terms)} topic-based terms: {unique_terms}")
            return unique_terms
            
        except Exception as e:
            self.logger.error(f"Error in topic analysis: {e}")
            return []
    
    def fallback_expand(self, user_input: str) -> List[str]:
        """Fallback expansion when BM25 has no results"""
        if not self.lda_model:
            self.logger.warning("Topic model not loaded, returning empty list")
            return []
        
        self.logger.info(f"Fallback topic expansion for: '{user_input}'")
        
        # Preprocess user input
        user_input = user_input.lower()
        user_input = ' '.join(user_input.split())
        
        # Transform user input
        try:
            user_vector = self.vectorizer.transform([user_input])
            
            # Get topic distribution for user input
            topic_distribution = self.lda_model.transform(user_vector)[0]
            
            # Find the most relevant topics
            top_topic_indices = topic_distribution.argsort()[-2:][::-1]  # Top 2 topics
            
            # Extract terms from relevant topics
            topic_terms = []
            for topic_idx in top_topic_indices:
                topic = self.lda_model.components_[topic_idx]
                top_terms = [self.feature_names[i] for i in topic.argsort()[-6:][::-1]]
                topic_terms.extend(top_terms)
            
            # Remove duplicates and limit
            unique_terms = list(set(topic_terms))[:10]
            
            self.logger.info(f"Fallback extracted {len(unique_terms)} terms: {unique_terms}")
            return unique_terms
            
        except Exception as e:
            self.logger.error(f"Error in fallback topic expansion: {e}")
            return []
    
    def get_topic_info(self) -> List[Dict[str, Any]]:
        """Get information about the trained topics"""
        if not self.lda_model:
            return []
        
        topics = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_terms = [self.feature_names[i] for i in topic.argsort()[-10:][::-1]]
            topics.append({
                'topic_id': topic_idx + 1,
                'top_terms': top_terms,
                'term_weights': topic.argsort()[-10:][::-1].tolist()
            })
        
        return topics
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about the topic model"""
        return {
            'model_loaded': self.lda_model is not None,
            'n_topics': self.n_topics,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'topics_info': self.get_topic_info()
        } 