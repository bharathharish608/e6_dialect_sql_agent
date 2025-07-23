# BM25-based query enhancement
from rank_bm25 import BM25Okapi
import json
import re
import logging

class BM25Enhancer:
    def __init__(self, docs_path='data/chunks.json'):
        self.docs_path = docs_path
        self.bm25 = None
        self.docs = None
        self.logger = logging.getLogger(__name__)
        self._load_bm25()
    
    def _load_bm25(self):
        """Load and initialize BM25 model"""
        self.logger.info(f"Loading BM25 model from {self.docs_path}")
        try:
            with open(self.docs_path, 'r') as f:
                self.docs = json.load(f)
            
            # Tokenize documents
            tokenized_docs = []
            for doc in self.docs:
                tokens = doc['chunk'].lower().split()
                tokenized_docs.append(tokens)
            
            self.bm25 = BM25Okapi(tokenized_docs)
            self.logger.info(f"BM25 model initialized with {len(self.docs)} documents")
        except Exception as e:
            self.logger.error(f"Error loading BM25 model: {e}")
            raise
    
    def search(self, query, top_k=5):
        """Perform BM25 search"""
        self.logger.info(f"BM25 search for query: '{query}'")
        
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top documents
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include relevant docs
                doc_with_score = self.docs[idx].copy()
                doc_with_score['bm25_score'] = scores[idx]
                results.append(doc_with_score)
        
        self.logger.info(f"BM25 found {len(results)} relevant documents")
        for i, doc in enumerate(results):
            self.logger.info(f"BM25 Result {i+1}: Score={doc['bm25_score']:.3f}, Source={doc.get('source', 'unknown')}")
        
        return results
    
    def extract_terms(self, docs, top_terms=10):
        """Extract most important terms from BM25 results"""
        self.logger.info(f"Extracting terms from {len(docs)} BM25 documents")
        
        all_text = ' '.join([doc['chunk'] for doc in docs])
        
        # Simple term frequency extraction
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = {}
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        for word in words:
            if len(word) > 2 and word not in stop_words:  # Filter short words and stop words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top terms
        top_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_terms]
        extracted_terms = [term for term, freq in top_terms]
        
        self.logger.info(f"Extracted terms: {extracted_terms}")
        return extracted_terms
    
    def get_search_stats(self):
        """Get statistics about the BM25 model"""
        return {
            'total_documents': len(self.docs),
            'model_loaded': self.bm25 is not None
        } 