# Combine all enhancement signals into final query
import logging

class QueryCombiner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def combine(self, original_query, bm25_terms, topic_terms, keybert_terms):
        """Combine all enhancement signals into final query"""
        self.logger.info("Combining query enhancement signals")
        self.logger.info(f"Original query: '{original_query}'")
        self.logger.info(f"BM25 terms: {bm25_terms}")
        self.logger.info(f"Topic terms: {topic_terms}")
        self.logger.info(f"KeyBERT terms: {keybert_terms}")
        
        # Combine all terms into enhanced query
        all_terms = []
        
        # Add original query terms (highest priority)
        original_terms = original_query.split()
        all_terms.extend(original_terms)
        self.logger.info(f"Original terms: {original_terms}")
        
        # Add BM25 terms (high weight - from relevant docs)
        all_terms.extend(bm25_terms)
        self.logger.info(f"Added BM25 terms: {bm25_terms}")
        
        # Add topic terms (medium weight - contextual)
        all_terms.extend(topic_terms)
        self.logger.info(f"Added topic terms: {topic_terms}")
        
        # Add KeyBERT terms (high weight - document-aware keywords)
        all_terms.extend(keybert_terms)
        self.logger.info(f"Added KeyBERT terms: {keybert_terms}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in all_terms:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                unique_terms.append(term)
        
        self.logger.info(f"Unique terms after deduplication: {unique_terms}")
        
        # Limit query length to avoid overwhelming vector search
        # Keep original terms + top enhancement terms
        max_terms = 20
        if len(unique_terms) > max_terms:
            # Prioritize: original terms first, then BM25, then KeyBERT, then topic
            original_count = len(original_terms)
            bm25_count = min(len(bm25_terms), 5)
            keybert_count = min(len(keybert_terms), 5)
            topic_count = min(len(topic_terms), 5)
            
            final_terms = []
            final_terms.extend(original_terms)
            final_terms.extend(bm25_terms[:bm25_count])
            final_terms.extend(keybert_terms[:keybert_count])
            final_terms.extend(topic_terms[:topic_count])
            
            unique_terms = final_terms
        
        enhanced_query = ' '.join(unique_terms)
        
        self.logger.info(f"Final enhanced query: '{enhanced_query}'")
        self.logger.info(f"Query length: {len(unique_terms)} terms")
        
        return enhanced_query
    
    def get_enhancement_stats(self, original_query, bm25_terms, topic_terms, keybert_terms):
        """Get statistics about the enhancement process"""
        original_length = len(original_query.split())
        enhanced_length = len(self.combine(original_query, bm25_terms, topic_terms, keybert_terms).split())
        
        return {
            'original_query_length': original_length,
            'enhanced_query_length': enhanced_length,
            'bm25_terms_count': len(bm25_terms),
            'topic_terms_count': len(topic_terms),
            'keybert_terms_count': len(keybert_terms),
            'total_enhancement_terms': len(bm25_terms) + len(topic_terms) + len(keybert_terms),
            'enhancement_ratio': enhanced_length / original_length if original_length > 0 else 0
        } 