#!/usr/bin/env python3
"""
Script to chunk raw web documentation from docs.json
This creates the initial web chunks that were missing from the pipeline
"""

import os
import json
from typing import List, Dict, Any

DATA_DIR = 'data'
DOCS_FILE = os.path.join(DATA_DIR, 'docs.json')
CHUNKS_FILE = os.path.join(DATA_DIR, 'chunks.json')

def load_json(path):
    """Load JSON file safely"""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

def save_json(obj, path):
    """Save JSON file safely"""
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def chunk_web_documentation():
    """
    Chunk the raw web documentation from docs.json
    """
    print("Loading raw web documentation...")
    docs = load_json(DOCS_FILE)
    
    if not docs:
        print("No web documentation found in docs.json")
        return []
    
    print(f"Found {len(docs)} web documents")
    
    web_chunks = []
    
    for doc in docs:
        url = doc.get('url', '')
        text = doc.get('text', '')
        
        if not text.strip():
            continue
            
        # Split text into chunks of ~600 characters
        chunk_size = 600
        overlap = 100
        
        # Simple chunking by character count with overlap
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        # Create chunk objects
        for i, chunk_text in enumerate(chunks):
            web_chunk = {
                'url': url,
                'chunk': chunk_text,
                'chunk_type': 'web',
                'function_name': None,
                'source': 'web',
                'content_category': 'general',
                'is_sql_function_page': 'sql-command-reference' in url,
                'is_data_type_page': 'supported-data-types' in url,
                'function_signature': None,
                'aliases': [],
                'examples': [],
                'metadata_file': 'web',
                'length': len(chunk_text)
            }
            web_chunks.append(web_chunk)
    
    print(f"Created {len(web_chunks)} web chunks")
    return web_chunks

def main():
    """
    Main function to chunk web documentation
    """
    print("Starting web documentation chunking...")
    
    # Create web chunks
    web_chunks = chunk_web_documentation()
    
    if web_chunks:
        # Save web chunks
        save_json(web_chunks, CHUNKS_FILE)
        print(f"Saved {len(web_chunks)} web chunks to {CHUNKS_FILE}")
        
        # Print some statistics
        urls = set(chunk['url'] for chunk in web_chunks)
        print(f"Covered {len(urls)} unique URLs")
        
        # Show some sample chunks
        print("\nSample chunks:")
        for i, chunk in enumerate(web_chunks[:3]):
            print(f"Chunk {i+1}: {chunk['url']}")
            print(f"Length: {chunk['length']} chars")
            print(f"Preview: {chunk['chunk'][:100]}...")
            print()
    else:
        print("No web chunks created")

if __name__ == "__main__":
    main() 