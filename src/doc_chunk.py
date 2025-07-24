import os
import json
import re
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path

DATA_DIR = 'data'
DOCS_FILE = os.path.join(DATA_DIR, 'docs.json')
CHUNKS_FILE = os.path.join(DATA_DIR, 'chunks.json')
IDIOMS_FILE = 'input/e6data_sql_rules.txt'
FUNCTION_METADATA_DIR = os.path.join(DATA_DIR, 'sql-functions-metadata')

os.makedirs(DATA_DIR, exist_ok=True)

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

def load_idioms_file():
    """Load the e6data_sql_rules.txt file"""
    if os.path.exists(IDIOMS_FILE):
        with open(IDIOMS_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        return {
            'url': 'e6data_sql_rules',
            'text': content,
            'source': 'idiom'
        }
    return None

def load_function_metadata():
    """
    Load all JSON files from data/sql-functions-metadata/
    Returns: List of function objects with standardized structure
    """
    functions = []
    
    # Load master index if it exists
    master_file = os.path.join(FUNCTION_METADATA_DIR, 'all_functions.json')
    if os.path.exists(master_file):
        with open(master_file, 'r') as f:
            master_data = json.load(f)
            for func in master_data:
                if 'metadata' in func:
                    functions.append(func['metadata'])
    else:
        # Load individual category files
        category_files = [
            'string-functions-1.json', 'string-functions-2.json',
            'datetime-functions-1.json', 'datetime-functions-2.json',
            'numeric-functions-1.json', 'numeric-functions-2.json',
            'array-functions.json', 'aggregate-functions.json',
            'conversion-system-functions.json', 'encoding-hash-functions.json',
            'regex-pattern-functions.json', 'geospatial-functions.json',
            'bitwise-url-functions.json', 'miscellaneous-functions.json'
        ]
        
        for filename in category_files:
            filepath = os.path.join(FUNCTION_METADATA_DIR, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if 'functions' in data:
                        functions.extend(data['functions'])
    
    print(f"Loaded {len(functions)} functions from metadata")
    return functions

def create_function_chunks(metadata_functions):
    """
    Create detailed chunks for each function
    Target: 1200 characters with complete information
    """
    function_chunks = []
    
    for func in metadata_functions:
        # Build comprehensive function chunk
        chunk_parts = []
        
        # Function header
        chunk_parts.append(f"Function: {func['name']}")
        
        # Signature (if we can construct it)
        if 'parameters' in func:
            param_desc = func['parameters'].get('description', '')
            param_count = func['parameters'].get('count', '')
            signature = f"{func['name']}({param_desc})"
            chunk_parts.append(f"Signature: {signature}")
        
        # Description
        if 'description' in func:
            chunk_parts.append(f"Description: {func['description']}")
        
        # Parameters
        if 'parameters' in func:
            params = func['parameters']
            param_info = []
            if 'count' in params:
                param_info.append(f"Count: {params['count']}")
            if 'types' in params:
                param_info.append(f"Types: {params['types']}")
            if 'description' in params:
                param_info.append(f"Description: {params['description']}")
            if param_info:
                chunk_parts.append(f"Parameters: {'; '.join(param_info)}")
        
        # Return type
        if 'returnType' in func:
            chunk_parts.append(f"Return Type: {func['returnType']}")
        
        # Examples
        if 'examples' in func and func['examples']:
            chunk_parts.append("Examples:")
            for example in func['examples']:
                chunk_parts.append(f"- {example}")
        
        # Aliases
        if 'aliases' in func and func['aliases']:
            chunk_parts.append(f"Aliases: {', '.join(func['aliases'])}")
        
        # Category
        if 'category' in func:
            chunk_parts.append(f"Category: {func['category']}")
        
        # SQL Kind
        if 'sqlKind' in func:
            chunk_parts.append(f"SQL Kind: {func['sqlKind']}")
        
        # Create the chunk text
        chunk_text = '\n'.join(chunk_parts)
        
        # Ensure chunk is around 1200 characters
        if len(chunk_text) > 1200:
            # Truncate examples if too long
            lines = chunk_text.split('\n')
            truncated_lines = []
            current_length = 0
            
            for line in lines:
                if current_length + len(line) + 1 > 1200:
                    break
                truncated_lines.append(line)
                current_length += len(line) + 1
            
            chunk_text = '\n'.join(truncated_lines)
        
        # Create chunk object
        chunk = {
            'url': f"function://{func['name']}",
            'chunk': chunk_text,
            'chunk_type': 'function',
            'function_name': func['name'],
            'source': 'metadata',
            'content_category': 'sql_function',
            'is_sql_function_page': True,
            'is_data_type_page': False,
            'function_signature': f"{func['name']}({func['parameters'].get('description', '')})" if 'parameters' in func else func['name'],
            'aliases': func.get('aliases', []),
            'examples': func.get('examples', []),
            'metadata_file': 'function_metadata',
            'length': len(chunk_text)
        }
        
        function_chunks.append(chunk)
    
    print(f"Created {len(function_chunks)} function chunks")
    return function_chunks

def detect_functions_in_web_chunks(web_chunks, function_names):
    """
    Scan existing web chunks for function mentions
    Returns: Dict mapping function_name -> list of web chunks
    """
    function_matches = {}
    
    # Create set of function names for faster lookup
    function_name_set = {name.lower() for name in function_names}
    
    for chunk in web_chunks:
        chunk_text = chunk.get('chunk', '').lower()
        
        for func_name in function_names:
            # Check for function name in chunk
            if func_name.lower() in chunk_text:
                if func_name not in function_matches:
                    function_matches[func_name] = []
                function_matches[func_name].append(chunk)
    
    print(f"Found function mentions in {len(function_matches)} functions across web chunks")
    return function_matches

def calculate_chunk_similarity(chunk1, chunk2):
    """
    Determine if chunks contain similar function information
    Returns: Similarity score (0-1)
    """
    text1 = chunk1.get('chunk', '').lower()
    text2 = chunk2.get('chunk', '').lower()
    
    # Extract function names
    func1 = chunk1.get('function_name', '').lower()
    func2 = chunk2.get('function_name', '').lower()
    
    # If same function name, high similarity
    if func1 and func2 and func1 == func2:
        return 0.9
    
    # Keyword overlap
    words1 = set(re.findall(r'\w+', text1))
    words2 = set(re.findall(r'\w+', text2))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    jaccard_similarity = len(intersection) / len(union)
    
    return jaccard_similarity

def enhance_web_chunks(web_chunks, function_metadata):
    """
    Merge web content with function metadata where applicable
    """
    enhanced_chunks = []
    function_names = [func['name'] for func in function_metadata]
    function_matches = detect_functions_in_web_chunks(web_chunks, function_names)
    
    # Create lookup for function metadata
    func_metadata_lookup = {func['name']: func for func in function_metadata}
    
    for chunk in web_chunks:
        chunk_text = chunk.get('chunk', '')
        chunk_lower = chunk_text.lower()
        
        # Check if this chunk mentions any functions
        mentioned_functions = []
        for func_name in function_names:
            if func_name.lower() in chunk_lower:
                mentioned_functions.append(func_name)
        
        if mentioned_functions:
            # This chunk mentions functions - enhance it
            enhanced_text = chunk_text + "\n\n"
            enhanced_text += "Related Functions:\n"
            
            for func_name in mentioned_functions:
                if func_name in func_metadata_lookup:
                    func = func_metadata_lookup[func_name]
                    enhanced_text += f"- {func_name}: {func.get('description', '')}\n"
                    if func.get('examples'):
                        enhanced_text += f"  Example: {func['examples'][0]}\n"
            
            # Truncate if too long
            if len(enhanced_text) > 1200:
                enhanced_text = enhanced_text[:1197] + "..."
            
            enhanced_chunk = {
                'url': chunk.get('url', ''),
                'chunk': enhanced_text,
                'chunk_type': 'enhanced_web',
                'function_name': mentioned_functions[0] if mentioned_functions else None,
                'source': 'merged',
                'content_category': 'sql_function',
                'is_sql_function_page': 'sql-command-reference' in chunk.get('url', ''),
                'is_data_type_page': 'supported-data-types' in chunk.get('url', ''),
                'function_signature': None,
                'aliases': [],
                'examples': [],
                'metadata_file': 'enhanced_web',
                'length': len(enhanced_text)
            }
            
            enhanced_chunks.append(enhanced_chunk)
        else:
            # No function mentions - keep as is
            enhanced_chunk = {
                'url': chunk.get('url', ''),
                'chunk': chunk_text,
                'chunk_type': 'web',
                'function_name': None,
                'source': 'web',
                'content_category': 'general',
                'is_sql_function_page': 'sql-command-reference' in chunk.get('url', ''),
                'is_data_type_page': 'supported-data-types' in chunk.get('url', ''),
                'function_signature': None,
                'aliases': [],
                'examples': [],
                'metadata_file': 'web',
                'length': len(chunk_text)
            }
            
            enhanced_chunks.append(enhanced_chunk)
    
    print(f"Created {len(enhanced_chunks)} enhanced web chunks")
    return enhanced_chunks

def deduplicate_function_chunks(function_chunks, enhanced_web_chunks, original_web_chunks=None):
    """
    Remove duplicates and merge overlapping content
    Priority: Function metadata > Enhanced web > Pure web
    """
    # Group chunks by function name
    function_groups = {}
    
    # Add function chunks
    for chunk in function_chunks:
        func_name = chunk.get('function_name')
        if func_name:
            if func_name not in function_groups:
                function_groups[func_name] = []
            function_groups[func_name].append(chunk)
    
    # Add enhanced web chunks
    for chunk in enhanced_web_chunks:
        func_name = chunk.get('function_name')
        if func_name:
            if func_name not in function_groups:
                function_groups[func_name] = []
            function_groups[func_name].append(chunk)
    
    # Deduplicate each group
    deduplicated_chunks = []
    
    for func_name, chunks in function_groups.items():
        if len(chunks) == 1:
            # Only one chunk for this function
            deduplicated_chunks.append(chunks[0])
        else:
            # Multiple chunks - keep the best one
            # Priority: function > enhanced_web > web
            best_chunk = None
            best_priority = -1
            
            for chunk in chunks:
                chunk_type = chunk.get('chunk_type', '')
                if chunk_type == 'function':
                    priority = 3
                elif chunk_type == 'enhanced_web':
                    priority = 2
                else:
                    priority = 1
                
                if priority > best_priority:
                    best_priority = priority
                    best_chunk = chunk
            
            if best_chunk:
                deduplicated_chunks.append(best_chunk)
    
    # Add enhanced web chunks without function names
    for chunk in enhanced_web_chunks:
        if not chunk.get('function_name'):
            deduplicated_chunks.append(chunk)
    
    # Add original web chunks (deployment docs, etc.) - these are crucial!
    if original_web_chunks:
        for chunk in original_web_chunks:
            # Only add if it doesn't have a function name (general docs)
            if not chunk.get('function_name'):
                # Convert to standard format
                web_chunk = {
                    'url': chunk.get('url', ''),
                    'chunk': chunk.get('text', chunk.get('chunk', '')),
                    'chunk_type': 'web',
                    'function_name': None,
                    'source': 'web',
                    'content_category': 'general',
                    'is_sql_function_page': 'sql-command-reference' in chunk.get('url', ''),
                    'is_data_type_page': 'supported-data-types' in chunk.get('url', ''),
                    'function_signature': None,
                    'aliases': [],
                    'examples': [],
                    'metadata_file': 'web',
                    'length': len(chunk.get('text', chunk.get('chunk', '')))
                }
                deduplicated_chunks.append(web_chunk)
    
    print(f"Deduplicated {len(function_chunks) + len(enhanced_web_chunks)} chunks to {len(deduplicated_chunks)} chunks")
    return deduplicated_chunks

def create_unified_chunks():
    """
    Main function to create unified chunk collection
    """
    print("Starting unified chunk creation...")
    
    # Step 1: Load all data sources
    print("Loading data sources...")
    
    # Load existing web chunks
    existing_chunks = load_json(CHUNKS_FILE)
    print(f"Loaded {len(existing_chunks)} existing web chunks")
    
    # Load function metadata
    function_metadata = load_function_metadata()
    
    # Load SQL idioms
    idioms_doc = load_idioms_file()
    
    # Step 2: Process function metadata
    print("Processing function metadata...")
    function_chunks = create_function_chunks(function_metadata)
    
    # Step 3: Enhance web chunks
    print("Enhancing web chunks...")
    enhanced_web_chunks = enhance_web_chunks(existing_chunks, function_metadata)
    
    # Step 4: Create idiom chunks
    print("Processing SQL idioms...")
    idiom_chunks = []
    if idioms_doc:
        # Split idioms into chunks
        lines = idioms_doc['text'].split('\n')
        current_chunk = []
        current_length = 0
        
        for line in lines:
            if current_length + len(line) > 400:  # Smaller chunks for idioms
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    idiom_chunk = {
                        'url': 'e6data_sql_rules',
                        'chunk': chunk_text,
                        'chunk_type': 'idiom',
                        'function_name': None,
                        'source': 'idiom',
                        'content_category': 'general',
                        'is_sql_function_page': False,
                        'is_data_type_page': False,
                        'function_signature': None,
                        'aliases': [],
                        'examples': [],
                        'metadata_file': 'e6data_sql_rules.txt',
                        'length': len(chunk_text)
                    }
                    idiom_chunks.append(idiom_chunk)
                current_chunk = [line]
                current_length = len(line)
            else:
                current_chunk.append(line)
                current_length += len(line) + 1
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            idiom_chunk = {
                'url': 'e6data_sql_rules',
                'chunk': chunk_text,
                'chunk_type': 'idiom',
                'function_name': None,
                'source': 'idiom',
                'content_category': 'general',
                'is_sql_function_page': False,
                'is_data_type_page': False,
                'function_signature': None,
                'aliases': [],
                'examples': [],
                'metadata_file': 'e6data_sql_rules.txt',
                'length': len(chunk_text)
            }
            idiom_chunks.append(idiom_chunk)
    
    print(f"Created {len(idiom_chunks)} idiom chunks")
    
    # Step 5: Deduplicate and merge
    print("Deduplicating chunks...")
    all_chunks = function_chunks + enhanced_web_chunks + idiom_chunks
    deduplicated_chunks = deduplicate_function_chunks(function_chunks, enhanced_web_chunks, existing_chunks)
    
    # Add idiom chunks (no deduplication needed)
    final_chunks = deduplicated_chunks + idiom_chunks
    
    # Step 6: Save unified chunks
    print("Saving unified chunks...")
    save_json(final_chunks, CHUNKS_FILE)
    
    # Step 7: Print statistics
    print("\n=== Unified Chunk Statistics ===")
    function_chunks_count = len([c for c in final_chunks if c.get('chunk_type') == 'function'])
    enhanced_web_count = len([c for c in final_chunks if c.get('chunk_type') == 'enhanced_web'])
    web_count = len([c for c in final_chunks if c.get('chunk_type') == 'web'])
    idiom_count = len([c for c in final_chunks if c.get('chunk_type') == 'idiom'])
    
    print(f"- Function chunks: {function_chunks_count}")
    print(f"- Enhanced web chunks: {enhanced_web_count}")
    print(f"- Web chunks: {web_count}")
    print(f"- Idiom chunks: {idiom_count}")
    print(f"- Total chunks: {len(final_chunks)}")
    
    # Average chunk sizes
    function_avg = sum(c.get('length', 0) for c in final_chunks if c.get('chunk_type') == 'function') / max(function_chunks_count, 1)
    enhanced_avg = sum(c.get('length', 0) for c in final_chunks if c.get('chunk_type') == 'enhanced_web') / max(enhanced_web_count, 1)
    web_avg = sum(c.get('length', 0) for c in final_chunks if c.get('chunk_type') == 'web') / max(web_count, 1)
    idiom_avg = sum(c.get('length', 0) for c in final_chunks if c.get('chunk_type') == 'idiom') / max(idiom_count, 1)
    
    print(f"\nAverage chunk sizes:")
    print(f"- Function chunks: {function_avg:.0f} chars")
    print(f"- Enhanced web chunks: {enhanced_avg:.0f} chars")
    print(f"- Web chunks: {web_avg:.0f} chars")
    print(f"- Idiom chunks: {idiom_avg:.0f} chars")
    
    return final_chunks

def chunk_and_persist_docs():
    """
    Main entry point - creates unified chunks from all sources
    """
    return create_unified_chunks()

if __name__ == "__main__":
    chunk_and_persist_docs() 