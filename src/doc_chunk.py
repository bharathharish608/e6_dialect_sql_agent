import os
import json
import re
from typing import List, Dict, Any

DATA_DIR = 'data'
DOCS_FILE = os.path.join(DATA_DIR, 'docs.json')
CHUNKS_FILE = os.path.join(DATA_DIR, 'chunks.json')
IDIOMS_FILE = 'input/apache_calcite_sql_idioms.txt'

os.makedirs(DATA_DIR, exist_ok=True)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)

def load_idioms_file():
    """Load the apache_calcite_sql_idioms.txt file"""
    if os.path.exists(IDIOMS_FILE):
        with open(IDIOMS_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        return {
            'url': 'apache_calcite_sql_idioms',
            'text': content
        }
    return None

def detect_function_sections(text: str) -> List[Dict[str, Any]]:
    """Detect function sections in documentation text"""
    function_sections = []
    
    # Function detection patterns
    patterns = [
        # SQL function patterns
        r'(\w+)\s*\(\s*[^)]*\s*\)\s*[-–—]\s*(.+)',  # FUNCTION_NAME(params) - description
        r'(\w+)\s*\(\s*[^)]*\s*\)\s*\n\s*(.+)',     # FUNCTION_NAME(params)\ndescription
        r'###\s*(\w+)\s*\([^)]*\)\s*Function:',     # ### FUNCTION_NAME(params) Function:
        r'\*\*(\w+)\s*Function:\*\*',                # **FUNCTION_NAME Function:**
        r'####\s*(\w+)\s*FUNCTION',                 # #### FUNCTION_NAME FUNCTION
        r'(\w+)\s*\(\s*[^)]*\s*\)\s*\n\s*Converts', # FUNCTION_NAME(params)\nConverts
        r'(\w+)\s*\(\s*[^)]*\s*\)\s*\n\s*Returns',  # FUNCTION_NAME(params)\nReturns
        r'(\w+)\s*\(\s*[^)]*\s*\)\s*\n\s*This',     # FUNCTION_NAME(params)\nThis
    ]
    
    lines = text.split('\n')
    current_section = None
    
    for i, line in enumerate(lines):
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                function_name = match.group(1).upper()
                # Skip common non-functions
                if function_name in ['COPY', 'SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'JOIN', 'UNION']:
                    continue
                
                # Find the end of this function section
                section_start = i
                section_end = i
                
                # Look ahead for next function or section
                for j in range(i + 1, len(lines)):
                    # Check if this line starts a new function
                    if any(re.search(p, lines[j], re.IGNORECASE) for p in patterns):
                        break
                    # Check if this line starts a new major section
                    if re.match(r'^#{1,3}\s+[A-Z]', lines[j]):
                        break
                    section_end = j
                
                # Extract the function section
                section_text = '\n'.join(lines[section_start:section_end + 1])
                if len(section_text.strip()) > 50:  # Minimum meaningful section
                    function_sections.append({
                        'function_name': function_name,
                        'start_line': section_start,
                        'end_line': section_end,
                        'text': section_text,
                        'length': len(section_text)
                    })
                break
    
    return function_sections

def create_semantic_chunks(text: str, url: str) -> List[Dict[str, Any]]:
    """Create semantic chunks with function-first partitioning and URL-based categorization"""
    chunks = []
    
    # Determine content type based on URL pattern
    is_sql_function_page = 'sql-command-reference' in url
    is_data_type_page = 'supported-data-types' in url
    is_function_doc = is_sql_function_page or is_data_type_page
    
    # Detect function sections first
    function_sections = detect_function_sections(text)
    
    # Create function chunks (800-1000 characters) with enhanced metadata
    for section in function_sections:
        section_text = section['text']
        
        # If section is too long, split it intelligently
        if len(section_text) > 1000:
            # Try to split on examples or subsections
            parts = re.split(r'\n\s*Copy\s*\n', section_text)
            if len(parts) > 1:
                for i, part in enumerate(parts):
                    if len(part.strip()) > 100:
                        chunks.append({
                            'url': url,
                            'chunk': part.strip(),
                            'chunk_type': 'function',
                            'function_name': section['function_name'],
                            'length': len(part.strip()),
                            'is_sql_function_page': is_sql_function_page,
                            'is_data_type_page': is_data_type_page,
                            'content_category': 'sql_function' if is_sql_function_page else 'data_type' if is_data_type_page else 'general'
                        })
            else:
                # Split by lines to fit within 1000 chars
                lines = section_text.split('\n')
                current_chunk = []
                current_length = 0
                
                for line in lines:
                    if current_length + len(line) > 1000 and current_chunk:
                        chunks.append({
                            'url': url,
                            'chunk': '\n'.join(current_chunk),
                            'chunk_type': 'function',
                            'function_name': section['function_name'],
                            'length': current_length,
                            'is_sql_function_page': is_sql_function_page,
                            'is_data_type_page': is_data_type_page,
                            'content_category': 'sql_function' if is_sql_function_page else 'data_type' if is_data_type_page else 'general'
                        })
                        current_chunk = [line]
                        current_length = len(line)
                    else:
                        current_chunk.append(line)
                        current_length += len(line) + 1
                
                if current_chunk:
                    chunks.append({
                        'url': url,
                        'chunk': '\n'.join(current_chunk),
                        'chunk_type': 'function',
                        'function_name': section['function_name'],
                        'length': current_length,
                        'is_sql_function_page': is_sql_function_page,
                        'is_data_type_page': is_data_type_page,
                        'content_category': 'sql_function' if is_sql_function_page else 'data_type' if is_data_type_page else 'general'
                    })
        else:
            # Section fits in one chunk
            chunks.append({
                'url': url,
                'chunk': section_text,
                'chunk_type': 'function',
                'function_name': section['function_name'],
                'length': len(section_text),
                'is_sql_function_page': is_sql_function_page,
                'is_data_type_page': is_data_type_page,
                'content_category': 'sql_function' if is_sql_function_page else 'data_type' if is_data_type_page else 'general'
            })
    
    # Create normal chunks for remaining text (400-500 characters)
    # Find text that's not covered by function sections
    covered_lines = set()
    for section in function_sections:
        for i in range(section['start_line'], section['end_line'] + 1):
            covered_lines.add(i)
    
    lines = text.split('\n')
    normal_text_lines = []
    
    for i, line in enumerate(lines):
        if i not in covered_lines:
            normal_text_lines.append(line)
    
    normal_text = '\n'.join(normal_text_lines)
    
    # Split normal text into 400-500 character chunks
    if normal_text.strip():
        words = normal_text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > 500 and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.strip()) > 100:  # Minimum meaningful chunk
                    chunks.append({
                        'url': url,
                        'chunk': chunk_text,
                        'chunk_type': 'normal',
                        'function_name': None,
                        'length': len(chunk_text),
                        'is_sql_function_page': is_sql_function_page,
                        'is_data_type_page': is_data_type_page,
                        'content_category': 'sql_function' if is_sql_function_page else 'data_type' if is_data_type_page else 'general'
                    })
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) > 100:
                chunks.append({
                    'url': url,
                    'chunk': chunk_text,
                    'chunk_type': 'normal',
                    'function_name': None,
                    'length': len(chunk_text),
                    'is_sql_function_page': is_sql_function_page,
                    'is_data_type_page': is_data_type_page,
                    'content_category': 'sql_function' if is_sql_function_page else 'data_type' if is_data_type_page else 'general'
                })
    
    return chunks

def chunk_and_persist_docs():
    # Load main docs
    docs = load_json(DOCS_FILE)
    
    # Load idioms file
    idioms_doc = load_idioms_file()
    if idioms_doc:
        docs.append(idioms_doc)
        print(f"Added apache_calcite_sql_idioms.txt to processing")
    
    all_chunks = []
    
    for idx, doc in enumerate(docs, 1):
        print(f"[{idx}/{len(docs)}] Processing: {doc['url']}")
        
        # Create semantic chunks
        chunks = create_semantic_chunks(doc['text'], doc['url'])
        
        # Add metadata
        for chunk in chunks:
            chunk['source'] = doc.get('source', '')
            chunk['section'] = doc.get('section', '')
        
        all_chunks.extend(chunks)
        print(f"  Created {len(chunks)} chunks ({sum(c['length'] for c in chunks)} total chars)")
        
        # Show chunk types
        function_chunks = [c for c in chunks if c['chunk_type'] == 'function']
        normal_chunks = [c for c in chunks if c['chunk_type'] == 'normal']
        print(f"  - Function chunks: {len(function_chunks)}")
        print(f"  - Normal chunks: {len(normal_chunks)}")
        
        if function_chunks:
            function_names = list(set(c['function_name'] for c in function_chunks))
            print(f"  - Functions found: {', '.join(function_names[:5])}{'...' if len(function_names) > 5 else ''}")
    
    save_json(all_chunks, CHUNKS_FILE)
    print(f"\nSaved {len(all_chunks)} total chunks.")
    
    # Summary statistics
    function_chunks = [c for c in all_chunks if c['chunk_type'] == 'function']
    normal_chunks = [c for c in all_chunks if c['chunk_type'] == 'normal']
    
    print(f"\nChunk Statistics:")
    print(f"- Function chunks: {len(function_chunks)} (avg length: {sum(c['length'] for c in function_chunks) // len(function_chunks) if function_chunks else 0})")
    print(f"- Normal chunks: {len(normal_chunks)} (avg length: {sum(c['length'] for c in normal_chunks) // len(normal_chunks) if normal_chunks else 0})")
    print(f"- Total chunks: {len(all_chunks)}")
    
    return all_chunks

if __name__ == "__main__":
    chunk_and_persist_docs() 