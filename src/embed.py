import os
import json
from sentence_transformers import SentenceTransformer
import chromadb

DATA_DIR = 'data'
CHUNKS_FILE = os.path.join(DATA_DIR, 'chunks.json')
CHROMA_PATH = os.path.join(DATA_DIR, 'chroma_db')
CALCITE_FILE = os.path.join('input', 'e6data_sql_rules.txt')

# Load chunks
with open(CHUNKS_FILE) as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

# Add Apache Calcite idioms as bullet-point chunks
if os.path.exists(CALCITE_FILE):
    with open(CALCITE_FILE) as f:
        lines = f.readlines()
    section = None
    bullet = None
    bullet_lines = []
    for line in lines:
        line = line.rstrip('\n')
        if line.strip().endswith(':') and not line.strip().startswith('-'):
            section = line.strip()
        elif line.strip().startswith('- '):
            # Save previous bullet
            if bullet_lines:
                chunks.append({
                    'chunk': '\n'.join(bullet_lines),
                    'url': 'file://e6data_sql_rules.txt',
                    'source': 'e6data_sql_rules',
                    'section': section or '',
                    'chunk_type': 'idiom',
                    'function_name': None,
                    'length': len('\n'.join(bullet_lines))
                })
                bullet_lines = []
            bullet_lines = [line.strip()]
        elif bullet_lines:
            bullet_lines.append(line.strip())
    # Save last bullet
    if bullet_lines:
        chunks.append({
            'chunk': '\n'.join(bullet_lines),
            'url': 'file://e6data_sql_rules.txt',
            'source': 'e6data_sql_rules',
            'section': section or '',
            'chunk_type': 'idiom',
            'function_name': None,
            'length': len('\n'.join(bullet_lines))
        })

print(f"Added {len([c for c in chunks if c.get('chunk_type') == 'idiom'])} idiom chunks")

# Prepare data with enhanced metadata
texts = []
metadatas = []
ids = []

for i, chunk in enumerate(chunks):
    texts.append(chunk['chunk'])
    
    # Enhanced metadata for better retrieval with URL pattern awareness
    metadata = {
        'url': chunk.get('url', ''),
        'source': chunk.get('source', ''),
        'section': chunk.get('section', ''),
        'chunk_type': chunk.get('chunk_type', 'normal'),
        'function_name': chunk.get('function_name', '') or '',
        'length': chunk.get('length', len(chunk['chunk'])),
        'has_function': 'yes' if chunk.get('function_name') else 'no',
        'is_sql_function_page': str(chunk.get('is_sql_function_page', False)).lower(),
        'is_data_type_page': str(chunk.get('is_data_type_page', False)).lower(),
        'content_category': chunk.get('content_category', 'general')
    }
    
    # Add function-specific metadata for better search
    if chunk.get('function_name'):
        metadata['search_keywords'] = f"{chunk['function_name']} function syntax parameters examples"
        metadata['function_category'] = 'sql_function'
        
        # Add URL pattern-based keywords for better retrieval
        if chunk.get('is_sql_function_page'):
            metadata['search_keywords'] += f" sql command reference {chunk['function_name']} function"
        elif chunk.get('is_data_type_page'):
            metadata['search_keywords'] += f" data type {chunk['function_name']} supported types"
    else:
        metadata['search_keywords'] = chunk['chunk'][:100]  # First 100 chars as keywords
        
        # Add URL pattern-based keywords for non-function chunks
        if chunk.get('is_sql_function_page'):
            metadata['search_keywords'] += " sql command reference functions"
        elif chunk.get('is_data_type_page'):
            metadata['search_keywords'] += " supported data types"
    
    metadatas.append(metadata)
    ids.append(f"chunk_{i}")

# Print chunk statistics
function_chunks = [c for c in chunks if c.get('chunk_type') == 'function']
normal_chunks = [c for c in chunks if c.get('chunk_type') == 'normal']
idiom_chunks = [c for c in chunks if c.get('chunk_type') == 'idiom']

print(f"\nChunk Statistics:")
print(f"- Function chunks: {len(function_chunks)}")
print(f"- Normal chunks: {len(normal_chunks)}")
print(f"- Idiom chunks: {len(idiom_chunks)}")
print(f"- Total chunks: {len(chunks)}")

if function_chunks:
    function_names = list(set(c['function_name'] for c in function_chunks if c['function_name']))
    print(f"\nFunctions found: {', '.join(function_names[:10])}{'...' if len(function_names) > 10 else ''}")

# Embed
print(f"\nCreating embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

# Store in ChromaDB
print(f"Storing in ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Delete existing collection if it exists
try:
    client.delete_collection("e6data_docs")
    print("Deleted existing collection")
except:
    pass

collection = client.create_collection("e6data_docs")

# Add in batches to avoid batch size limit
batch_size = 5000
for i in range(0, len(texts), batch_size):
    end_idx = min(i + batch_size, len(texts))
    print(f"Adding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} (chunks {i+1}-{end_idx})")
    
    collection.add(
        embeddings=embeddings[i:end_idx],
        documents=texts[i:end_idx],
        metadatas=metadatas[i:end_idx],
        ids=ids[i:end_idx]
    )

print(f"\n✅ Successfully embedded and stored {len(texts)} chunks in ChromaDB")
print(f"Collection: e6data_docs")
print(f"Path: {CHROMA_PATH}")

# Verify the collection
print(f"\nVerification:")
print(f"- Collection count: {collection.count()}")
print(f"- Sample metadata keys: {list(metadatas[0].keys()) if metadatas else 'None'}") 