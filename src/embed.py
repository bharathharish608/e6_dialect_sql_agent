import os
import json
from sentence_transformers import SentenceTransformer
import chromadb

DATA_DIR = 'data'
CHUNKS_FILE = os.path.join(DATA_DIR, 'chunks.json')
CHROMA_PATH = os.path.join(DATA_DIR, 'chroma_db')
CALCITE_FILE = os.path.join('input', 'apache_calcite_sql_idioms.txt')

# Load chunks
with open(CHUNKS_FILE) as f:
    chunks = json.load(f)

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
                    'url': 'file://apache_calcite_sql_idioms.txt',
                    'source': 'apache_calcite',
                    'section': section or ''
                })
                bullet_lines = []
            bullet_lines = [line.strip()]
        elif bullet_lines:
            bullet_lines.append(line.strip())
    # Save last bullet
    if bullet_lines:
        chunks.append({
            'chunk': '\n'.join(bullet_lines),
            'url': 'file://apache_calcite_sql_idioms.txt',
            'source': 'apache_calcite',
            'section': section or ''
        })

# Prepare data
texts = [chunk['chunk'] for chunk in chunks]
metadatas = [{'url': chunk.get('url', ''), 'source': chunk.get('source', ''), 'section': chunk.get('section', '')} for chunk in chunks]

# Embed
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

# Store in ChromaDB
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("e6data_docs")
collection.add(
    embeddings=embeddings,
    documents=texts,
    metadatas=metadatas,
    ids=[f"chunk_{i}" for i in range(len(texts))]
)
print(f"Embedded and stored {len(texts)} chunks (including bullet-point idioms) in ChromaDB.") 