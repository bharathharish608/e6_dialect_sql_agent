import os
import json
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser

DATA_DIR = 'data'
DOCS_FILE = os.path.join(DATA_DIR, 'docs.json')
CHUNKS_FILE = os.path.join(DATA_DIR, 'chunks.json')

os.makedirs(DATA_DIR, exist_ok=True)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)

def chunk_and_persist_docs():
    docs = load_json(DOCS_FILE)
    parser = SimpleNodeParser()
    all_chunks = []
    for idx, doc in enumerate(docs, 1):
        nodes = parser.get_nodes_from_documents([Document(text=doc['text'], metadata={"url": doc['url']})])
        for node in nodes:
            all_chunks.append({"url": doc['url'], "chunk": node.text})
        print(f"[{idx}/{len(docs)}] Chunked doc into {len(nodes)} chunks.")
    save_json(all_chunks, CHUNKS_FILE)
    print(f"Saved {len(all_chunks)} chunks.")
    return all_chunks

if __name__ == "__main__":
    chunk_and_persist_docs() 