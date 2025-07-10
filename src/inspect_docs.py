import json

with open('data/docs.json') as f:
    docs = json.load(f)

for i, doc in enumerate(docs[:5], 1):
    url = doc.get('url', '')
    text = doc.get('text', '')
    print(f"Doc {i} URL: {url}")
    print(f"Text (first 200 chars): {text[:200]!r}\n") 