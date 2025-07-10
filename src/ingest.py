import os
import json
from datetime import datetime
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

load_dotenv("config.env")

DATA_DIR = 'data'
DOCS_FILE = os.path.join(DATA_DIR, 'docs.json')
META_FILE = os.path.join(DATA_DIR, 'metadata.json')

os.makedirs(DATA_DIR, exist_ok=True)

def get_all_internal_links(start_url, domain):
    visited = set()
    to_visit = [start_url]
    while to_visit:
        url = to_visit.pop()
        if url in visited:
            continue
        try:
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, 'html.parser')
            visited.add(url)
            print(f"Discovered: {url} (total: {len(visited)})")
            for a in soup.find_all('a', href=True):
                link = a['href']
                if link.startswith('/'):
                    link = f"https://{domain}{link}"
                if link.startswith(f"https://{domain}") and link not in visited:
                    to_visit.append(link)
        except Exception as e:
            print(f"Error crawling {url}: {e}")
    return list(visited)

def extract_main_content(soup):
    # Try <main>, <article>, or the largest <div> as main content
    main = soup.find('main')
    if main:
        return main.get_text(separator=' ', strip=True)
    article = soup.find('article')
    if article:
        return article.get_text(separator=' ', strip=True)
    # Find the largest <div> by text length
    divs = soup.find_all('div')
    if divs:
        largest_div = max(divs, key=lambda d: len(d.get_text(separator=' ', strip=True)))
        text = largest_div.get_text(separator=' ', strip=True)
        if len(text) > 200:  # Arbitrary threshold to avoid navbars
            return text
    # Fallback to body text
    body = soup.find('body')
    if body:
        return body.get_text(separator=' ', strip=True)
    # Fallback to all text
    return soup.get_text(separator=' ', strip=True)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)

def ingest_docs(force=False):
    docs_home = "https://docs.e6data.com/product-documentation"
    domain = "docs.e6data.com"
    if not force and os.path.exists(DOCS_FILE) and os.path.exists(META_FILE):
        print("Loading previously ingested docs...")
        with open(DOCS_FILE) as f:
            docs = json.load(f)
        with open(META_FILE) as f:
            meta = json.load(f)
        print(f"Last crawled: {meta['last_crawled']}, {len(docs)} docs loaded.")
        return docs, meta
    print("Crawling all internal documentation links...")
    urls = get_all_internal_links(docs_home, domain)
    print(f"Found {len(urls)} documentation URLs.")
    docs = []
    for idx, url in enumerate(urls, 1):
        try:
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, 'html.parser')
            text = extract_main_content(soup)
            docs.append({"url": url, "text": text})
            print(f"[{idx}/{len(urls)}] Extracted {len(text)} chars from {url}")
        except Exception as e:
            print(f"Error extracting {url}: {e}")
    save_json(docs, DOCS_FILE)
    meta = {
        "last_crawled": datetime.now().isoformat(),
        "url_count": len(urls),
        "doc_count": len(docs),
        "urls": urls[:5] + (["..."] if len(urls) > 5 else [])
    }
    save_json(meta, META_FILE)
    print(f"Crawled and saved {len(docs)} docs.")
    return docs, meta

if __name__ == "__main__":
    ingest_docs(force=True) 