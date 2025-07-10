# main_agent.py
# Main conversational agent for e6data SQL with memory and coreference resolution
# See conversational_design.md for design notes and sequential plan

import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import anthropic
from typing import TypedDict, List
from rank_bm25 import BM25Okapi
import re
import datetime

# --- LangChain Memory Integration ---
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_anthropic import ChatAnthropic

# --- Claude Sonnet LLM Wrapper for LangChain ---
class ClaudeLLMForLangChain:
    def __init__(self, api_key, model="claude-sonnet-4-20250514", max_tokens=8192):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
    def __call__(self, prompt, stop=None):
        # LangChain expects a string return value
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

# --- Load Anthropic API Key from config.env or .env ---
from dotenv import load_dotenv
if os.path.exists('config.env'):
    load_dotenv('config.env')
else:
    load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.strip() == "":
    print("ERROR: ANTHROPIC_API_KEY not found in config.env or .env. Please set it and try again.")
    exit(1)

# Initialize LangChain Memory objects
conversation_buffer_memory = ConversationBufferMemory(return_messages=True)
summary_llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key=ANTHROPIC_API_KEY)
conversation_summary_memory = ConversationSummaryMemory(llm=summary_llm)

# Define state schema for LangGraph
class DocAgentState(TypedDict):
    user_input: str
    retrieved: List[dict]
    response: str

# Load environment variables from config.env if present, else .env
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.strip() == "":
#     print("ERROR: ANTHROPIC_API_KEY not found in config.env or .env. Please set it and try again.")
#     exit(1)

# Load all chunk texts and metadatas for BM25
DATA_DIR = 'data'
CHROMA_PATH = os.path.join(DATA_DIR, 'chroma_db')
COLLECTION_NAME = "e6data_docs"
CHUNKS_FILE = os.path.join(DATA_DIR, 'chunks.json')

with open(CHUNKS_FILE) as f:
    all_chunks = json.load(f)
all_texts = [chunk['chunk'] for chunk in all_chunks]
all_metadatas = [
    {
        'url': chunk.get('url', ''),
        'source': chunk.get('source', ''),
        'section': chunk.get('section', ''),
        'chunk': chunk['chunk']
    } for chunk in all_chunks
]
# Pre-tokenize for BM25
bm25_corpus = [re.findall(r"\w+", text.lower()) for text in all_texts]
bm25 = BM25Okapi(bm25_corpus)

# Vector Search Tool Node
def vector_search_tool(query, top_k=5):
    print(f"[TOOL CALL] vector_search_tool called with query: {query}")
    # Vector search
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    vector_hits = [
        {"text": doc, **meta} for doc, meta in zip(docs, metadatas)
    ]
    for i, hit in enumerate(vector_hits):
        src = hit.get('url', hit.get('source', ''))
        print(f"[VECTOR RETRIEVED] #{i+1} Source: {src}\nSnippet: {hit['text'][:120]}\n---")
    # BM25 search
    query_tokens = re.findall(r"\w+", query.lower())
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:top_k]
    bm25_hits = []
    for i in bm25_top_idx:
        if bm25_scores[i] > 0:
            hit = all_metadatas[i].copy()
            hit['text'] = hit['chunk']  # Normalize for downstream
            bm25_hits.append(hit)
    for i, hit in enumerate(bm25_hits):
        src = hit.get('url', hit.get('source', ''))
        print(f"[BM25 RETRIEVED] #{i+1} Source: {src}\nSnippet: {hit['text'][:120]}\n---")
    # Merge and deduplicate (prioritize vector hits, then add BM25 hits not already present)
    seen = set()
    merged = []
    for hit in vector_hits + bm25_hits:
        key = (hit.get('url', ''), hit.get('text', '')[:60])
        if key not in seen:
            merged.append(hit)
            seen.add(key)
    return merged[:top_k]

# Claude LLM Node
def claude_llm_node(state):
    user_input = state["user_input"]
    retrieved = state["retrieved"]
    context = "\n\n".join([f"Source: {c.get('url', c.get('source', ''))}\n{c['text']}" for c in retrieved])
    prompt = f"""
You are an expert e6data and Apache Calcite SQL assistant. Use the following context to answer the user's question or diagnose their error. Always cite the source (url or 'apache_calcite').

Context:
{context}

User:
{user_input}

Answer:
"""
    print("\n[DEBUG] Claude prompt:\n", prompt)
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}]
        )
        print("\n[DEBUG] Claude raw response:\n", response)
        # Defensive: print content if present
        if hasattr(response, 'content') and response.content:
            return {"response": response.content[0].text}
        else:
            print("[DEBUG] Claude response missing 'content' or is empty.")
            return {"response": "[LLM call returned no content]"}
    except Exception as e:
        print(f"[EXCEPTION in claude_llm_node] {e}")
        return {"response": f"[LLM call failed: {e}]"}

def expand_query_with_llm(user_query):
    expansion_prompt = (
        "Rewrite the following user question to be as explicit and technical as possible, using synonyms and related terms, "
        "so it matches documentation and SQL idioms.\n"
        "e6data is a SQL engine for analytical querying on object store native big data ecosystems, supporting both basic and advanced SQL features.\n"
        "The documentation is at https://docs.e6data.com/\n"
        "e6data uses Apache Calcite for its sql parsing and planning\n"
        "Do not answer the question, just rewrite it for search.\n"
        "When you expand the query, ensure you do not change the meaning and the intent of the user's question. You cannot add, remove or alter the original user intent behind the question.\n"
        f"User question: {user_query}\nExpanded query:"
    )
    print("\n[QUERY EXPANSION] Original user query:", user_query)
    print("[QUERY EXPANSION] Expansion prompt sent to LLM:\n", expansion_prompt)
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": expansion_prompt}]
    )
    expanded_query = response.content[0].text.strip()
    print("[QUERY EXPANSION] Expanded query generated by LLM:\n", expanded_query)
    return expanded_query

# --- Coreference Resolution Utility ---
def resolve_coreferences_with_llm(conversation_history, user_input, max_tokens=8192):
    """
    Use Claude LLM to resolve coreferences in user_input based on the conversation_history.
    conversation_history: list of dicts [{"role": "user"|"agent", "content": str}]
    user_input: str (the new user message)
    Returns: str (user_input with coreferences resolved)
    """
    # Construct the prompt for coreference resolution
    history_str = "\n".join([
        f"{turn['role'].capitalize()}: {turn['content']}" for turn in conversation_history
    ])
    prompt = (
        "Given the following conversation, rewrite the last user message so that all references (like 'it', 'that', 'above', 'previously', etc.) are replaced with their explicit meaning, based on the conversation so far. "
        "Do not answer the question, just rewrite it with all references resolved.\n"
        f"Conversation so far:\n{history_str}\nUser: {user_input}\n\nRewritten user message:"
    )
    print("\n[COREFERENCE RESOLUTION] Prompt sent to LLM:\n", prompt)
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}]
    )
    resolved = response.content[0].text.strip()
    print("[COREFERENCE RESOLUTION] Resolved user input:\n", resolved)
    return resolved

def extract_tables(sql):
    if not sql or not isinstance(sql, str):
        return []
    # Normalize whitespace and convert to lowercase
    sql = re.sub(r'\s+', ' ', sql.lower())
    # Remove CTE definitions to avoid capturing them as tables
    # First, identify all CTE names
    cte_names = []
    cte_pattern = r'with\s+([a-z0-9_]+)\s+as\s*\('
    cte_matches = re.finditer(cte_pattern, sql)
    for match in cte_matches:
        cte_name = match.group(1).strip()
        cte_names.append(cte_name)
    # Extract tables from various clauses
    tables = []
    # FROM clause
    from_tables = re.findall(r'from\s+([a-z0-9_\.]+)', sql)
    tables.extend(from_tables)
    # JOIN clause
    join_tables = re.findall(r'join\s+([a-z0-9_\.]+)', sql)
    tables.extend(join_tables)
    # INSERT INTO clause
    insert_tables = re.findall(r'insert\s+(?:into)?\s+([a-z0-9_\.]+)', sql)
    tables.extend(insert_tables)
    # UPDATE clause
    update_tables = re.findall(r'update\s+([a-z0-9_\.]+)', sql)
    tables.extend(update_tables)
    # DELETE FROM clause
    delete_tables = re.findall(r'delete\s+from\s+([a-z0-9_\.]+)', sql)
    tables.extend(delete_tables)
    # Filter out CTE names
    filtered_tables = [table for table in tables if table not in cte_names]
    # Remove duplicates
    unique_tables = list(set(filtered_tables))
    return unique_tables

# --- Entity Extraction Utility ---
def extract_columns(sql):
    """
    Extract column names and expressions from SQL, including SELECT, GROUP BY, ORDER BY, WHERE, HAVING, and JOIN ON clauses.
    Exclude aliases (after AS) and dedupe results.
    """
    if not sql or not isinstance(sql, str):
        return []
    sql = re.sub(r'\s+', ' ', sql)
    columns = set()
    # --- SELECT columns/expressions ---
    select_pattern = re.compile(r'SELECT (.+?) FROM', re.IGNORECASE | re.DOTALL)
    select_match = select_pattern.search(sql)
    if select_match:
        select_clause = select_match.group(1)
        # Split on commas not inside parentheses (to handle SUM(a.id), etc.)
        parts = re.split(r',(?![^()]*\))', select_clause)
        for part in parts:
            # Remove aliases (after AS)
            col = re.split(r'\s+AS\s+', part, flags=re.IGNORECASE)[0]
            col = col.strip()
            # Ignore literals and wildcards
            if col and col != '*' and not re.match(r"^['\"]", col):
                columns.add(col)
    # --- GROUP BY columns ---
    group_by_pattern = re.compile(r'GROUP BY (.+?)(ORDER BY|HAVING|LIMIT|$)', re.IGNORECASE)
    group_by_match = group_by_pattern.search(sql)
    if group_by_match:
        group_by_clause = group_by_match.group(1)
        for col in group_by_clause.split(','):
            col = col.strip()
            if col:
                columns.add(col)
    # --- ORDER BY columns ---
    order_by_pattern = re.compile(r'ORDER BY (.+?)(LIMIT|$)', re.IGNORECASE)
    order_by_match = order_by_pattern.search(sql)
    if order_by_match:
        order_by_clause = order_by_match.group(1)
        for col in order_by_clause.split(','):
            col = col.strip()
            if col:
                columns.add(col)
    # --- WHERE and HAVING columns ---
    for clause in ['WHERE', 'HAVING']:
        clause_pattern = re.compile(rf'{clause} (.+?)(GROUP BY|ORDER BY|LIMIT|$)', re.IGNORECASE)
        clause_match = clause_pattern.search(sql)
        if clause_match:
            expr = clause_match.group(1)
            # Extract possible column names (simple heuristic: words with dot or underscores)
            for col in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_\.]*\b', expr):
                columns.add(col)
    # --- JOIN ON columns ---
    join_on_pattern = re.compile(r'JOIN [^ ]+ ON (.+?)(JOIN|WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|$)', re.IGNORECASE)
    for match in join_on_pattern.finditer(sql):
        on_clause = match.group(1)
        # Extract columns from join conditions (e.g., a.id = b.id)
        for col in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_\.]*\b', on_clause):
            columns.add(col)
    # --- Remove aliases (after AS) from all columns ---
    filtered_columns = set()
    for col in columns:
        # Remove anything after AS (if present)
        col = re.split(r'\s+AS\s+', col, flags=re.IGNORECASE)[0].strip()
        # Ignore if it's a literal or wildcard
        if col and col != '*' and not re.match(r"^['\"]", col):
            filtered_columns.add(col)
    return list(filtered_columns)

# Update extract_entities to use extract_columns for each SQL query

def extract_entities(user_input, agent_response):
    """
    Extract SQL queries, error messages, table names, and column names from user_input and agent_response.
    Returns a dict of entities.
    """
    entities = {}
    # Extract SQL queries (look for SELECT/INSERT/UPDATE/DELETE/WITH)
    sql_pattern = re.compile(r"\b(WITH|SELECT|INSERT|UPDATE|DELETE)[\s\S]+?;", re.IGNORECASE)
    sql_matches = sql_pattern.findall(user_input + "\n" + agent_response)
    if sql_matches:
        entities["sql"] = sql_matches
        # Extract tables and columns for each SQL
        all_tables = set()
        all_columns = set()
        for sql in sql_matches:
            for t in extract_tables(sql):
                all_tables.add(t)
            for c in extract_columns(sql):
                all_columns.add(c)
        if all_tables:
            entities["tables"] = list(all_tables)
        if all_columns:
            entities["columns"] = list(all_columns)
    # Extract error messages (look for 'error', 'exception', stack traces, or 'Query execution failed:')
    error_pattern = re.compile(r'(error|exception|Query execution failed:)[^\n]+', re.IGNORECASE)
    error_matches = error_pattern.findall(user_input + "\n" + agent_response)
    if error_matches:
        entities["errors"] = error_matches
    return entities

# --- Custom Entity Memory Implementation ---
import faiss
import threading
FAISS_DIR = os.path.join('data', 'FAISS')
os.makedirs(FAISS_DIR, exist_ok=True)
ENTITY_MEMORY_FILE = os.path.join(FAISS_DIR, 'entity_memory.jsonl')
ENTITY_FAISS_INDEX = os.path.join(FAISS_DIR, 'entity_memory.index')
ENTITY_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

class CustomEntityMemory:
    def __init__(self, embedding_model=ENTITY_EMBEDDING_MODEL, faiss_index_path=ENTITY_FAISS_INDEX, jsonl_path=ENTITY_MEMORY_FILE):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(embedding_model)
        self.faiss_index_path = faiss_index_path
        self.jsonl_path = jsonl_path
        self.lock = threading.Lock()
        self.embeddings = []  # In-memory cache
        self.metadata = []    # In-memory cache
        self.index = None
        self._load_or_init_index()
        self.last_write_log = None
        self.last_retrieve_log = None

    def _load_or_init_index(self):
        import os
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.jsonl_path):
            self._load_index()
        else:
            self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
            self.embeddings = []
            self.metadata = []

    def _load_index(self):
        import os
        import json
        self.index = faiss.read_index(self.faiss_index_path)
        self.embeddings = []
        self.metadata = []
        with open(self.jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                self.metadata.append(rec)
        # Embeddings are not needed in memory for search, only for adding new

    def add(self, conversation_id, user_input, agent_response, entities):
        import json
        # Prepare text for embedding (user_input + response + entities)
        entity_text = f"User: {user_input}\nAgent: {agent_response}\nEntities: {json.dumps(entities)}"
        embedding = self.model.encode([entity_text])[0]
        with self.lock:
            # Append to JSONL
            with open(self.jsonl_path, 'a') as f:
                f.write(json.dumps({
                    'conversation_id': conversation_id,
                    'user_input': user_input,
                    'agent_response': agent_response,
                    'entities': entities
                }) + '\n')
            # Add to FAISS
            if self.index is None:
                self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
            self.index.add(embedding.reshape(1, -1))
            self.metadata.append({
                'conversation_id': conversation_id,
                'user_input': user_input,
                'agent_response': agent_response,
                'entities': entities
            })
            faiss.write_index(self.index, self.faiss_index_path)
            self.last_write_log = f"[ENTITY_MEMORY] FAISS index updated with new embedding (conversation_id={conversation_id})"

    def get_entity_digest(self, query, top_k=4):
        # Embed the query/entities
        query_emb = self.model.encode([query])[0].reshape(1, -1)
        if self.index is None or self.index.ntotal == 0:
            self.last_retrieve_log = f"[ENTITY_MEMORY] FAISS similarity search performed, but index is empty."
            return ""
        D, I = self.index.search(query_emb, top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.metadata):
                rec = self.metadata[idx]
                results.append(f"User: {rec['user_input']}\nAgent: {rec['agent_response']}\nEntities: {rec['entities']}")
        self.last_retrieve_log = f"[ENTITY_MEMORY] FAISS similarity search performed, top_k={top_k} returned {len(results)} results."
        return '\n---\n'.join(results)

# LangGraph StateGraph
class DocAgentGraph(StateGraph):
    def __init__(self):
        super().__init__(DocAgentState)
        self.add_node("vector_search", self.vector_search_node)
        self.add_node("llm", self.llm_node)
        self.add_edge("vector_search", "llm")
        self.add_edge("llm", END)
        self.set_entry_point("vector_search")

    def vector_search_node(self, state):
        query = state["user_input"]
        retrieved = vector_search_tool(query)
        return {"user_input": query, "retrieved": retrieved}

    def llm_node(self, state):
        return {**state, **claude_llm_node(state)}

# --- Logging Setup ---
APP_LOG_DIR = 'application_logs'
CONV_LOG_DIR = 'conversation_logs'
os.makedirs(APP_LOG_DIR, exist_ok=True)
os.makedirs(CONV_LOG_DIR, exist_ok=True)
def get_log_paths():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    conv_id = f'conversation_id_{ts}'
    return (
        os.path.join(CONV_LOG_DIR, f'{conv_id}.log'),
        os.path.join(APP_LOG_DIR, f'{conv_id}.log'),
        conv_id
    )

# --- Main Conversational Loop Integration ---
def main():
    import os
    import anthropic
    import json
    import datetime
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Please set ANTHROPIC_API_KEY in your environment.")
        return
    # Initialize Anthropic client for main agent
    client = anthropic.Anthropic(api_key=api_key)
    conversation_history = []  # List of {role, content}
    conv_log_path, app_log_path, conversation_id = get_log_paths()
    print(f"e6data Conversational SQL Agent (multi-turn, with memory)\nLogging to: {conv_log_path} (conversation), {app_log_path} (application)")
    print("Type 'exit' to quit.")
    turn_idx = 0
    entity_memory = CustomEntityMemory()
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        turn_idx += 1
        # 1. Coreference resolution
        resolved_input = resolve_coreferences_with_llm(conversation_history, user_input)
        # 2. Retrieve documentation context using vector_search_tool (hybrid: vector + BM25)
        retrieval_query = resolved_input if resolved_input else user_input
        retrieved_docs = vector_search_tool(retrieval_query, top_k=5)
        # 3. Update conversation buffer memory
        conversation_buffer_memory.save_context({"input": user_input}, {"output": ""})
        # 4. Update conversation summary memory
        conversation_summary_memory.save_context({"input": user_input}, {"output": ""})
        # 5. Prepare memory digest for prompt
        buffer_digest = conversation_buffer_memory.load_memory_variables({})["history"]
        summary_digest = conversation_summary_memory.load_memory_variables({}).get("history", "")

        # --- Ensure buffer_digest and summary_digest are JSON serializable ---
        def serialize_message_obj(msg):
            # Handles LangChain message objects or dicts
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                return {"type": getattr(msg, 'type', type(msg).__name__), "content": getattr(msg, 'content', str(msg))}
            elif isinstance(msg, dict) and 'type' in msg and 'content' in msg:
                return msg
            else:
                return str(msg)
        def serialize_digest(digest):
            if isinstance(digest, list):
                return [serialize_message_obj(m) for m in digest]
            return digest
        buffer_digest_serializable = serialize_digest(buffer_digest)
        summary_digest_serializable = serialize_digest(summary_digest)
        # 6. Extract entities
        last_agent_response = conversation_history[-1]["content"] if conversation_history and conversation_history[-1]["role"] == "agent" else ""
        entities = extract_entities(user_input, last_agent_response)
        # 7. Add to custom entity memory
        entity_memory.add(conversation_id, user_input, last_agent_response, entities)
        # 8. Entity digest via FAISS similarity search
        entity_digest = entity_memory.get_entity_digest(user_input, top_k=4)
        # 9. Construct prompt for LLM (grounded in retrieved docs)
        tool_description = (
            "Available Tool:\n"
            "- vector_search_tool(query): Retrieves relevant documentation chunks using both semantic vector search and BM25 keyword search. "
            "Always uses this tool to ground answers in the official documentation."
        )
        context_str = "\n\n".join([
            f"Source: {doc.get('url', doc.get('source', ''))}\n{doc['text']}" for doc in retrieved_docs
        ])
        prompt = f"""
You are an expert SQL assistant for e6data. You must answer ONLY using the provided documentation context below, and cite the source (url or 'apache_calcite') for every fact or claim. If the answer is not in the context, say you don't know.

{tool_description}

Context:
{context_str}

Conversation summary:
{summary_digest}

Recent turns:
{buffer_digest}

Entities:
{entity_digest}

User (coreference-resolved):
{resolved_input}
"""
        # 10. Call Claude LLM for answer
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}]
            )
            agent_output = response.content[0].text.strip() if hasattr(response.content[0], 'text') else str(response.content)
            llm_error = None
        except Exception as e:
            agent_output = "[LLM ERROR] " + str(e)
            llm_error = str(e)
        print(f"Agent: {agent_output}\n")
        # 11. Update conversation history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "agent", "content": agent_output})
        # 12. Update memories with agent response
        conversation_buffer_memory.save_context({"input": user_input}, {"output": agent_output})
        conversation_summary_memory.save_context({"input": user_input}, {"output": agent_output})
        # 13. Log the turn (application log: detailed)
        app_log_entry = {
            "turn": turn_idx,
            "conversation_id": conversation_id,
            "user_input": user_input,
            "resolved_input": resolved_input,
            "retrieval_query": retrieval_query,
            "retrieved_docs": retrieved_docs,
            "agent_output": agent_output,
            "summary_digest": summary_digest_serializable,
            "buffer_digest": buffer_digest_serializable,
            "entity_digest": entity_digest,
            "entities": entities,
            "conversation_history": conversation_history.copy(),
            "timestamp": datetime.datetime.now().isoformat(),
            "entity_faiss_write_log": entity_memory.last_write_log,
            "entity_faiss_retrieve_log": entity_memory.last_retrieve_log,
            "llm_error": llm_error,
            "llm_prompt": prompt
        }
        with open(app_log_path, 'a') as logf:
            logf.write(json.dumps(app_log_entry, indent=2) + '\n')
        # 14. Log the turn (conversation log: human-readable)
        conv_log_entry = {
            "turn": turn_idx,
            "conversation_id": conversation_id,
            "user_input": user_input,
            "agent_output": agent_output,
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open(conv_log_path, 'a') as logf:
            logf.write(json.dumps(conv_log_entry, indent=2) + '\n')
    # Log final conversation on exit
    with open(app_log_path, 'a') as logf:
        logf.write("\n--- Conversation Ended ---\n")
        logf.write(json.dumps({"final_conversation_history": conversation_history}, indent=2) + '\n')
    with open(conv_log_path, 'a') as logf:
        logf.write("\n--- Conversation Ended ---\n")
        logf.write(json.dumps({"final_conversation_history": conversation_history}, indent=2) + '\n')

if __name__ == "__main__":
    main() 