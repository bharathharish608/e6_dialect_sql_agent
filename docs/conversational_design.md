# Conversational Memory Design for e6data SQL Agent

## Goal
Enable the agent to support follow-up questions and multi-turn conversations by maintaining conversational memory, so users can reference previous queries, answers, and entities (e.g., SQL, error codes, etc.).

## Memory Requirements
- Store user questions and agent responses (verbatim)
- Optionally extract and store entities (SQL, tables, error codes)
- Provide a memory digest (summary or relevant retrieval) in the prompt
- Support coreference resolution ("it", "above", "previously", etc.)
- Efficiently manage context window (summarize or retrieve as needed)

## Options Considered

### 1. **Custom Implementation**
- Full control over storage, retrieval, summarization, and entity extraction
- High engineering effort: must build history management, summarization, retrieval, and coreference logic
- Best for highly specialized needs or deep integration

### 2. **LangChain Memory**
- Mature, widely used in the LLM ecosystem
- Supports ConversationBufferMemory, ConversationSummaryMemory, EntityMemory, and more
- Handles context window management, summarization, and retrieval
- Pluggable with custom entity extraction or summarization if needed
- Good documentation and community support

### 3. **LangMem**
- Designed for robust, scalable conversational memory
- Handles summarization, retrieval, and sometimes coreference resolution
- Can be integrated with custom entity extraction or semantic retrieval
- Less mature than LangChain but focused on memory as a first-class feature

### 4. **Other Options**
- Haystack, LlamaIndex, or other frameworks with memory modules
- May be overkill or less focused on conversational memory

## Coreference Resolution
- Always include enough recent turns in the prompt for LLM to resolve references
- For advanced use, use built-in or LLM-based coreference resolution (supported in some libraries)

## Final Design Choice
- **Start with LangChain Memory** for rapid development, robust context management, and extensibility
- If limitations are encountered, consider LangMem or a custom solution for advanced memory or coreference needs
- Regularly update this design note as the implementation evolves

## Prompt Structure Example
```
Memory digest: [summary of relevant facts/entities]
Conversation so far:
User: ...
Agent: ...
...
User: [current question]
```

## Next Steps
- Prototype with LangChain Memory
- Evaluate for conversational continuity, entity tracking, and coreference
- Update this document with findings and any design changes 