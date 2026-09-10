# Agentic RAG demo

An end-to-end Retrieval-Augmented Generation application with:

- PDF and text ingestion.
- OpenAI embeddings and generation.
- A persistent local Chroma vector store.
- An agent that decides whether and how to retrieve.
- Cited answers.
- A baseline comparison evaluation harness.
- A Streamlit upload and chat interface.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pre-commit install
```

Export `OPENAI_API_KEY` in the shell before running features that call OpenAI.
Do not store it in the repository.

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy src tests
python -m pytest -q
```

The ingestion, retrieval, evaluation, and UI commands are documented as
their implementation slices are added.

## Ingest documents

The ingestion package accepts `.pdf` and `.txt` files, chunks them with
overlap, embeds each chunk with OpenAI, and upserts it into persistent Chroma:

```python
from pathlib import Path

from agentic_rag.embeddings import OpenAIEmbeddingProvider
from agentic_rag.ingestion import DocumentIngestionPipeline
from agentic_rag.vector_store import ChromaVectorStore

pipeline = DocumentIngestionPipeline(
    embeddings=OpenAIEmbeddingProvider(),
    vector_store=ChromaVectorStore("data/chroma"),
)
result = pipeline.ingest([Path("document.pdf")])
print(result)
```

## Ask with agentic retrieval

`RetrievalAgent` first asks the model whether document evidence is needed.
When it is, the agent rewrites the search query, retrieves relevant chunks,
and only accepts answers citing known source IDs.

```python
from agentic_rag.agent import RetrievalAgent
from agentic_rag.embeddings import OpenAIEmbeddingProvider
from agentic_rag.llm import OpenAIChatModel
from agentic_rag.vector_store import ChromaVectorStore

agent = RetrievalAgent(
    language_model=OpenAIChatModel(),
    embeddings=OpenAIEmbeddingProvider(),
    vector_store=ChromaVectorStore("data/chroma"),
)
answer = agent.answer("What does the document say about agentic retrieval?")
print(answer.answer)
for citation in answer.citations:
    print(citation.label)
```
