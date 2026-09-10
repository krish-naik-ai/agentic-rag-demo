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
