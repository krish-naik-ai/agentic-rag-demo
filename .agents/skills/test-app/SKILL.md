---
name: test-app
description: Install, validate, run, and browser-test the Agentic RAG application before pull requests.
---

# Test the Agentic RAG app

Run every step in order from the repository root. Stop and fix failures.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pre-commit install
```

`OPENAI_API_KEY` must come from the environment. Never write it to a file.

## 2. Static checks

```bash
source .venv/bin/activate
python -m ruff check .
python -m ruff format --check .
python -m mypy src tests
```

When `app.py` or `scripts/evaluate.py` exists, include the changed entrypoint
in the mypy command.

## 3. Automated tests

Run the affected test module while iterating, then the complete suite:

```bash
source .venv/bin/activate
python -m pytest -q
```

The default suite must not make live OpenAI calls.

## 4. Run the application

For a UI-changing PR:

```bash
source .venv/bin/activate
streamlit run app.py --server.headless true --server.port 8501
```

Wait for Streamlit to report its local URL.

## 5. Browser test and record UI changes

For a UI-changing PR, record a browser session that:

1. Opens the Streamlit app.
2. Uploads a small `.txt` document.
3. Confirms ingestion succeeds and reports stored chunks.
4. Asks a question whose answer is in the document.
5. Confirms the response contains at least one source citation.
6. Asks a conversational question that does not require retrieval.
7. Confirms the app remains usable after both responses.

Attach the recording to the PR. If live OpenAI access prevents the flow,
report the exact untested step rather than claiming success.
