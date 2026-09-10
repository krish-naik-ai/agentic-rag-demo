# Agentic RAG contributor guide

## Project structure

- `src/agentic_rag/`: application and domain logic.
- `tests/`: unit and integration tests mirroring the source package.
- `scripts/`: command-line utilities, including the evaluation harness.
- `app.py`: Streamlit entrypoint.
- `.agents/skills/`: repeatable repository workflows.
- `data/`: local uploads and Chroma persistence; never committed.

Keep OpenAI and vector-store integrations behind small interfaces so core
logic can be tested with deterministic fakes.

## Conventions

- Target Python 3.10 or newer and add type hints to public functions.
- Prefer focused modules and immutable dataclasses or Pydantic models for
  data passed between ingestion, retrieval, generation, and evaluation.
- Tests are required for core logic and bug fixes.
- Never put secrets, API keys, uploaded documents, or local vector data in
  source control. Read `OPENAI_API_KEY` from the environment.
- Do not make live OpenAI calls in the default test suite.
- Run Ruff lint and format checks, mypy, and pytest before every PR.
- Invoke the `test-app` skill before every PR. For UI changes, also run the
  Streamlit app, test the upload/chat flow in a browser, and attach a video.

## Git workflow

- Keep commits small and independently working.
- Use one PR per working slice.
- Do not bypass pre-commit hooks.
