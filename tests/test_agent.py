from collections.abc import Sequence

import pytest

from agentic_rag.agent import AgentResponseError, RetrievalAgent
from agentic_rag.models import ChatMessage, DocumentChunk, SearchResult


class FakeLanguageModel:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return next(self._responses)


class FakeEmbeddings:
    identifier = "fake-embeddings"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]


class FakeVectorStore:
    embedding_identifier = FakeEmbeddings.identifier

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.searches: list[tuple[list[float], int]] = []

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        raise AssertionError("The retrieval agent must not inspect stored IDs.")

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        raise AssertionError("The retrieval agent must not upsert documents.")

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 4,
    ) -> list[SearchResult]:
        self.searches.append((list(query_embedding), top_k))
        return self._results


def source_result() -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id="chunk-1",
            source="guide.pdf",
            page=2,
            chunk_index=3,
            text="Agentic retrieval decides whether a lookup is necessary.",
        ),
        score=0.92,
    )


def test_agent_plans_retrieval_and_returns_citations() -> None:
    language_model = FakeLanguageModel(
        [
            '{"needs_retrieval": true, "query": "agentic retrieval decision"}',
            (
                '{"answer": "It conditionally searches the corpus.", '
                '"citation_ids": ["S1"]}'
            ),
        ]
    )
    embeddings = FakeEmbeddings()
    vector_store = FakeVectorStore([source_result()])
    agent = RetrievalAgent(language_model, embeddings, vector_store, top_k=3)

    answer = agent.answer(
        "How does agentic retrieval work?",
        history=[ChatMessage(role="user", content="Use the uploaded guide.")],
    )

    assert answer.answer == "It conditionally searches the corpus."
    assert answer.used_retrieval is True
    assert answer.retrieval_query == "agentic retrieval decision"
    assert answer.citations[0].label == "guide.pdf, page 2, chunk 3"
    assert embeddings.queries == ["agentic retrieval decision"]
    assert vector_store.searches == [([1.0, 0.0], 3)]
    assert "[S1] source=guide.pdf, page=2, chunk=3" in language_model.calls[1][1]


def test_agent_skips_retrieval_for_conversation() -> None:
    language_model = FakeLanguageModel(
        [
            '{"needs_retrieval": false, "query": ""}',
            '{"answer": "Hello! What would you like to explore?", "citation_ids": []}',
        ]
    )
    embeddings = FakeEmbeddings()
    vector_store = FakeVectorStore([])
    agent = RetrievalAgent(language_model, embeddings, vector_store)

    answer = agent.answer("Hello")

    assert answer.answer == "Hello! What would you like to explore?"
    assert answer.used_retrieval is False
    assert answer.citations == ()
    assert embeddings.queries == []
    assert vector_store.searches == []


def test_agent_reports_when_retrieval_finds_nothing() -> None:
    language_model = FakeLanguageModel(
        ['{"needs_retrieval": true, "query": "missing subject"}']
    )
    agent = RetrievalAgent(
        language_model,
        FakeEmbeddings(),
        FakeVectorStore([]),
    )

    answer = agent.answer("What does the document say about the subject?")

    assert "couldn't find relevant information" in answer.answer
    assert answer.used_retrieval is True
    assert answer.citations == ()


def test_agent_rejects_unknown_citation_ids() -> None:
    language_model = FakeLanguageModel(
        [
            '{"needs_retrieval": true, "query": "retrieval"}',
            '{"answer": "Grounded answer.", "citation_ids": ["S9"]}',
        ]
    )
    agent = RetrievalAgent(
        language_model,
        FakeEmbeddings(),
        FakeVectorStore([source_result()]),
    )

    with pytest.raises(AgentResponseError, match="unknown source IDs"):
        agent.answer("Explain retrieval.")


def test_agent_requires_citations_for_retrieved_answers() -> None:
    language_model = FakeLanguageModel(
        [
            '{"needs_retrieval": true, "query": "retrieval"}',
            '{"answer": "Uncited answer.", "citation_ids": []}',
        ]
    )
    agent = RetrievalAgent(
        language_model,
        FakeEmbeddings(),
        FakeVectorStore([source_result()]),
    )

    with pytest.raises(AgentResponseError, match="at least one source"):
        agent.answer("Explain retrieval.")


def test_agent_rejects_invalid_planner_json() -> None:
    agent = RetrievalAgent(
        FakeLanguageModel(["not-json"]),
        FakeEmbeddings(),
        FakeVectorStore([]),
    )

    with pytest.raises(AgentResponseError, match="planner returned invalid JSON"):
        agent.answer("Explain retrieval.")


def test_agent_rejects_empty_questions() -> None:
    agent = RetrievalAgent(
        FakeLanguageModel([]),
        FakeEmbeddings(),
        FakeVectorStore([]),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        agent.answer("  ")
