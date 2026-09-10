from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic_rag.embeddings import EmbeddingProvider
from agentic_rag.llm import LanguageModel
from agentic_rag.models import (
    AgentAnswer,
    ChatMessage,
    Citation,
    SearchResult,
)
from agentic_rag.vector_store import VectorStore

PLANNER_SYSTEM_PROMPT = """You are the retrieval planner for a RAG assistant.
Decide whether the latest question needs evidence from the uploaded document
collection. Retrieval is needed for document-specific facts, summaries,
comparisons, quotations, or claims that should be grounded. Retrieval is not
needed for greetings, thanks, casual conversation, or questions about how to
use the assistant. When retrieval is needed, rewrite the question into one
concise semantic-search query.

Return JSON only:
{"needs_retrieval": true|false, "query": "search query or empty string"}
"""

ANSWER_SYSTEM_PROMPT = """You are a grounded RAG assistant. Answer only from
the supplied source chunks. If the sources do not contain the answer, say so.
Every factual claim based on a source must use one or more supplied source IDs.

Return JSON only:
{"answer": "concise answer", "citation_ids": ["S1", "S2"]}
"""

DIRECT_ANSWER_SYSTEM_PROMPT = """You are a concise assistant. The retrieval
planner determined that no document lookup is needed. Answer the conversational
question directly without claiming facts from uploaded documents.

Return JSON only:
{"answer": "concise answer", "citation_ids": []}
"""


class AgentResponseError(RuntimeError):
    pass


class _RetrievalPlanPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs_retrieval: bool
    query: str = ""


class _AnswerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citation_ids: list[str] = Field(default_factory=list)


class RetrievalAgent:
    def __init__(
        self,
        language_model: LanguageModel,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        top_k: int = 4,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if embeddings.identifier != vector_store.embedding_identifier:
            raise ValueError(
                "Embedding provider and vector store identifiers must match."
            )
        self._language_model = language_model
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._top_k = top_k

    def answer(
        self,
        question: str,
        history: Sequence[ChatMessage] = (),
    ) -> AgentAnswer:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty.")

        plan = self._plan(normalized_question, history)
        if not plan.needs_retrieval:
            payload = self._complete_answer(
                system_prompt=DIRECT_ANSWER_SYSTEM_PROMPT,
                user_prompt=self._conversation_prompt(normalized_question, history),
            )
            return AgentAnswer(
                answer=payload.answer,
                citations=(),
                used_retrieval=False,
            )

        query = plan.query.strip() or normalized_question
        query_embedding = self._embeddings.embed_query(query)
        results = self._vector_store.search(query_embedding, top_k=self._top_k)
        if not results:
            return AgentAnswer(
                answer=(
                    "I couldn't find relevant information in the uploaded documents."
                ),
                citations=(),
                used_retrieval=True,
                retrieval_query=query,
            )

        source_map, context = self._format_sources(results)
        payload = self._complete_answer(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_prompt=(
                f"Question:\n{normalized_question}\n\n"
                f"Conversation:\n{self._format_history(history)}\n\n"
                f"Sources:\n{context}"
            ),
        )
        citation_ids = list(dict.fromkeys(payload.citation_ids))
        invalid_ids = [
            citation_id for citation_id in citation_ids if citation_id not in source_map
        ]
        if invalid_ids:
            raise AgentResponseError(
                f"Model cited unknown source IDs: {', '.join(invalid_ids)}."
            )
        if not citation_ids:
            raise AgentResponseError("Retrieved answers must cite at least one source.")

        citations = tuple(
            self._to_citation(citation_id, source_map[citation_id])
            for citation_id in citation_ids
        )
        return AgentAnswer(
            answer=payload.answer,
            citations=citations,
            used_retrieval=True,
            retrieval_query=query,
        )

    def _plan(
        self,
        question: str,
        history: Sequence[ChatMessage],
    ) -> _RetrievalPlanPayload:
        response = self._language_model.complete_json(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=self._conversation_prompt(question, history),
        )
        try:
            return _RetrievalPlanPayload.model_validate_json(response)
        except ValidationError as error:
            raise AgentResponseError(
                "Retrieval planner returned invalid JSON."
            ) from error

    def _complete_answer(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> _AnswerPayload:
        response = self._language_model.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        try:
            return _AnswerPayload.model_validate_json(response)
        except ValidationError as error:
            raise AgentResponseError("Answer model returned invalid JSON.") from error

    @staticmethod
    def _conversation_prompt(
        question: str,
        history: Sequence[ChatMessage],
    ) -> str:
        return (
            f"Conversation:\n{RetrievalAgent._format_history(history)}\n\n"
            f"Latest question:\n{question}"
        )

    @staticmethod
    def _format_history(history: Sequence[ChatMessage]) -> str:
        if not history:
            return "(none)"
        return "\n".join(f"{message.role}: {message.content}" for message in history)

    @staticmethod
    def _format_sources(
        results: Sequence[SearchResult],
    ) -> tuple[dict[str, SearchResult], str]:
        source_map: dict[str, SearchResult] = {}
        blocks: list[str] = []
        for index, result in enumerate(results, start=1):
            source_id = f"S{index}"
            source_map[source_id] = result
            page = (
                f", page={result.chunk.page}" if result.chunk.page is not None else ""
            )
            blocks.append(
                f"[{source_id}] source={result.chunk.source}{page}, "
                f"chunk={result.chunk.chunk_index}, score={result.score:.3f}\n"
                f"{result.chunk.text}"
            )
        return source_map, "\n\n".join(blocks)

    @staticmethod
    def _to_citation(source_id: str, result: SearchResult) -> Citation:
        return Citation(
            id=source_id,
            source=result.chunk.source,
            page=result.chunk.page,
            chunk_index=result.chunk.chunk_index,
            text=result.chunk.text,
        )
