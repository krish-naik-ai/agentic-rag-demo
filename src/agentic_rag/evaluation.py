import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from agentic_rag.models import AgentAnswer, ChatMessage


class QuestionAnswerer(Protocol):
    def answer(
        self,
        question: str,
        history: Sequence[ChatMessage] = (),
    ) -> AgentAnswer: ...


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_terms: list[str]
    should_retrieve: bool


@dataclass(frozen=True)
class EvaluationResult:
    system: str
    case_id: str
    answer_score: float
    retrieval_score: float
    citation_score: float

    @property
    def overall_score(self) -> float:
        return (self.answer_score + self.retrieval_score + self.citation_score) / 3


@dataclass(frozen=True)
class EvaluationSummary:
    system: str
    answer_score: float
    retrieval_score: float
    citation_score: float
    overall_score: float


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    adapter = TypeAdapter(list[EvaluationCase])
    return adapter.validate_json(path.read_text(encoding="utf-8"))


def evaluate_system(
    system_name: str,
    answerer: QuestionAnswerer,
    cases: Sequence[EvaluationCase],
) -> list[EvaluationResult]:
    if not cases:
        raise ValueError("At least one evaluation case is required.")

    results: list[EvaluationResult] = []
    for case in cases:
        answer = answerer.answer(case.question)
        results.append(
            EvaluationResult(
                system=system_name,
                case_id=case.id,
                answer_score=_answer_score(answer.answer, case.expected_terms),
                retrieval_score=float(answer.used_retrieval == case.should_retrieve),
                citation_score=float(bool(answer.citations) == case.should_retrieve),
            )
        )
    return results


def summarize_results(
    results: Sequence[EvaluationResult],
) -> list[EvaluationSummary]:
    grouped: dict[str, list[EvaluationResult]] = {}
    for result in results:
        grouped.setdefault(result.system, []).append(result)

    return [
        EvaluationSummary(
            system=system,
            answer_score=_mean(item.answer_score for item in system_results),
            retrieval_score=_mean(item.retrieval_score for item in system_results),
            citation_score=_mean(item.citation_score for item in system_results),
            overall_score=_mean(item.overall_score for item in system_results),
        )
        for system, system_results in grouped.items()
    ]


def render_comparison_table(summaries: Sequence[EvaluationSummary]) -> str:
    headers = (
        "| System | Answer | Retrieval decision | Citations | Overall |\n"
        "|---|---:|---:|---:|---:|"
    )
    rows = [
        (
            f"| {summary.system} | {_percent(summary.answer_score)} | "
            f"{_percent(summary.retrieval_score)} | "
            f"{_percent(summary.citation_score)} | "
            f"{_percent(summary.overall_score)} |"
        )
        for summary in summaries
    ]
    return "\n".join([headers, *rows])


def render_detail_table(results: Sequence[EvaluationResult]) -> str:
    headers = (
        "| System | Case | Answer | Retrieval | Citations | Overall |\n"
        "|---|---|---:|---:|---:|---:|"
    )
    rows = [
        (
            f"| {result.system} | {result.case_id} | "
            f"{_percent(result.answer_score)} | "
            f"{_percent(result.retrieval_score)} | "
            f"{_percent(result.citation_score)} | "
            f"{_percent(result.overall_score)} |"
        )
        for result in results
    ]
    return "\n".join([headers, *rows])


def _answer_score(answer: str, expected_terms: Sequence[str]) -> float:
    if not expected_terms:
        return 1.0
    normalized_answer = _normalize(answer)
    matches = sum(_normalize(term) in normalized_answer for term in expected_terms)
    return matches / len(expected_terms)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _mean(values: Iterable[float]) -> float:
    collected_values = list(values)
    if not collected_values:
        raise ValueError("Cannot average an empty sequence.")
    return sum(collected_values) / len(collected_values)


def _percent(score: float) -> str:
    return f"{score:.0%}"
