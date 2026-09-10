from collections.abc import Sequence
from pathlib import Path

import pytest

from agentic_rag.evaluation import (
    EvaluationCase,
    evaluate_system,
    load_evaluation_cases,
    render_comparison_table,
    render_detail_table,
    summarize_results,
)
from agentic_rag.models import AgentAnswer, ChatMessage, Citation


class FakeAnswerer:
    def __init__(self, answers: Sequence[AgentAnswer]) -> None:
        self._answers = iter(answers)

    def answer(
        self,
        question: str,
        history: Sequence[ChatMessage] = (),
    ) -> AgentAnswer:
        return next(self._answers)


def test_evaluation_scores_answers_retrieval_and_citations() -> None:
    cases = [
        EvaluationCase(
            id="grounded",
            question="Which store?",
            expected_terms=["Chroma", "local"],
            should_retrieve=True,
        ),
        EvaluationCase(
            id="chat",
            question="Hello",
            expected_terms=[],
            should_retrieve=False,
        ),
    ]
    citation = Citation(
        id="S1",
        source="guide.txt",
        chunk_index=0,
        text="Chroma is local.",
    )
    answerer = FakeAnswerer(
        [
            AgentAnswer(
                answer="Chroma is used.",
                citations=(citation,),
                used_retrieval=True,
            ),
            AgentAnswer(
                answer="Hello!",
                citations=(),
                used_retrieval=False,
            ),
        ]
    )

    results = evaluate_system("Agentic", answerer, cases)
    summaries = summarize_results(results)

    assert results[0].answer_score == 0.5
    assert results[0].retrieval_score == 1.0
    assert results[0].citation_score == 1.0
    assert summaries[0].answer_score == 0.75
    assert summaries[0].overall_score == pytest.approx(11 / 12)


def test_tables_render_percentages() -> None:
    result = evaluate_system(
        "Agentic",
        FakeAnswerer(
            [
                AgentAnswer(
                    answer="Chroma",
                    citations=(),
                    used_retrieval=False,
                )
            ]
        ),
        [
            EvaluationCase(
                id="case-1",
                question="Question",
                expected_terms=["Chroma"],
                should_retrieve=False,
            )
        ],
    )

    assert "| Agentic | case-1 | 100% | 100% | 100% | 100% |" in (
        render_detail_table(result)
    )
    assert "| Agentic | 100% | 100% | 100% | 100% |" in (
        render_comparison_table(summarize_results(result))
    )


def test_load_evaluation_cases(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    questions.write_text(
        (
            '[{"id":"one","question":"Hello","expected_terms":[],'
            '"should_retrieve":false}]'
        ),
        encoding="utf-8",
    )

    cases = load_evaluation_cases(questions)

    assert cases[0].id == "one"
    assert cases[0].should_retrieve is False


def test_evaluation_requires_cases() -> None:
    with pytest.raises(ValueError, match="At least one"):
        evaluate_system("Agentic", FakeAnswerer([]), [])
