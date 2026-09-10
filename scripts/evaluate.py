from pathlib import Path
from tempfile import TemporaryDirectory

from agentic_rag.agent import RetrievalAgent
from agentic_rag.baseline import NaiveRetrievalBaseline
from agentic_rag.embeddings import OpenAIEmbeddingProvider
from agentic_rag.evaluation import (
    evaluate_system,
    load_evaluation_cases,
    render_comparison_table,
    render_detail_table,
    summarize_results,
)
from agentic_rag.ingestion import DocumentIngestionPipeline
from agentic_rag.llm import OpenAIChatModel
from agentic_rag.vector_store import ChromaVectorStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cases = load_evaluation_cases(ROOT / "eval" / "questions.json")
    embeddings = OpenAIEmbeddingProvider()
    language_model = OpenAIChatModel()

    with TemporaryDirectory(prefix="agentic-rag-eval-") as directory:
        vector_store = ChromaVectorStore(
            persist_directory=directory,
            collection_name="evaluation",
            embedding_identifier=embeddings.identifier,
        )
        pipeline = DocumentIngestionPipeline(
            embeddings=embeddings,
            vector_store=vector_store,
        )
        pipeline.ingest([ROOT / "eval" / "sample_corpus.txt"])

        agentic = RetrievalAgent(
            language_model=language_model,
            embeddings=embeddings,
            vector_store=vector_store,
        )
        baseline = NaiveRetrievalBaseline(
            language_model=language_model,
            embeddings=embeddings,
            vector_store=vector_store,
        )
        results = [
            *evaluate_system("Agentic", agentic, cases),
            *evaluate_system("Naive retrieval", baseline, cases),
        ]

    print("Per-question results")
    print(render_detail_table(results))
    print("\nComparison")
    print(render_comparison_table(summarize_results(results)))


if __name__ == "__main__":
    main()
