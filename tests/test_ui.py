from pathlib import Path

import pytest

from agentic_rag.models import Citation
from agentic_rag.ui import DisplayMessage, save_uploaded_document, to_chat_history


def test_save_uploaded_document_sanitizes_filename(tmp_path: Path) -> None:
    path = save_uploaded_document(
        filename="../guide.txt",
        content=b"Agentic retrieval guide",
        upload_directory=tmp_path,
    )

    assert path == tmp_path / "guide.txt"
    assert path.read_bytes() == b"Agentic retrieval guide"


def test_save_uploaded_document_rejects_unsupported_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Only PDF and text"):
        save_uploaded_document(
            filename="notes.csv",
            content=b"a,b",
            upload_directory=tmp_path,
        )


def test_to_chat_history_excludes_citations() -> None:
    citation = Citation(
        id="S1",
        source="guide.txt",
        chunk_index=0,
        text="Grounded source",
    )
    messages = [
        DisplayMessage(role="user", content="Question"),
        DisplayMessage(
            role="assistant",
            content="Answer",
            citations=(citation,),
        ),
    ]

    history = to_chat_history(messages)

    assert [message.role for message in history] == ["user", "assistant"]
    assert [message.content for message in history] == ["Question", "Answer"]
