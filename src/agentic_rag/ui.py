from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agentic_rag.models import ChatMessage, Citation


@dataclass(frozen=True)
class DisplayMessage:
    role: Literal["user", "assistant"]
    content: str
    citations: tuple[Citation, ...] = ()


def save_uploaded_document(
    *,
    filename: str,
    content: bytes,
    upload_directory: Path,
) -> Path:
    safe_filename = Path(filename).name
    if not safe_filename:
        raise ValueError("Uploaded document must have a filename.")
    if Path(safe_filename).suffix.lower() not in {".pdf", ".txt"}:
        raise ValueError("Only PDF and text documents are supported.")

    upload_directory.mkdir(parents=True, exist_ok=True)
    destination = upload_directory / safe_filename
    destination.write_bytes(content)
    return destination


def to_chat_history(messages: Sequence[DisplayMessage]) -> tuple[ChatMessage, ...]:
    return tuple(
        ChatMessage(role=message.role, content=message.content) for message in messages
    )
