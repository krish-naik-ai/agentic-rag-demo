from dataclasses import dataclass
from pathlib import Path
from typing import cast

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from agentic_rag.agent import RetrievalAgent
from agentic_rag.embeddings import OpenAIEmbeddingProvider
from agentic_rag.ingestion import DocumentIngestionPipeline
from agentic_rag.llm import OpenAIChatModel
from agentic_rag.ui import DisplayMessage, save_uploaded_document, to_chat_history
from agentic_rag.vector_store import ChromaVectorStore

DATA_DIRECTORY = Path("data")
UPLOAD_DIRECTORY = DATA_DIRECTORY / "uploads"
CHROMA_DIRECTORY = DATA_DIRECTORY / "chroma"


@dataclass(frozen=True)
class AppServices:
    ingestion: DocumentIngestionPipeline
    agent: RetrievalAgent


@st.cache_resource
def build_services() -> AppServices:
    embeddings = OpenAIEmbeddingProvider()
    vector_store = ChromaVectorStore(
        CHROMA_DIRECTORY,
        embedding_identifier=embeddings.identifier,
    )
    return AppServices(
        ingestion=DocumentIngestionPipeline(
            embeddings=embeddings,
            vector_store=vector_store,
        ),
        agent=RetrievalAgent(
            language_model=OpenAIChatModel(),
            embeddings=embeddings,
            vector_store=vector_store,
        ),
    )


def get_messages() -> list[DisplayMessage]:
    stored_messages = st.session_state.get("messages")
    if stored_messages is None:
        messages: list[DisplayMessage] = []
        st.session_state["messages"] = messages
        return messages
    return cast(list[DisplayMessage], stored_messages)


def ingest_uploads(
    uploads: list[UploadedFile],
    services: AppServices,
) -> None:
    paths = [
        save_uploaded_document(
            filename=upload.name,
            content=upload.getvalue(),
            upload_directory=UPLOAD_DIRECTORY,
        )
        for upload in uploads
    ]
    result = services.ingestion.ingest(paths)
    st.success(
        f"Ingested {result.chunks_stored} chunks from "
        f"{result.documents_loaded} document sections."
    )


def render_message(message: DisplayMessage) -> None:
    with st.chat_message(message.role):
        st.markdown(message.content)
        if message.citations:
            st.caption(
                "Sources: "
                + ", ".join(
                    f"[{citation.id}] {citation.label}"
                    for citation in message.citations
                )
            )
            for citation in message.citations:
                with st.expander(f"[{citation.id}] {citation.label}"):
                    st.write(citation.text)


def main() -> None:
    st.set_page_config(page_title="Agentic RAG", page_icon="📚")
    st.title("Agentic RAG")
    st.caption(
        "Upload PDF or text documents, then ask grounded questions with citations."
    )

    services = build_services()
    messages = get_messages()

    with st.sidebar:
        st.header("Documents")
        uploads = st.file_uploader(
            "Upload files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        ingest_clicked = st.button(
            "Ingest documents",
            type="primary",
            disabled=not uploads,
            use_container_width=True,
        )
        if ingest_clicked and uploads:
            with st.spinner("Embedding and storing document chunks..."):
                ingest_uploads(uploads, services)

        if st.button("Clear chat", use_container_width=True):
            messages.clear()
            st.rerun()

    for message in messages:
        render_message(message)

    question = st.chat_input("Ask about your documents")
    if question:
        user_message = DisplayMessage(role="user", content=question)
        render_message(user_message)
        history = to_chat_history(messages)
        messages.append(user_message)

        with st.spinner("Thinking..."):
            try:
                answer = services.agent.answer(question, history)
            except Exception as error:
                st.error(f"Unable to answer the question: {error}")
            else:
                assistant_message = DisplayMessage(
                    role="assistant",
                    content=answer.answer,
                    citations=answer.citations,
                )
                messages.append(assistant_message)
                render_message(assistant_message)


if __name__ == "__main__":
    main()
