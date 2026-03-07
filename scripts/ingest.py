import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import CHROMA_PATH, EMBED_MODEL

# app.config already calls load_dotenv()
if not os.getenv("HF_TOKEN"):
    print("Warning: HF_TOKEN not set. HuggingFace downloads may be rate limited.")


def ingest_docs(pdf_path: str, company: str, filing_type: str = "10-K"):
    """
    Ingest a PDF filing into the local ChromaDB vector store.

    Args:
        pdf_path:    Path to the PDF file, e.g. "data/Nvidia-10k.pdf"
        company:     Company name tag, e.g. "Nvidia"
        filing_type: Filing type tag, e.g. "10-K"
    """
    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(data)

    for chunk in chunks:
        chunk.metadata["company"] = company
        chunk.metadata["filing"] = filing_type

    print(f"Embedding {len(chunks)} chunks for {company}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    print(f"Ingestion complete — {company} {filing_type} stored at {CHROMA_PATH}")


if __name__ == "__main__":
    # Add more entries here as you ingest new filings
    filings = [
        ("data/Nvidia-10k.pdf", "Nvidia", "10-K"),
        # ("data/Apple-10k.pdf",  "Apple",  "10-K"),
        # ("data/Tesla-10k.pdf",  "Tesla",  "10-K"),
    ]

    for pdf_path, company, filing_type in filings:
        ingest_docs(pdf_path, company, filing_type)