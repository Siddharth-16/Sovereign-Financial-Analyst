from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_docs():
    loader = PyPDFLoader("data/Nvidia-10k.pdf")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(data)

    for chunk in chunks:
        chunk.metadata["company"] = "Nvidia"
        chunk.metadata["filing"] = "10-K"

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    print("Ingestion Complete. Vector DB created at ./chroma_db")

if __name__ == "__main__":
    ingest_docs()