import os
from dotenv import load_dotenv
from pathlib import Path
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore


# Load environment variables
load_dotenv()

# Client automatically uses os.getenv("GEMINI_API_KEY")
client = genai.Client()

# 1. Load the PDF document
# PyPDFLoader parses the PDF and returns a list of Document objects (one per page)
pdf_path = Path(__file__).parent / "nodejs.pdf"
loader = PyPDFLoader(file_path=pdf_path)

docs = loader.load()

# 2. Initialize the text splitter
# Chunk size is measured in characters. Overlap prevents context loss between blocks.
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
)

# 3. Split the documents into smaller chunks
# This automatically handles individual pages and preserves metadata like page numbers
split_docs = text_spliter.split_documents(documents=docs)

embedder = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
)

#vector_store = QdrantVectorStore.from_documents(
#    documents=[],
#    url="http://localhost:6333",
#    collection_name="learning rag",
#    embedding=embedder
#)

#vector_store.add_documents(documents=split_docs)
print("Injection Done")

retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning rag",
    embedding=embedder
)

relevant_chunks = retriver.similarity_search(
    query="What is FS Module?"
)

print("Relevent Chunks",relevant_chunks)