import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from supabase.client import Client, create_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def validate_environment_variables():
    """Validate that required environment variables are set."""
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "OPENAI_API_KEY"]
    for var in required_vars:
        if not os.environ.get(var):
            logger.error(f"Environment variable {var} is not set.")
            raise ValueError(f"Environment variable {var} is not set.")
    logger.info("All required environment variables are set.")

def preprocess_document_text(documents):
    """Preprocess document text to handle OCR artifacts."""
    for doc in documents:
        # Remove repetitive "technical, technical" sequences
        doc.page_content = doc.page_content.replace(", technical" * 10, "")
        # Handle truncated text markers (e.g., "truncated 160 characters")
        doc.page_content = doc.page_content.replace("(truncated 160 characters)", "")
    return documents

def main():
    try:
        # Load environment variables
        load_dotenv()
        validate_environment_variables()

        # Initialize Supabase client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully.")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        logger.info("OpenAI embeddings initialized.")

        docs_path = Path("docs")
        if not docs_path.exists() or not docs_path.is_dir():
            logger.error("Directory 'docs' does not exist or is not a directory.")
            raise FileNotFoundError("Directory 'docs' does not exist or is not a directory.")

        loader = PyPDFDirectoryLoader("docs")
        documents = loader.load()
        if not documents:
            logger.error("No PDF documents found in 'docs' directory.")
            raise ValueError("No PDF documents found in 'docs' directory.")
        logger.info(f"Loaded {len(documents)} PDF documents.")


        documents = preprocess_document_text(documents)
        logger.info("Document preprocessing completed.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            add_start_index=True,
        )
        docs = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(docs)} chunks.")

        vector_store = SupabaseVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
            chunk_size=500,
        )
        logger.info("Documents successfully stored in Supabase vector store.")

        return vector_store

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
