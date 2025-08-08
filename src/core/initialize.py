import os

from dotenv import load_dotenv
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from supabase.client import create_client, Client

from src.log.log import logger


def validate_environment_variables():
    """Validate that required environment variables are set."""
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "OPENAI_API_KEY"]
    for var in required_vars:
        if not os.environ.get(var):
            logger.error(f"Environment variable {var} is not set.")
            raise ValueError(f"Environment variable {var} is not set.")
    logger.info("All required environment variables are set.")

def initialize_rag_components():
    """Initialize Supabase client, embeddings, vector store, and LLM."""
    try:
        load_dotenv()
        validate_environment_variables()
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully.")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)
        logger.info("OpenAI embeddings initialized.")
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        logger.info("Supabase vector store initialized.")
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        logger.info("ChatOpenAI model initialized.")
        return vector_store, llm, supabase
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise