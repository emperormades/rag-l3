import logging, os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import  PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from supabase.client import create_client, Client


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

def initialize_rag_components():
    """Initialize Supabase client, embeddings, vector store, and LLM."""
    try:
        # Load environment variables
        load_dotenv()
        validate_environment_variables()

        # Initialize Supabase client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully.")

        # Initialize embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)
        logger.info("OpenAI embeddings initialized.")

        # Initialize vector store
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        logger.info("Supabase vector store initialized.")

        # Initialize LLM
        # Temperature of 0.7 balances creativity and coherence in responses
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        logger.info("ChatOpenAI model initialized.")

        return vector_store, llm

    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

def create_hate_speech_guardrail(llm):
    """Create a guardrail using OpenAI to detect hate speech."""
    guardrail_prompt = ChatPromptTemplate.from_template(
        """
        You are a content moderator. Analyze the following text and determine if it contains hate speech, defined as language that promotes hatred, discrimination, or violence against individuals or groups based on characteristics such as race, ethnicity, religion, gender, sexual orientation, or others. Respond only with "Yes" (if hate speech is present) or "No" (if not present).

        Text: {text}

        Response:
        """
    )
    guardrail_chain = guardrail_prompt | llm | StrOutputParser()

    def check_hate_speech(text):
        try:
            result = guardrail_chain.invoke({"text": text})
            if result.strip() == "Yes":
                return "Sorry, I cannot respond to content that promotes hate speech or discrimination. Please rephrase your message."
            return text
        except Exception as e:
            logger.error(f"Error in hate speech guardrail: {str(e)}")
            return text  # Continue if guardrail fails to avoid blocking legitimate content

    return RunnableLambda(check_hate_speech)

def create_rag_chain(vector_store, llm):
    """Create the RAG chain for querying and generating responses."""

    prompt_template = """
    You are a helpful assistant answering questions based on provided documents.
    The question may be in any language, but the documents are primarily in English.
    If the question is in a language other than English, translate it to English to understand the context,
    then provide the answer in the same language as the question.
    Use the following context to answer the question as accurately as possible.
    If the context doesn't contain relevant information, say so and provide a general response.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template)

    # Define the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Retrieve top 4 relevant documents
    )

    # Create the guardrail
    hate_speech_guardrail = create_hate_speech_guardrail(llm)

    # Create the RAG chain
    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), "question": RunnablePassthrough()}
        | prompt
        | llm
        | hate_speech_guardrail
        | StrOutputParser()
    )

    return rag_chain


def query_rag(rag_chain, question):
    """Query the RAG system with a user question."""
    try:
        # Process the query
        response = rag_chain.invoke(question)
        logger.info("Query processed successfully.")
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        # Return guardrail message if hate speech was detected
        if "hate speech" in str(e).lower():
            return "Sorry, I cannot respond to content that promotes hate speech or discrimination. Please rephrase your message."
        raise

def main():
    """Main function to run the Streamlit RAG application."""
    try:
        # Set Streamlit page configuration
        st.set_page_config(page_title="RAG Chatbot", page_icon="🧠", layout="wide")
        st.title("L3 Chatbot")
        st.markdown("Ask questions based on document: Which Economic Tasks are Performed with AI? Evidence from Millions of Claude Conversations")

        # Initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Initialize RAG components
        vector_store, llm = initialize_rag_components()
        rag_chain = create_rag_chain(vector_store, llm)

        # Display chat history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Input for user question
        question = st.chat_input("Enter your question:")

        if question:
            # Add user question to chat history
            st.session_state["messages"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            # Query RAG system and display response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = query_rag(rag_chain, question)
                    st.write(response)
                    st.session_state["messages"].append({"role": "assistant", "content": response})

    except Exception as e:
        logger.error(f"An error occurred in main loop: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()