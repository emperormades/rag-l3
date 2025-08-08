import logging
import os
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
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

        return vector_store, llm, supabase # Return supabase client

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

    def check_hate_speech(input_dict):
        """Checks if the generated answer contains hate speech."""
        text_to_check = input_dict["answer"]
        try:
            result = guardrail_chain.invoke({"text": text_to_check})
            if result.strip().lower() == "yes":
                input_dict["answer"] = "Sorry, I cannot respond to content that promotes hate speech or discrimination. Please rephrase your message."
                input_dict["context"] = ""
                input_dict["doc_ids"] = []
            return input_dict
        except Exception as e:
            logger.error(f"Error in hate speech guardrail: {str(e)}")
            return input_dict # Continue with original input if guardrail fails

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

    # Define a chain to retrieve documents and format them
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the RAG chain
    rag_chain = (
        {
            "context": retriever, # Pass the raw documents
            "question": RunnablePassthrough()
        }
        | RunnablePassthrough.assign(
            formatted_context=lambda x: format_docs(x["context"])
        )
        | {
            "answer": prompt | llm | StrOutputParser(), # LLM generates answer
            "context": lambda x: x["formatted_context"], # Pass formatted context
            "doc_ids": lambda x: [doc.metadata.get("id") for doc in x["context"]] # Extract doc IDs
        }
        | hate_speech_guardrail # Apply guardrail to the dictionary output
    )

    return rag_chain

def query_rag(rag_chain, question):
    """Query the RAG system with a user question."""
    try:
        # Process the query
        response_dict = rag_chain.invoke(question)
        logger.info("Query processed successfully.")
        return response_dict
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"answer": f"An error occurred: {str(e)}", "context": "", "doc_ids": []}

def authenticate_user(supabase: Client, email: str, password: str, is_signup: bool):
    """Handles user login or signup with Supabase."""
    try:
        if is_signup:
            response = supabase.auth.sign_up({"email": email, "password": password})
            if response.user:
                logger.info(f"User signed up: {response.user.email}")
                return response
            else:
                logger.error(f"Sign up error: {response.session}")
                # Attempt to extract a more specific error message
                error_message = "Unknown signup error"
                if response.session and response.session.user and response.session.user.identities:
                    if response.session.user.identities[0].identity_data and 'error' in response.session.user.identities[0].identity_data:
                        error_message = response.session.user.identities[0].identity_data['error']
                raise Exception(error_message)
        else:
            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
            if response.user:
                logger.info(f"User logged in: {response.user.email}")
                return response
            else:
                logger.error(f"Login error: {response.session}")
                # Attempt to extract a more specific error message
                error_message = "Unknown login error"
                if response.session and response.session.user and response.session.user.identities:
                    if response.session.user.identities[0].identity_data and 'error' in response.session.user.identities[0].identity_data:
                        error_message = response.session.user.identities[0].identity_data['error']
                raise Exception(error_message)
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        st.error(f"Authentication failed: {str(e)}")
        return None

def logout_user(supabase: Client):
    """Handles user logout with Supabase."""
    try:
        supabase.auth.sign_out()
        logger.info("User logged out.")
        return True
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        st.error(f"Logout failed: {str(e)}")
        return False

def save_conversation(supabase_client, user_id, question, response, context, doc_ids):
    """Saves a conversation turn to the Supabase 'conversations' table."""
    try:
        conversation_id = str(uuid.uuid4()) # Unique ID for each Q&A pair

        data = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "question": question,
            "response": response,
            "context": context,
            "doc_ids": doc_ids,
            "created_at": datetime.now().isoformat()
        }
        response = supabase_client.table("conversations").insert(data).execute()
        if response.data:
            logger.info(f"Conversation saved successfully for user {user_id}.")
        else:
            logger.error(f"Failed to save conversation: {response.error}")
    except Exception as e:
        logger.error(f"Error saving conversation: {str(e)}")

def load_conversations(supabase_client, user_id):
    """Loads all conversations for a given user from Supabase."""
    try:
        response = supabase_client.table("conversations").select("*").eq("user_id", user_id).order("created_at", desc=False).execute()
        if response.data:
            logger.info(f"Conversations loaded successfully for user {user_id}.")
            return response.data
        else:
            logger.info(f"No conversations found for user {user_id}.")
            return []
    except Exception as e:
        logger.error(f"Error loading conversations: {str(e)}")
        return []

def main():
    """Main function to run the Streamlit RAG application."""
    st.set_page_config(page_title="RAG Chatbot", page_icon="🧠", layout="wide")
    st.title("L3 Chatbot")
    st.markdown("Ask questions based on document: Which Economic Tasks are Performed with AI? Evidence from Millions of Claude Conversations")

    # Initialize session state for authentication and chat history
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "user_email" not in st.session_state:
        st.session_state["user_email"] = None
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "supabase_client" not in st.session_state:
        st.session_state["supabase_client"] = None
    if "rag_chain" not in st.session_state:
        st.session_state["rag_chain"] = None

    # Initialize Supabase and RAG components only once
    if st.session_state["supabase_client"] is None or st.session_state["rag_chain"] is None:
        try:
            vector_store, llm, supabase_client = initialize_rag_components()
            st.session_state["supabase_client"] = supabase_client
            st.session_state["rag_chain"] = create_rag_chain(vector_store, llm)
        except ValueError as e:
            st.error(f"Configuration Error: {e}. Please ensure all environment variables are set.")
            return
        except Exception as e:
            st.error(f"Initialization Error: {e}")
            return

    supabase_client = st.session_state["supabase_client"]
    rag_chain = st.session_state["rag_chain"]

    # --- Authentication Section ---
    if not st.session_state["logged_in"]:
        st.subheader("Login / Sign Up")
        auth_tab = st.tabs(["Login", "Sign Up"])

        with auth_tab[0]: # Login tab
            with st.form("login_form"):
                login_email = st.text_input("Email", key="login_email")
                login_password = st.text_input("Password", type="password", key="login_password")
                login_button = st.form_submit_button("Login")

                if login_button:
                    try:
                        user_data = authenticate_user(supabase_client, login_email, login_password, is_signup=False)
                        if user_data and user_data.user:
                            st.session_state["logged_in"] = True
                            st.session_state["user_email"] = user_data.user.email
                            st.session_state["user_id"] = user_data.user.id
                            st.success(f"Logged in as {user_data.user.email}")
                            # Load previous conversations
                            st.session_state["messages"] = [] # Clear current messages
                            loaded_convs = load_conversations(supabase_client, st.session_state["user_id"])
                            for conv in loaded_convs:
                                st.session_state["messages"].append({"role": "user", "content": conv["question"]})
                                st.session_state["messages"].append({"role": "assistant", "content": conv["response"]})
                            st.rerun() # Rerun to switch to chat view
                        else:
                            st.error("Login failed. Please check your credentials.")
                    except Exception as e:
                        st.error(f"Login error: {e}")

        with auth_tab[1]: # Sign Up tab
            with st.form("signup_form"):
                signup_email = st.text_input("Email", key="signup_email")
                signup_password = st.text_input("Password", type="password", key="signup_password")
                signup_button = st.form_submit_button("Sign Up")

                if signup_button:
                    try:
                        user_data = authenticate_user(supabase_client, signup_email, signup_password, is_signup=True)
                        if user_data and user_data.user:
                            st.session_state["logged_in"] = True
                            st.session_state["user_email"] = user_data.user.email
                            st.session_state["user_id"] = user_data.user.id
                            st.success(f"Account created and logged in as {user_data.user.email}")
                            st.rerun() # Rerun to switch to chat view
                        else:
                            st.error("Sign up failed.")
                    except Exception as e:
                        st.error(f"Sign up error: {e}")
    else:
        # --- Chat Section (Logged In) ---
        st.sidebar.write(f"Logged in as: **{st.session_state['user_email']}**")
        if st.sidebar.button("Logout"):
            logout_user(supabase_client)
            st.session_state["logged_in"] = False
            st.session_state["user_email"] = None
            st.session_state["user_id"] = None
            st.session_state["messages"] = [] # Clear messages on logout
            st.success("Logged out successfully.")
            st.rerun() # Rerun to switch to login view

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
                    response_dict = query_rag(rag_chain, question)
                    answer = response_dict.get("answer", "No answer found.")
                    context_text = response_dict.get("context", "")
                    doc_ids_list = response_dict.get("doc_ids", [])

                    st.write(answer)
                    st.session_state["messages"].append({"role": "assistant", "content": answer})

                    # Save conversation if user is logged in
                    if st.session_state["user_id"]:
                        save_conversation(
                            supabase_client,
                            st.session_state["user_id"],
                            question,
                            answer,
                            context_text,
                            doc_ids_list
                        )
                    else:
                        st.warning("You are not logged in. Conversation will not be saved.")


if __name__ == "__main__":
    main()