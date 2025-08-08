import streamlit as st

from src.auth import authenticate_user, logout_user
from src.initialize import initialize_rag_components
from src.rag import create_rag_chain, query_rag
from src.supabase_utils import save_conversation, load_conversations
from src.translations import (
    is_english,
    is_portuguese,
    translate_to_english,
    translate_to_portuguese
)

def main():
    """Main function to run the Streamlit RAG application."""
    st.set_page_config(page_title="RAG Chatbot", page_icon="🧠", layout="wide")
    st.title("L3 Chatbot")
    st.markdown("Ask questions based on document: Which Economic Tasks are Performed with AI? Evidence from Millions of Claude Conversations")
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
    if not st.session_state["logged_in"]:
        st.subheader("Login / Sign Up")
        # Envolver toda a seção de login/signup em uma coluna de 30%
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            auth_tab = st.tabs(["Login", "Sign Up"])
            with auth_tab[0]:
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
                                st.session_state["messages"] = []
                                loaded_convs = load_conversations(supabase_client, st.session_state["user_id"])
                                for conv in loaded_convs:
                                    st.session_state["messages"].append({"role": "user", "content": conv["question"]})
                                    st.session_state["messages"].append({"role": "assistant", "content": conv["response"]})
                                st.rerun()
                            else:
                                st.error("Login failed. Please check your credentials.")
                        except Exception as e:
                            st.error(f"Login error: {e}")
            with auth_tab[1]:
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
                                st.rerun()
                            else:
                                st.error("Sign up failed.")
                        except Exception as e:
                            st.error(f"Sign up error: {e}")
    else:
        st.sidebar.write(f"Logged in as: **{st.session_state['user_email']}**")
        if st.sidebar.button("Logout"):
            logout_user(supabase_client)
            st.session_state["logged_in"] = False
            st.session_state["user_email"] = None
            st.session_state["user_id"] = None
            st.session_state["messages"] = []
            st.success("Logged out successfully.")
            st.rerun()
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        question = st.chat_input("Enter your question:")
        if question:
            st.session_state["messages"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if is_portuguese(question):
                        translated_question = translate_to_english(question)
                        response_dict = query_rag(rag_chain, translated_question)
                    else:
                        response_dict = query_rag(rag_chain, question)
                    answer = response_dict.get("answer", "No answer found.")
                    context_text = response_dict.get("context", "")
                    doc_ids_list = response_dict.get("doc_ids", [])
                    if is_english(answer) and is_portuguese(question):
                        translated_answer = translate_to_portuguese(answer)
                        st.write(translated_answer)
                    else:
                        st.write(answer)
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
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