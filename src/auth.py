import streamlit as st
from supabase.client import Client

from src.log import logger


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