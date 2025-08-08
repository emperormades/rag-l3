import uuid
from datetime import datetime
from supabase.client import Client
from src.log import logger

def save_conversation(
    supabase_client: Client,
    user_id: str,
    question: str,
    response: str,
    context: str,
    doc_ids: list
):
    """Saves a conversation turn to the Supabase 'conversations' table."""
    try:
        conversation_id = str(uuid.uuid4())
        data = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "question": question,
            "response": response,
            "context": context,
            "doc_ids": doc_ids,
            "created_at": datetime.now().isoformat()
        }
        response_data = supabase_client.table("conversations").insert(data).execute()
        if response_data.data:
            logger.info(f"Conversation saved successfully for user {user_id}.")
        else:
            logger.error(f"Failed to save conversation: {response_data.error}")
    except Exception as e:
        logger.error(f"Error saving conversation: {str(e)}")

def load_conversations(supabase_client: Client, user_id: str):
    """Loads all conversations for a given user from Supabase."""
    try:
        response = supabase_client.table("conversations").select("*").eq("user_id", user_id).order("created_at", desc=False).execute()
        return response.data or []
    except Exception as e:
        logger.error(f"Error loading conversations: {str(e)}")
        return []