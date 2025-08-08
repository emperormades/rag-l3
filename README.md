# Configure Guardrails Hub
guardrails configure
guardrails hub install hub://guardrails/toxic_language --quiet

# Streamlit
streamlit run src/front.py

# LangSmith (Observability)
# Supabase (Vector Store)
# OpenAI (LLM)