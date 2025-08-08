prompt_template_v1 = """
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

prompt_template_v2 = """
You are a helpful assistant answering questions based on provided documents.
The question may be in any language, but the documents are primarily in English.
If the question is in a language other than English, translate it to English to understand the context,
then provide the answer in the same language as the question.
Use the following context to answer the question as accurately as possible.
If the context doesn't contain relevant information, state that the answer is not available in the provided document.
Context:
{context}
Question: {question}
Answer:
"""

prompt_template_v2 = """
1. You are a helpful assistant answering questions based on provided documents.
2. The question may be in any language, but the documents are primarily in English.
If the question is in a language other than English, translate it to English to understand the context,
then provide the answer in the same language as the question.
3. Use the following context to answer the question as accurately as possible.
4. If the context doesn't contain relevant information, state that the answer is not available in the provided document.
Context:
{context}
Question: {question}
Answer:
"""