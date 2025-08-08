from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.log.log import logger
from src.core.promp_template import prompt_template_v2


def create_rag_chain(vector_store, llm):
    """Create the RAG chain for querying and generating responses."""

    prompt = PromptTemplate.from_template(prompt_template_v2)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | RunnablePassthrough.assign(
            formatted_context=lambda x: format_docs(x["context"])
        )
        | {
            "answer": prompt | llm | StrOutputParser(),
            "context": lambda x: x["formatted_context"],
            "doc_ids": lambda x: [doc.metadata.get("id") for doc in x["context"]]
        }
    )
    return rag_chain

def query_rag(rag_chain, question):
    """Query the RAG system with a user question."""
    try:
        response_dict = rag_chain.invoke(question)
        logger.info("Query processed successfully.")
        return response_dict
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"answer": f"An error occurred: {str(e)}", "context": "", "doc_ids": []}