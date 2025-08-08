from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

from src.log import logger
from src.promp_template import prompt_template_v2


# Define the state for the graph
# Define the state for the graph
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question.
        context: The retrieved context from the vector store.
        answer: The final generated answer.
        doc_ids: A list of document IDs from the retrieved context.
        documents: A list of Document objects.
        chat_history: A list of messages representing the conversation history.
    """
    question: str
    context: str
    answer: str
    doc_ids: List[str]
    documents: List[Document]
    chat_history: List[BaseMessage]

def create_rag_chain(vector_store, llm):
    """
    Creates the RAG chain using LangGraph with conversation history.

    Args:
        vector_store: The Supabase vector store instance.
        llm: The ChatOpenAI model instance.

    Returns:
        A compiled LangGraph state graph.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Node 1: Retrieve documents
    def retrieve_node(state: GraphState):
        question = state['question']
        docs = retriever.invoke(question)
        doc_ids = [doc.metadata.get("id") for doc in docs]

        # Format the documents for the prompt
        formatted_context = "\n\n".join(doc.page_content for doc in docs)

        return {
            "documents": docs,
            "doc_ids": doc_ids,
            "context": formatted_context,
            "question": question,
            "chat_history": state.get("chat_history", [])
        }

    # Node 2: Generate the answer
    def generate_node(state: GraphState):
        # The prompt template needs to include a placeholder for chat history
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template_v2),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )

        llm_chain = prompt_template | llm | StrOutputParser()

        answer = llm_chain.invoke({
            "context": state['context'],
            "question": state['question'],
            "chat_history": state['chat_history']
        })

        return {
            "answer": answer,
            "question": state['question'],
            "context": state['context'],
            "doc_ids": state['doc_ids'],
            "chat_history": state['chat_history']
        }

    # Build the graph
    workflow = StateGraph(GraphState)

    # Add the nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # Set the entry point of the graph
    workflow.set_entry_point("retrieve")

    # Add the edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile the graph
    rag_chain = workflow.compile()

    return rag_chain

def query_rag(rag_chain, question, chat_history):
    """
    Queries the RAG system with a user question using LangGraph.

    Args:
        rag_chain: The compiled LangGraph chain.
        question: The user's question string.
        chat_history: The list of `BaseMessage` objects representing the conversation.

    Returns:
        A dictionary containing the answer, context, and document IDs.
    """
    try:
        # LangGraph's invoke call returns the final state dictionary
        response_dict = rag_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })

        logger.info("Query processed successfully.")
        return {
            "answer": response_dict.get("answer", "No answer found."),
            "context": response_dict.get("context", ""),
            "doc_ids": response_dict.get("doc_ids", [])
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"answer": f"An error occurred: {str(e)}", "context": "", "doc_ids": []}
