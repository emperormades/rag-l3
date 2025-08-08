import os
import logging
import json
from difflib import SequenceMatcher
from dotenv import load_dotenv
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from supabase.client import create_client, Client
from deep_translator import GoogleTranslator
from langdetect import detect
import numpy as np
from typing import List, TypedDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Translation functions
def is_portuguese(text):
    try:
        language = detect(text)
        return language == 'pt'
    except:
        logger.error("Error detecting language for Portuguese check.")
        return False

def is_english(text):
    try:
        language = detect(text)
        return language == 'en'
    except:
        logger.error("Error detecting language for English check.")
        return False

def translate_to_english(text):
    try:
        if is_portuguese(text):
            translator = GoogleTranslator(source='pt', target='en')
            translated_text = translator.translate(text)
            logger.info("Translated text to English successfully.")
            return translated_text
        else:
            logger.error("Error: The input text is not in Portuguese.")
            return text
    except:
        logger.error("Error: Unable to detect the language or translate the text to English.")
        return text

def translate_to_portuguese(text):
    try:
        if is_english(text):
            translator = GoogleTranslator(source='en', target='pt')
            translated_text = translator.translate(text)
            logger.info("Translated text to Portuguese successfully.")
            return translated_text
        else:
            logger.error("Error: The input text is not in English.")
            return text
    except:
        logger.error("Error: Unable to detect the language or translate the text to Portuguese.")
        return text

# Prompt template
prompt_template_v2 = """
1. You are a helpful assistant answering questions based on provided documents.
2. The question is in English, translated from the user's input if necessary, and the documents are in English.
3. Use the following context to answer the question as accurately as possible, prioritizing exact numerical values, percentages, and key terms from the context without altering them.
4. If the context doesn't contain relevant information, state that the answer is not available in the provided document.
Context:
{context}
Question: {question}
Answer:
"""

class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    doc_ids: List[str]
    documents: List[Document]
    chat_history: List[BaseMessage]
    original_question: str
    is_portuguese: bool

def create_rag_chain(vector_store, llm):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    def retrieve_node(state: GraphState):
        question = state['question']
        docs = retriever.invoke(question)
        doc_ids = [doc.metadata.get("id") for doc in docs]
        formatted_context = "\n\n".join(doc.page_content for doc in docs)
        return {
            "documents": docs,
            "doc_ids": doc_ids,
            "context": formatted_context,
            "question": question,
            "chat_history": state.get("chat_history", []),
            "original_question": state['original_question'],
            "is_portuguese": state['is_portuguese']
        }

    def generate_node(state: GraphState):
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
        final_answer = translate_to_portuguese(answer) if state['is_portuguese'] else answer
        return {
            "answer": final_answer,
            "question": state['question'],
            "context": state['context'],
            "doc_ids": state['doc_ids'],
            "chat_history": state['chat_history'],
            "original_question": state['original_question'],
            "is_portuguese": state['is_portuguese']
        }

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()

def query_rag(rag_chain, question, chat_history):
    try:
        is_portuguese_question = is_portuguese(question)
        question_to_process = translate_to_english(question) if is_portuguese_question else question
        logger.info(f"Processing question: {question_to_process} (Original: {question})")

        response_dict = rag_chain.invoke({
            "question": question_to_process,
            "chat_history": chat_history,
            "original_question": question,
            "is_portuguese": is_portuguese_question
        })
        logger.info("Query processed successfully.")
        return {
            "answer": response_dict.get("answer", "No answer found."),
            "context": response_dict.get("context", ""),
            "doc_ids": response_dict.get("doc_ids", []),
            "original_question": response_dict.get("original_question", question)
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"answer": f"An error occurred: {str(e)}", "context": "", "doc_ids": [], "original_question": question}

def validate_environment_variables():
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "OPENAI_API_KEY"]
    for var in required_vars:
        if not os.environ.get(var):
            logger.error(f"Environment variable {var} is not set.")
            raise ValueError(f"Environment variable {var} is not set.")
    logger.info("All required environment variables are set.")

def initialize_rag_components():
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
        llm = ChatOpenAI(model="gpt-4", temperature=0.0)
        logger.info("ChatOpenAI model initialized.")
        return vector_store, llm
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

def create_evaluation_dataset():
    try:
        evaluation_dataset = [
            {
                "question_en": "What percentage of Claude.ai queries are associated with Computer and Mathematical occupations?",
                "question_pt": "Qual a porcentagem de consultas no Claude.ai associadas a ocupações de Computação e Matemática?",
                "ground_truth_en": "37.2% of Claude.ai queries are associated with Computer and Mathematical occupations.",
                "ground_truth_pt": "37,2% das consultas no Claude.ai são associadas a ocupações de Computação e Matemática."
            },
            {
                "question_en": "Which task has the highest percentage of Claude.ai conversations?",
                "question_pt": "Qual tarefa tem a maior porcentagem de conversas no Claude.ai?",
                "ground_truth_en": "Software Development and Website Maintenance accounts for 14% of Claude.ai conversations.",
                "ground_truth_pt": "Desenvolvimento de Software e Manutenção de Websites representa 14% das conversas no Claude.ai."
            },
            {
                "question_en": "What is the most prevalent skill in Claude.ai conversations?",
                "question_pt": "Qual é a habilidade mais prevalente nas conversas do Claude.ai?",
                "ground_truth_en": "Critical Thinking is the most prevalent skill, with an estimated 20% of Claude.ai conversations exhibiting it.",
                "ground_truth_pt": "Pensamento Crítico é a habilidade mais prevalente, com cerca de 20% das conversas no Claude.ai exibindo-a."
            },
            {
                "question_en": "In which wage quartile does Claude.ai usage peak?",
                "question_pt": "Em qual quartil de salário o uso do Claude.ai atinge o pico?",
                "ground_truth_en": "Claude.ai usage peaks in the upper wage quartile, with an estimated 40% of usage.",
                "ground_truth_pt": "O uso do Claude.ai atinge o pico no quartil superior de salários, com cerca de 40% do uso."
            },
            {
                "question_en": "Which Job Zone has the highest AI usage in Claude.ai conversations?",
                "question_pt": "Qual Zona de Trabalho tem o maior uso de IA nas conversas do Claude.ai?",
                "ground_truth_en": "Job Zone 4 (Considerable Preparation) has the highest AI usage, with 54.45% of Claude.ai usage.",
                "ground_truth_pt": "A Zona de Trabalho 4 (Preparação Considerável) tem o maior uso de IA, com 54,45% do uso do Claude.ai."
            },
            {
                "question_en": "What is the percentage of augmentative vs. automative conversations in Claude.ai?",
                "question_pt": "Qual é a porcentagem de conversas aumentativas versus automativas no Claude.ai?",
                "ground_truth_en": "57% of Claude.ai conversations are augmentative, and 43% are automative.",
                "ground_truth_pt": "57% das conversas no Claude.ai são aumentativas, e 43% são automativas."
            },
            {
                "question_en": "Which Claude model is preferred for coding tasks?",
                "question_pt": "Qual modelo Claude é preferido para tarefas de codificação?",
                "ground_truth_en": "Claude 3.5 Sonnet is preferred for coding and technical tasks, with an estimated 60% usage for these tasks.",
                "ground_truth_pt": "Claude 3.5 Sonnet é preferido para tarefas de codificação e técnicas, com cerca de 60% de uso para essas tarefas."
            }
        ]
        return evaluation_dataset
    except Exception as e:
        logger.error(f"Error loading golden dataset: {str(e)}")
        raise

def semantic_similarity(ground_truth, generated_answer):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        gt_embedding = embeddings.embed_query(ground_truth)
        gen_embedding = embeddings.embed_query(generated_answer)
        return np.dot(gt_embedding, gen_embedding) / (np.linalg.norm(gt_embedding) * np.linalg.norm(gen_embedding))
    except:
        logger.error("Error calculating semantic similarity.")
        return 0.0

def compare_answers(ground_truth, generated_answer):
    seq_similarity = SequenceMatcher(None, ground_truth.lower(), generated_answer.lower()).ratio()
    sem_similarity = semantic_similarity(ground_truth, generated_answer)
    return seq_similarity, sem_similarity

def main():
    try:
        vector_store, llm = initialize_rag_components()
        rag_chain = create_rag_chain(vector_store, llm)
        evaluation_dataset = create_evaluation_dataset()

        # Lists to store similarity scores for averaging
        en_seq_scores = []
        en_sem_scores = []
        pt_seq_scores = []
        pt_sem_scores = []

        for data in evaluation_dataset:
            for lang in ["en", "pt"]:
                question_key = f"question_{lang}"
                ground_truth_key = f"ground_truth_{lang}"
                question = data[question_key]
                ground_truth = data[ground_truth_key]

                response = query_rag(rag_chain, question, chat_history=[])
                generated_answer = response['answer']
                retrieved_context = response['context']
                retrieved_doc_ids = response['doc_ids']
                original_question = response['original_question']

                print("=" * 70)
                print(f"🕵️‍♂️ **Question Evaluation ({'Portuguese' if lang == 'pt' else 'English'}):** {original_question}")
                print("-" * 70)

                print("\n- **Retrieval Evaluation:**")
                print(f"Retrieved Document IDs: {retrieved_doc_ids}")
                print(f"Retrieved Context: {retrieved_context[:500]}...")
                print("\n- Evaluation Tip: Check if the retrieved context contains relevant keywords like '37.2%' or 'Computer and Mathematical'.")

                print("-" * 70)

                print("📝 **Generation Evaluation:**")
                print(f"Reference Answer: {ground_truth}\n")
                print(f"Generated Answer: {generated_answer}\n")

                seq_similarity, sem_similarity = compare_answers(ground_truth, generated_answer)
                print(f"Text Similarity (SequenceMatcher): {seq_similarity:.2f}")
                print(f"Semantic Similarity (Cosine): {sem_similarity:.2f}")

                # Store scores based on language
                if lang == "en":
                    en_seq_scores.append(seq_similarity)
                    en_sem_scores.append(sem_similarity)
                else:
                    pt_seq_scores.append(seq_similarity)
                    pt_sem_scores.append(sem_similarity)

                print("\n- Evaluation Tip: Verify if the generated answer is factually correct and includes exact numerical values from the context.")
                print("=" * 70)

        # Calculate and display average efficiency
        print("=" * 70)
        print("📊 **Model Efficiency Summary**")
        print("-" * 70)

        if en_seq_scores:
            avg_en_seq = sum(en_seq_scores) / len(en_seq_scores)
            avg_en_sem = sum(en_sem_scores) / len(en_sem_scores)
            print(f"English Questions:")
            print(f"  Average Text Similarity (SequenceMatcher): {avg_en_seq:.2f}")
            print(f"  Average Semantic Similarity (Cosine): {avg_en_sem:.2f}")
            logger.info(f"English - Avg SequenceMatcher: {avg_en_seq:.2f}, Avg Semantic: {avg_en_sem:.2f}")

        if pt_seq_scores:
            avg_pt_seq = sum(pt_seq_scores) / len(pt_seq_scores)
            avg_pt_sem = sum(pt_sem_scores) / len(pt_sem_scores)
            print(f"Portuguese Questions:")
            print(f"  Average Text Similarity (SequenceMatcher): {avg_pt_seq:.2f}")
            print(f"  Average Semantic Similarity (Cosine): {avg_pt_sem:.2f}")
            logger.info(f"Portuguese - Avg SequenceMatcher: {avg_pt_seq:.2f}, Avg Semantic: {avg_pt_sem:.2f}")

        # Overall average
        all_seq_scores = en_seq_scores + pt_seq_scores
        all_sem_scores = en_sem_scores + pt_sem_scores
        if all_seq_scores:
            avg_all_seq = sum(all_seq_scores) / len(all_seq_scores)
            avg_all_sem = sum(all_sem_scores) / len(all_sem_scores)
            print(f"\nOverall (English + Portuguese):")
            print(f"  Average Text Similarity (SequenceMatcher): {avg_all_seq:.2f}")
            print(f"  Average Semantic Similarity (Cosine): {avg_all_sem:.2f}")
            logger.info(f"Overall - Avg SequenceMatcher: {avg_all_seq:.2f}, Avg Semantic: {avg_all_sem:.2f}")

        print("=" * 70)

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")

if __name__ == "__main__":
    main()