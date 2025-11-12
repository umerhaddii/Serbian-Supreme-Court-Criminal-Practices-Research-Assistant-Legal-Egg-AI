import os
import torch
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*pydantic.*')

# Load environment variables
load_dotenv()

# -----------------------------
# Helper: Get API Keys
# -----------------------------
def get_api_key(key_name: str) -> str:
    """Get API key from Streamlit secrets or environment variables"""
    try:
        return st.secrets[key_name]
    except (KeyError, AttributeError):
        return os.getenv(key_name)

# -----------------------------
# Load API Keys
# -----------------------------
try:
    OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
    PINECONE_API_KEY = get_api_key("PINECONE_API_KEY")

    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        st.error("❌ Missing API keys. Please configure OPENAI_API_KEY and PINECONE_API_KEY in your secrets.")
        st.stop()

except Exception as e:
    st.error(f"❌ Error loading API keys: {str(e)}")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -----------------------------
# Initialize Model and Vector DB
# -----------------------------
system_prompt = """You are an expert legal assistant specializing in Serbian Supreme Court criminal practice. Your role is to provide comprehensive, practice-oriented responses that lawyers can immediately apply to their cases.

[... same long system prompt you already have ...]

Note: Remember to respond always in English not in Serbian language.
"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize Pinecone
PINECONE_ENVIRONMENT = "us-east-1"
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index_name = "criminal-practices"
index = pc.Index(index_name)

# Initialize embeddings
embedding_function = HuggingFaceEmbeddings(
    model_name="djovak/embedic-base",
    model_kwargs={'device': 'cpu'}
)

if not torch.cuda.is_available():
    print("Warning: CUDA is not available. Using CPU for embeddings.")

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_function,
    text_key='text',
    namespace="text_chunks"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# -----------------------------
# Query Refinement Prompt
# -----------------------------
refinement_template = """Create a focused Serbian search query for the RAG retriever bot. Convert to Serbian language if not already. Include key terms, synonyms, and domain-specific vocabulary. Remove filler words. Output only the refined query in the following format: {{refined_query}},{{keyterms}},{{synonyms}}

Query: {original_question}

Refined Query:"""

refinement_prompt = PromptTemplate(
    input_variables=["original_question"],
    template=refinement_template
)

refinement_chain = refinement_prompt | llm

# -----------------------------
# Retrieval Chain Construction
# -----------------------------
combined_template = f"""{system_prompt}

Please answer the following question using only the context provided:
{{context}}

Question: {{question}}
Answer:"""

retrieval_prompt = ChatPromptTemplate.from_template(combined_template)

# Create new style retrieval chain
question_answer_chain = create_stuff_documents_chain(llm, retrieval_prompt)
retrieval_chain = create_retrieval_chain(retriever=retriever, combine_documents_chain=question_answer_chain)

# -----------------------------
# Query Processing
# -----------------------------
def process_query(query: str):
    try:
        max_retries = 3

        # Step 1: Refine query
        for attempt in range(max_retries):
            try:
                refined_query_msg = refinement_chain.invoke({"original_question": query})
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    refined_query_msg = query
                    break

        if isinstance(refined_query_msg, dict):
            refined_query = refined_query_msg.get("text", "").strip()
        elif hasattr(refined_query_msg, 'content'):
            refined_query = refined_query_msg.content.strip()
        else:
            refined_query = str(refined_query_msg).strip()

        # Step 2: Run retrieval chain
        for attempt in range(max_retries):
            try:
                response_msg = retrieval_chain.invoke({"input": refined_query})
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise e

        # Step 3: Extract response
        if isinstance(response_msg, dict):
            response = response_msg.get("answer", "") or response_msg.get("output", "")
        elif hasattr(response_msg, 'content'):
            response = response_msg.content
        else:
            response = str(response_msg)

        return response

    except Exception as e:
        if "429" in str(e):
            return "⚠️ The AI service is currently experiencing high demand. Please try again in a few moments."
        else:
            return f"An error occurred: {str(e)}"

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Legal Egg AI 🥚")
st.write("Welcome to Research Assistant of Serbian Supreme Court Criminal Practices!")

with st.sidebar:
    st.header("Common Criminal Law Queries")
    example_questions = [
        "1. What are the latest Supreme Court positions on self-defense conditions?",
        "2. How does the Court interpret intent in corruption cases?",
        "3. What evidence standards apply in drug trafficking cases?",
        "4. Recent practice on plea bargaining requirements?",
        "5. Court's position on mandatory mitigation factors?",
        "6. How are aggravating circumstances evaluated in violent crimes?",
        "7. Standards for accepting circumstantial evidence?",
        "8. Requirements for extended confiscation?",
        "9. Practice on repeated offenses qualification?",
        "10. Court's interpretation of organized crime elements?"
    ]
    for q in example_questions:
        st.markdown(f"• {q}")

    st.markdown("---")
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# -----------------------------
# Chat Input
# -----------------------------
if prompt := st.chat_input("ask question..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = process_query(prompt)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
