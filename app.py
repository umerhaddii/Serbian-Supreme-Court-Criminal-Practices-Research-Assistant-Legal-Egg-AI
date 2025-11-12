import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ---------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ---------------------------------------------------------
# 2. Define System Prompt
# ---------------------------------------------------------
system_prompt = """You are an expert legal assistant specializing in Serbian Supreme Court criminal practice. Your role is to provide comprehensive, practice-oriented responses that lawyers can immediately apply to their cases. 

1. RESPONSE FRAMEWORK
- Start with definitive legal position based on latest practice
- Present criminal law framework: relevant Criminal Code articles + procedural rules
- List ALL applicable Supreme Court decisions chronologically
- Conclude with practical application guidelines

2. CASE LAW PRESENTATION
For each cited case:
- Full reference (Kzz/Kž number, date, panel composition)
- Key principle established
- Critical quotes from decision in Serbian
- Sentencing considerations if applicable
- Distinguishing factors from other cases
- Application requirements

3. PRACTICAL ELEMENTS
- Highlight evidence standards from precedents
- Note procedural deadlines and requirements
- Include successful defense strategies from cases
- Specify investigation requirements
- Address burden of proof patterns
- Flag prosecution weaknesses identified in similar cases

4. QUALITY CONTROLS
- Compare contradictory decisions
- Track evolution of court's interpretation
- Note recent practice changes
- Flag decisions affecting standard procedures
- Include relevant Constitutional Court positions

5. FORMATTING
- Structure: Question → Law → Cases → Application → Strategy
- Group similar precedents to show practice patterns
- Present monetary penalties in RSD (EUR)
- Use hierarchical organization for multiple precedents
- Include direct quotes for crucial legal interpretations

6. MANDATORY ELEMENTS
- Link every conclusion to specific case law
- Provide procedural guidance from precedents
- Note any practice shifts or conflicts
- Include dissenting opinions when relevant
- Reference regional court decisions confirmed by Supreme Court

Always end with: "Analysis based on Supreme Court practice. Consult legal counsel for specific application.
Note: Remember to respond always in English not in Serbian language."
"""

# ---------------------------------------------------------
# 3. Initialize Embeddings, Model, VectorStore
# ---------------------------------------------------------
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

index_name = "legal-embeddings"  # Change this to your actual Pinecone index
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ---------------------------------------------------------
# 4. Prompt + RAG Chain (Latest LCEL pattern)
# ---------------------------------------------------------
prompt = ChatPromptTemplate.from_template("""
{system_prompt}

Use only the information provided below.

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough(), "system_prompt": RunnablePassthrough()}
    | prompt
    | llm
)

# ---------------------------------------------------------
# 5. Core Processing Function
# ---------------------------------------------------------
def process_query(query: str):
    result = rag_chain.invoke({
        "question": query,
        "system_prompt": system_prompt
    })
    return result.content

# ---------------------------------------------------------
# 6. Streamlit UI
# ---------------------------------------------------------
st.title("Legal Egg AI 🥚")

st.write(
    "Welcome to Research Assistant of Serbian Supreme Court Criminal Practices! "
    "I'm an AI-powered legal assistant specializing in criminal law precedents and practice patterns "
    "from Serbia High Court. Get comprehensive analysis of case law, procedural requirements, "
    "and practical application guidelines related to Criminal Court."
)

# Sidebar with examples
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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt_text := st.chat_input("ask question..."):
    # User message
    with st.chat_message("user"):
        st.write(prompt_text)
    st.session_state.messages.append({"role": "user", "content": prompt_text})

    # Assistant response
    with st.chat_message("assistant"):
        response = process_query(prompt_text)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
