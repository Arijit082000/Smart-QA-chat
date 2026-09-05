import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page Config
st.set_page_config(page_title="TechVision Q&A Assistant", page_icon="🏢")

# API Key Setup from Streamlit Secrets
try:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("API Key not found! Please set GEMINI_API_KEY in Streamlit Advanced settings -> Secrets.")
    st.stop()

# Predefined Questions List
PREDEFINED_QUESTIONS = [
    "Select a suggested question...",
    "1. When was TechVision Solutions Inc. founded, and what type of services does the company primarily provide?",
    "2. How many professionals does TechVision Solutions have, and across how many continents do they work?",
    "3. How many clients has TechVision Solutions delivered projects for, and which industries has the company served?",
    "4. What is the Mission Statement of TechVision Solutions?",
    "5. What is the Vision Statement of the company, and what does it aim to become in the future?",
    "6. What are the five Core Values of TechVision Solutions?",
    "7. Which cloud platforms are included in TechVision Solutions’ Cloud Infrastructure services for deployment, migration, and management?",
    "8. What types of Data Analytics and AI/ML Solutions does TechVision Solutions provide?",
    "9. What are the key competitive advantages of TechVision Solutions, and which technology vendors does the company have strategic partnerships with?",
    "10. What are TechVision Solutions’ customer retention rate and Net Promoter Score?",
    "11. How many projects has TechVision Solutions successfully delivered, and what is its average project success rate?",
    "12. What type of organizational structure does TechVision Solutions operate under, and what are the main objectives of this structure?",
    "13. Who are the CEO, CTO, and CFO of the company, and how many years of professional experience does each have?",
    "14. How did TechVision Solutions’ revenue change between 2020 and 2024?",
    "15. What were the company’s EBITDA margins in 2020 and 2024, and what were the main reasons for this improvement?",
    "16. What were TechVision Solutions’ revenue, EBITDA margin, and employee count in 2024?",
    "17. What problem did TechVision solve in the Global Manufacturing Transformation case study, and what measurable improvements resulted from the AWS migration?",
    "18. How did TechVision’s solution in the Healthcare Data Analytics case study help reduce patient readmissions and operational costs?",
    "19. What are the major strategic initiatives of TechVision Solutions for 2024–2026?",
    "20. Which emerging technologies is TechVision Solutions investing in for the future, and what is the purpose of its AI Center of Excellence?"
]

# Initialize and Cache RAG Pipeline
@st.cache_resource(show_spinner="Setting up the AI model and reading the document...")
def setup_rag_pipeline():
    # PDF is in the same directory
    file_path = "TechVision_Company_Report.pdf"
    
    # 1. Load and split
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # 2. Embeddings & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 3. LLM Setup
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # 4. Format Docs helper
    def format_docs(docs):
        formatted_docs = []
        for doc in docs:
            page_number = doc.metadata.get("page", "Unknown")
            if page_number != "Unknown":
                page_number = page_number + 1
            formatted_docs.append(f"[Page {page_number}]\n{doc.page_content}")
        return "\n\n".join(formatted_docs)

    # 5. Prompt & Chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are TechVision Solutions' Corporate Document Q&A Assistant.
Answer the user's question ONLY using the retrieved context from the provided corporate document.
Rules:
1. Do not use outside knowledge.
2. Do not make assumptions.
3. Do not make up information.
4. If the answer is not available in the context, say: "I could not find this information in the provided document."
5. Give a clear and concise answer.
6. When possible, mention the relevant page number.

Retrieved Context:
{context}"""),
        ("human", "{input}")
    ])
    
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

rag_chain, retriever = setup_rag_pipeline()

# ----------------- UI Design -----------------

st.title("🏢 TechVision Corporate Q&A Assistant")
st.markdown("""
Ask questions about the TechVision Solutions Annual Report. 
The assistant uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the document.
""")

# Dropdown for predefined questions
selected_q = st.selectbox("Suggested Questions", PREDEFINED_QUESTIONS)

# Textbox for user input
default_text = "" if selected_q == "Select a suggested question..." else selected_q
user_question = st.text_input("Ask your question", value=default_text, placeholder="Example: What is the mission statement of TechVision Solutions?")

# Ask Button
if st.button("Ask", type="primary"):
    if not user_question or not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                # Get Answer
                response = rag_chain.invoke(user_question)
                
                # Get Sources
                retrieved_docs = retriever.invoke(user_question)
                sources = []
                for doc in retrieved_docs:
                    page = doc.metadata.get("page")
                    if page is not None:
                        sources.append(f"Page {page + 1}")
                        
                unique_sources = list(dict.fromkeys(sources))
                if unique_sources:
                    source_text = "Sources: " + ", ".join(unique_sources)
                else:
                    source_text = "Source information unavailable."
                
                # Display Results
                st.markdown("### TechVision AI Response")
                st.info(response)
                st.caption(f"*{source_text}*")
                
            except Exception as e:
                st.error(f"An error occurred while generating the answer: {str(e)}")
    
