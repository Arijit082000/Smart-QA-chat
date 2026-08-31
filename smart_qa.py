import os
import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. API Key Setup
# Set your API key in your environment variables before running the app
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

# 2. PDF load and chunking
# Ensure the PDF is in the same directory as this script, or provide the full local path
file_path = "TechVision_Company_Report.pdf" 

loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
splits = text_splitter.split_documents(docs)

# 3. Google Gemini Embeddings and FAISS Vector Store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# 4. Gemini LLM set-up
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 4.2 Format retrieved document chunks and include their source page numbers
def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        page_number = doc.metadata.get("page", "Unknown")
        if page_number != "Unknown":
            page_number = page_number + 1
        formatted_docs.append(f"[Page {page_number}]\n{doc.page_content}")
    return "\n\n".join(formatted_docs)

# 5. RAG Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are TechVision Solutions' Corporate Document Q&A Assistant.

Answer the user's question ONLY using the retrieved context
from the provided corporate document.

Rules:
1. Do not use outside knowledge.
2. Do not make assumptions.
3. Do not make up information.
4. If the answer is not available in the context, say:
   "I could not find this information in the provided document."
5. Give a clear and concise answer.
6. When possible, mention the relevant page number.

Retrieved Context:
{context}
"""
    ),
    (
        "human",
        "{input}"
    )
])

# RAG Chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 6. List of all 20 questions
PREDEFINED_QUESTIONS = [
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

# 7. Backend answer generation
def answer_question(question):
    if not question or not question.strip():
        return "Please enter a question."
    try:
        response = rag_chain.invoke(question)
        return response
    except Exception as e:
        return f"An error occurred while generating the answer: {str(e)}"

# 7.1 Retrieve source pages
def get_sources(question):
    if not question or not question.strip():
        return "No question provided."
    try:
        retrieved_docs = retriever.invoke(question)
        sources = []
        for doc in retrieved_docs:
            page = doc.metadata.get("page")
            if page is not None:
                sources.append(f"Page {page + 1}")
        unique_sources = list(dict.fromkeys(sources))
        if unique_sources:
            return "Sources: " + ", ".join(unique_sources)
        return "Source information unavailable."
    except Exception as e:
        return f"Could not retrieve sources: {str(e)}"

# 8. Gradio UI
with gr.Blocks() as interface:
    gr.Markdown(
        """
        # 🏢 TechVision Corporate Q&A Assistant

        Ask questions about the TechVision Solutions Annual Report.
        The assistant uses Retrieval-Augmented Generation (RAG) to
        retrieve relevant information from the document.
        """
    )
    
    question = gr.Textbox(
        label="Ask your question",
        placeholder="Example: What is the mission statement of TechVision Solutions?",
        lines=2
    )
    
    suggested_questions = gr.Dropdown(
        choices=PREDEFINED_QUESTIONS,
        label="Suggested Questions",
        info="You can select a suggested question or type your own."
    )
    
    ask_button = gr.Button("Ask")
    
    answer = gr.Textbox(
        label="TechVision AI Response",
        lines=8
    )
    
    sources = gr.Textbox(
        label="Source Pages",
        lines=2
    )
    
    suggested_questions.change(
        fn=lambda x: x,
        inputs=suggested_questions,
        outputs=question
    )
    
    ask_button.click(
        fn=answer_question,
        inputs=question,
        outputs=answer
    )
    
    ask_button.click(
        fn=get_sources,
        inputs=question,
        outputs=sources
    )

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
