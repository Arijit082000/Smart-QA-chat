#  TechVision Corporate Q&A Assistant (RAG Pipeline)

##  Overview
This project is an end-to-end Retrieval-Augmented Generation (RAG) application designed to extract and answer queries from corporate documents. It acts as an intelligent Q&A assistant for the TechVision Solutions Annual Report, providing precise answers alongside the exact source page numbers.

** Live Demo:** ( https://smart-app-chat-xuvjm2xwrx784csrebves6.streamlit.app/#tech-vision-corporate-q-and-a-assistant )

##  Features
* **Document Parsing:** Efficiently loads and chunks complex PDF documents using PyPDFLoader and RecursiveCharacterTextSplitter.
* **Semantic Search:** Utilizes Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and FAISS vector database for fast, context-aware information retrieval.
* **Generative AI:** Powered by Google's Gemini 2.5 Flash LLM to synthesize natural language answers based strictly on retrieved context.
* **Source Tracking:** Prevents hallucinations by grounding every answer in the document and citing the specific page number.
* **Interactive UI:** A clean, user-friendly web interface built entirely with Streamlit.

##  Tech Stack
* **Language:** Python
* **Framework:** Streamlit
* **AI & NLP:** LangChain, Google Gemini API, Hugging Face
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Data Processing:** PyPDF

##  How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/arijit082000/Smart-QA-chat.git](https://github.com/arijit082000/Smart-QA-chat.git)

   Install the required dependencies:
   pip install -r requirements.txt

   Add your Gemini API Key in the secrets.toml file (for Streamlit) or as an environment variable.

   Run the Streamlit app:
   streamlit run smart_qa.py
   
   Author:
   Arijit Dasgupta
   Data Science & AI Enthusiast
   Specializing in Machine Learning, NLP, and Data Analytics.
   
   
