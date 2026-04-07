import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

# 1. SETUP THE WEBSITE PAGE
st.set_page_config(page_title="Guru Chat", layout="wide")
st.title("📚 Guru Chat: Your AI Study Buddy")
st.write("Upload a PDF and ask me anything about it!")

# 2. SIDEBAR FOR API KEY AND UPLOAD
with st.sidebar:
    st.title("Settings")
    user_api_key = st.text_input("Enter Google Gemini API Key:", type="password")
    pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
    
    if st.button("Train Guru Chat"):
        if not user_api_key:
            st.error("Please enter your API Key!")
        elif not pdf_docs:
            st.error("Please upload at least one PDF!")
        else:
            with st.spinner("Reading and Indexing..."):
                # A. Extract Text
                raw_text = ""
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        raw_text += page.extract_text()
                
                # B. Chunk Text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_text(raw_text)
                
                # C. Create Vector Database
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=user_api_key)
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                vector_store.save_local("faiss_index")
                st.success("Guru Chat is now ready!")

# 3. CHAT LOGIC (THE RAG PROCESS)
user_question = st.text_input("Ask a question from your notes:")

if user_question:
    if not user_api_key:
        st.warning("Enter API Key in the sidebar.")
    else:
        # Load the Database
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=user_api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Search for relevant parts of the PDF
        docs = new_db.similarity_search(user_question)
        
        # Set the AI's "Personality"
        prompt_template = """
        You are a helpful study buddy. Answer the question based ONLY on the provided notes.
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=user_api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # Generate the Answer
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        st.info(response["output_text"])
