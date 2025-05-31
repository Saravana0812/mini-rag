import os
import shutil
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import ChatPromptTemplate


### Models ###
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
LLM_MODEL_NAME = "google/flan-t5-base"

# directory to save files
DOCUMENTS_DIR = "documents"

# vector db path
VECTOR_DB_PATH = "faiss_index"

# Document Loader
def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding='utf-8')
            documents.extend(loader.load())
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
    return documents

# Split documents into chunks
def split_documents(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

# Create or load FAISS vector store
def create_vector_store(chunks, embedding_model_name, vector_db_path):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    if os.path.exists(vector_db_path):
        # Delete existing FAISS index to ensure a new index is built
        shutil.rmtree(vector_db_path)
        print(f"Deleted existing FAISS index")

    print("Creating new FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(vector_db_path)
    print(f"Saved new FAISS index to {vector_db_path}")

    return vectorstore

# setup RAG qa-chain
def setup_rag_chain(vectorstore, llm_model_name, k_retrieval=4):
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    from transformers import AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt_template = """You are a helpful assistant whose primary goal is to answer questions accurately and concisely based *only* on the provided context.

If the answer cannot be found in the context, explicitly state "I don't know" or "The provided context does not contain enough information to answer this question." Do not invent information or use your prior knowledge.

Context:
{context}

Question: {question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": k_retrieval}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# Initialize RAG system
def initialize_rag_components(k_retrieval_value):
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        raise ValueError(f"No documents found. Please add .txt or .pdf files.")

    chunks = split_documents(documents, 500, 50)
    vectorstore = create_vector_store(chunks, EMBEDDING_MODEL_NAME, VECTOR_DB_PATH)
    qa_chain = setup_rag_chain(vectorstore, LLM_MODEL_NAME, k_retrieval=k_retrieval_value)
    return qa_chain

# Load vector store and build RAG (if index exists)
def load_existing_rag(k_retrieval_value):
    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError("Vector index not found. Please click the 'Update' button in the sidebar.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Load the FAISS index from local storage.
    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    qa_chain = setup_rag_chain(vectorstore, LLM_MODEL_NAME, k_retrieval=k_retrieval_value)
    return qa_chain
