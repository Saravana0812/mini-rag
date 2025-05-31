import os
import streamlit as st
from dotenv import load_dotenv
from rag_system import initialize_rag_components, load_existing_rag, DOCUMENTS_DIR

load_dotenv()

# Streamlit interface
def streamlit_interface():
    st.sidebar.title("Mini RAG System")

    k_retrieval_value = st.sidebar.slider(
        "Number of Retrieved Chunks (k)",
        min_value=1,
        max_value=10,
        value=4,
        step=1
    )

    uploaded_files = st.sidebar.file_uploader("Upload a new document (.txt or .pdf)", type=["txt", "pdf"],accept_multiple_files=True)
    if uploaded_files:
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"Uploaded {uploaded_file.name} to documents folder.")

    if st.sidebar.button("Update"):
        st.session_state.qa_chain = initialize_rag_components(k_retrieval_value)
        st.session_state.k_retrieval_cached = k_retrieval_value
        st.success("Updates complete!")

    if "qa_chain" not in st.session_state:
        try:
            st.session_state.qa_chain = load_existing_rag(k_retrieval_value)
            st.session_state.k_retrieval_cached = k_retrieval_value
        except Exception as e:
            st.warning("Please upload your file and update using the update button.")
            st.stop()

    if st.session_state.get("k_retrieval_cached") != k_retrieval_value:
        st.session_state.qa_chain = load_existing_rag(k_retrieval_value)
        st.session_state.k_retrieval_cached = k_retrieval_value

    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask your question:")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Generating answer..."):
            response = st.session_state.qa_chain.invoke({"query": query})
            generated_answer = response["result"]
            retrieved_chunks = response["source_documents"]

            with st.chat_message("assistant"):
                st.markdown(generated_answer)
                with st.expander("Retrieved Chunks (for transparency)"):
                    for i, doc in enumerate(retrieved_chunks):
                        st.write(f"**Chunk {i+1}:**")
                        st.info(doc.page_content)
                        if doc.metadata:
                            st.write(f"Source: {doc.metadata.get('source', 'N/A')}")
                        st.markdown("---")
            st.session_state.messages.append({"role": "assistant", "content": generated_answer})

if __name__ == "__main__":
    streamlit_interface()
