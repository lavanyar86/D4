import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# 1. Load keys from .env
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Multi-PDF RAG App", page_icon="📚")
st.title("📚 Multi-PDF RAG with Sources")

DB_DIR = "faiss_index"

# 2. Local HuggingFace Embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

if not os.getenv("OPENAI_API_KEY"):
    st.error("Missing `OPENAI_API_KEY`! Please check your .env file.")
    st.stop()

# 3. Multi-PDF uploader
uploaded_files = st.file_uploader(
    "Upload multiple PDF documents at once",
    type="pdf",
    accept_multiple_files=True
)

# 4. Process and Index into FAISS
if uploaded_files:
    file_names = sorted([f.name for f in uploaded_files])
    
    if "indexed_files" not in st.session_state or st.session_state.indexed_files != file_names:
        with st.spinner("Chunking and indexing PDFs into local FAISS..."):
            all_chunks = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                try:
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()

                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name
                        if "page" in doc.metadata:
                            doc.metadata["page"] = doc.metadata["page"] + 1

                    chunks = text_splitter.split_documents(docs)
                    all_chunks.extend(chunks)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            if all_chunks:
                vectorstore = FAISS.from_documents(all_chunks, embeddings)
                vectorstore.save_local(DB_DIR)

                st.session_state.store = vectorstore
                st.session_state.indexed_files = file_names
                st.session_state.chunk_count = len(all_chunks)

        st.success(f"Indexed {st.session_state.chunk_count} chunks from {len(file_names)} document(s).")
    else:
        st.info(f"Indexed {st.session_state.chunk_count} chunks available for search.")

# 5. Question Answering
if "store" in st.session_state:
    st.markdown("---")
    question = st.text_input("Ask a question about your uploaded documents:")

    if question:
        retriever = st.session_state.store.as_retriever(search_kwargs={"k": 4})
        
        with st.spinner("Formulating answer..."):
            matched_docs = retriever.invoke(question)

            template = """You are a helpful assistant answering questions using only the context below.
If multiple documents disagree, explicitly point out the disagreement.
If you do not know the answer based on the context, state that clearly.

Context:
{context}

Question:
{question}

Answer:"""
            prompt = ChatPromptTemplate.from_template(template)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            def format_docs(docs):
                return "\n\n".join(
                    f"[{d.metadata.get('source', 'Unknown')} - Page {d.metadata.get('page', '?')}]:\n{d.page_content}"
                    for d in docs
                )

            rag_chain = (
                {"context": lambda _: format_docs(matched_docs), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            answer = rag_chain.invoke(question)

        # 6. Output Answer & Sources
        st.markdown("### Answer")
        st.write(answer)

        sources = []
        for d in matched_docs:
            src_file = d.metadata.get("source", "Unknown")
            page_num = d.metadata.get("page", "?")
            citation = f"{src_file} - page {page_num}"
            if citation not in sources:
                sources.append(citation)

        if sources:
            st.markdown(f"**Sources:** {' | '.join(sources)}")

        with st.expander("View Retrieved Context Chunks"):
            for i, d in enumerate(matched_docs):
                st.caption(f"Chunk {i+1}: {d.metadata.get('source')} (Page {d.metadata.get('page')})")
                st.text(d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""))