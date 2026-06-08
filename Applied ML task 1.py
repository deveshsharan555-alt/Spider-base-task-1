import streamlit as st
from PyPDF2 import PdfReader

from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline


# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


# -----------------------------
# CHUNK TEXT
# -----------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# -----------------------------
# VECTOR STORE
# -----------------------------
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_texts(text_chunks, embedding=embeddings)


# -----------------------------
# LLM (FIXED)
# -----------------------------
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text-generation",
        model="microsoft/Phi-3-mini-4k-instruct",
        max_new_tokens=256,
        do_sample=False,
        return_full_text=False,
        device_map="auto"
    )

    return HuggingFacePipeline(pipeline=pipe)


# -----------------------------
# CONVERSATION CHAIN
# -----------------------------
def get_conversation_chain(vectorstore):
    llm = load_llm()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 4}
        ),
        memory=memory
    )

    return chain


# -----------------------------
# STREAMLIT APP
# -----------------------------
def main():

    st.set_page_config(page_title="PDF Chat", page_icon="📚")



    # -----------------------------
    # UI STYLING (ADD HERE)
    # -----------------------------
    st.markdown("""
        <style>
            .main {
                background-color: #0e1117;
                color: white;
            }

            .title {
                text-align: center;
                font-size: 40px;
                font-weight: 700;
                margin-bottom: 5px;
            }

            .subtitle {
                text-align: center;
                font-size: 16px;
                color: #a0a0a0;
                margin-bottom: 30px;
            }

            .chat-box {
                background: #1c1f26;
                padding: 15px;
                border-radius: 12px;
                margin-bottom: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">📄 PDF Chat Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Ask questions from your documents instantly</div>', unsafe_allow_html=True)




    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    

    user_question = st.text_input("Ask something from your PDFs")

    # -------------------------
    # SIDEBAR
    # -------------------------
    with st.sidebar:
        st.subheader("Upload PDFs")

        pdf_docs = st.file_uploader(
            "Upload and click PROCESS",
            accept_multiple_files=True
        )

        if st.button("PROCESS"):

            if not pdf_docs:
                st.warning("Upload PDFs first!")
                st.stop()

            with st.spinner("Processing..."):

                # Step 1: extract text
                raw_text = get_pdf_text(pdf_docs)

                # Step 2: chunk text
                text_chunks = get_text_chunks(raw_text)

                # Step 3: vector store
                vectorstore = get_vectorstore(text_chunks)

                # Step 4: conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success("Done! You can now chat.")

    # -------------------------
    # CHAT
    # -------------------------
    if user_question and st.session_state.conversation:

        
        st.markdown(f"""
        <div style="
                    text-align:right;
                    background:#2b313e;
                    padding:10px;
                    border-radius:10px;
                    margin:10px 0;
        ">
                    🙋‍♂️ {user_question}
        </div>
        """, unsafe_allow_html=True)

        response = st.session_state.conversation.invoke({"question": user_question})
        st.markdown(f"""
        <div class="chat-box">
                    🤖 <b>Answer:</b><br>{response["answer"]}
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()