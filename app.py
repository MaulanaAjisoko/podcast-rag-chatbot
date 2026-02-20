import streamlit as st
import tempfile
import os
from io import StringIO

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="Podcast RAG Chatbot",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded",
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "🎙️ Silahkan upload file transcript podcast YouTube (PDF/TXT) untuk memulai diskusi tentang kontennya."}
    ]
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "db_status" not in st.session_state:
    st.session_state.db_status = "⏳ Belum ada file transcript"
if "sources" not in st.session_state:
    st.session_state.sources = {}

with st.sidebar:
    st.header("🎙️ **Podcast Transcript**")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload file transcript podcast (PDF/TXT)", 
        type=["pdf", "txt"],
        help="Upload file transcript dari podcast YouTube favorit Anda"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        process_button = st.button("🚀 Proses", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Reset", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.messages = [st.session_state.messages[0]]
            st.session_state.db_status = "⏳ Belum ada file transcript"
            st.session_state.sources = {}
            st.rerun()
    
    st.markdown("---")
    st.info(st.session_state.db_status)
    
    if st.session_state.vectorstore:
        st.success("✅ Transcript siap untuk ditanyai")

def process_documents(uploaded_file):
    if not uploaded_file:
        st.error("Silahkan upload file transcript terlebih dahulu")
        return None

    try:
        temp_dir = "temp_podcast"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        tmp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        docs = loader.load()
        os.remove(tmp_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        embed_model = GoogleGenerativeAIEmbeddings(
            google_api_key=st.secrets['GEMINI_KEY'],
            model="gemini-embedding-001"
        )

        vectorstore = Chroma.from_documents(
            splits,
            embed_model,
            collection_name="podcast_rag_collection"
        )

        st.session_state.splits = splits
        return vectorstore
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        return None

if process_button and uploaded_file:
    with st.spinner("Memproses file transcript..."):
        data = process_documents(uploaded_file)
        if data:
            st.session_state.vectorstore = data
            st.session_state.db_status = f"✅ {uploaded_file.name} berhasil diproses!"
            st.rerun()

st.markdown("## 💬 Ngobrol tentang Podcast")

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and idx in st.session_state.sources:
            if st.button(f"📖 Lihat Sumber", key=f"source_btn_{idx}"):
                st.session_state[f"show_source_{idx}"] = True
            
            if st.session_state.get(f"show_source_{idx}", False):
                with st.expander("Bagian transcript yang menjadi referensi:", expanded=True):
                    sources = st.session_state.sources[idx]
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Bagian {i}**")
                        st.caption(f"Dari file: {source['file']}")
                        st.text(source['content'][:200] + "..." if len(source['content']) > 200 else source['content'])
                        if i < len(sources):
                            st.divider()

if prompt := st.chat_input("Tanyakan tentang isi podcast..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban di transcript..."):
            if st.session_state.vectorstore:
                docs_with_scores = st.session_state.vectorstore.similarity_search_with_relevance_scores(prompt, k=3)
                context = "\n\n".join([doc.page_content for doc, _ in docs_with_scores])
                
                llm = ChatGoogleGenerativeAI(
                    model='gemini-2.5-flash',
                    google_api_key=st.secrets['GEMINI_KEY']
                )

                template = """Anda adalah asisten AI yang ahli dalam menganalisis konten podcast.

                TUGAS:
                Jawab pertanyaan berdasarkan transcript podcast yang diberikan.

                ATURAN:
                1. Gunakan HANYA informasi dari transcript
                2. Jika informasi tidak ada, katakan "Maaf, informasi tersebut tidak ditemukan dalam transcript"
                3. JANGAN menambahkan informasi di luar transcript
                4. JANGAN mengulang pertanyaan dalam jawaban
                5. Jawab dengan bahasa Indonesia yang natural dan mudah dipahami
                6. JANGAN menambahkan opini pribadi

                TRANSCRIPT PODCAST:
                {context}

                PERTANYAAN: {question}

                JAWABAN (singkat dan langsung ke intinya):"""

                prompt_template = ChatPromptTemplate.from_template(template)
                
                chain = (
                    {
                        "context": lambda x: context,
                        "question": RunnablePassthrough()
                    }
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )
                
                response = chain.invoke(prompt)
                st.markdown(response)
                
                source_idx = len(st.session_state.messages)
                sources = []
                for doc, score in docs_with_scores:
                    sources.append({
                        "content": doc.page_content,
                        "file": doc.metadata.get("source", uploaded_file.name if uploaded_file else "Unknown"),
                        "score": score
                    })
                st.session_state.sources[source_idx] = sources
                
                if st.button("📖 Lihat Sumber", key=f"source_btn_new_{source_idx}"):
                    st.session_state[f"show_source_{source_idx}"] = True
                    st.rerun()
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                response = "Silahkan upload file transcript podcast terlebih dahulu."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})