import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =========================
#  PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Security Chatbot", page_icon="🔐", layout="centered")

# =========================
#  CUSTOM UI (CHAT STYLE)
# =========================
st.markdown("""
<style>
.chat-user {
    background-color: #1f77b4;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
.chat-bot {
    background-color: #2e2e2e;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🔐 AI Security Chatbot")
st.write("Upload a PDF and chat with it")

# =========================
#  FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")

# =========================
#  CACHE PDF PROCESSING
# =========================
@st.cache_resource
def process_pdf(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(docs, embeddings)
    return db

# =========================
#  MODEL CACHE
# =========================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

# =========================
#  CHAT HISTORY
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
#  MAIN LOGIC
# =========================
if uploaded_file:
    db = process_pdf(uploaded_file)
    tokenizer, model = load_model()

    query = st.chat_input("Ask something about your PDF...")

    if query:
        # Save user message
        st.session_state.messages.append(("user", query))

        results = db.similarity_search(query, k=2)
        context = "\n\n".join([doc.page_content for doc in results])[:800]

        prompt = f"""
You are a cybersecurity expert.

Answer clearly in 2-3 sentences.

Context:
{context}

Question: {query}

Answer:
"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.2,
            do_sample=False,
            repetition_penalty=1.2
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()

        # Save bot response
        st.session_state.messages.append(("bot", response))

    # =========================
    #  DISPLAY CHAT
    # =========================
    for role, msg in st.session_state.messages:
        if role == "user":
            st.markdown(f"<div class='chat-user'>👤 {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'>🤖 {msg}</div>", unsafe_allow_html=True)

    # =========================
    #  SHOW SOURCES (LAST MESSAGE)
    # =========================
    if query:
        st.markdown("### 📚 Sources:")
        for i, doc in enumerate(results):
            with st.expander(f"📄 Source {i+1}"):
                st.write(doc.page_content)