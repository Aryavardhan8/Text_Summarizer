import streamlit as st
import re
from io import BytesIO
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Function to read .docx file
def read_docx(file):
    doc = Document(file)
    full_text = '\n'.join([para.text for para in doc.paragraphs])
    return full_text

# Sentence splitter
def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 0]

# Summarizer using TextRank
def summarize_text(text, num_sentences=3):
    sentences = split_into_sentences(text)
    if len(sentences) <= num_sentences:
        return text
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(tfidf_matrix)
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = [s for _, s in ranked[:num_sentences]]
    return '\n'.join(summary)

# Download helper
def get_download_link(text):
    buffer = BytesIO()
    buffer.write(text.encode())
    buffer.seek(0)
    return buffer

# Streamlit Interface
st.title("🧠 AI-based Text Summarizer (Docx)")
st.markdown("Upload a `.docx` file and get a clean summary.")

uploaded_file = st.file_uploader("Choose a .docx file", type=["docx"])

if uploaded_file is not None:
    raw_text = read_docx(uploaded_file)
    st.subheader("📄 Original Text")
    st.text_area("Extracted Content", raw_text, height=250)

    if st.button("Summarize"):
        summary = summarize_text(raw_text, num_sentences=3)
        st.subheader("📝 Summary")
        st.text_area("Summary Output", summary, height=200)

        # Download option
        st.download_button(
            label="📥 Download Summary as TXT",
            data=get_download_link(summary),
            file_name="summary.txt",
            mime="text/plain"
        )
