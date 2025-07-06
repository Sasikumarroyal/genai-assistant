# genai-assistant
A GenAI-powered assistant that summarizes uploaded documents and answers questions based on content.
import streamlit as st
from transformers import pipeline
from pdfminer.high_level import extract_text

# Load models
summarizer = pipeline("summarization")
qa_model = pipeline("question-answering")

st.title("📄 GenAI Research Assistant")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    # Extract text
    if uploaded_file.type == "application/pdf":
        text = extract_text(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")

    st.success("✅ File uploaded successfully!")

    # Limit long text for summarization
    short_text = text[:2000] if len(text) > 2000 else text

    # Summarize
    summary = summarizer(short_text, max_length=150, min_length=50, do_sample=False)
    st.subheader("📝 Auto Summary:")
    st.write(summary[0]['summary_text'])

    # Ask Anything mode
    st.subheader("❓ Ask Anything:")
    question = st.text_input("Type your question about the document:")
    if question:
        answer = qa_model(question=question, context=text)
        st.write(f"**Answer:** {answer['answer']}")
        st.caption(f"Confidence: {round(answer['score'] * 100, 2)}%")
