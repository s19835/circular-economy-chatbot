import streamlit as st
from qa_advanced import answer_questions

st.title("Circular Economy QA Chatbot 🌍")
query = st.text_input("Ask a question:")

if query:
    answer = answer_questions(query)
    st.write("**Answer:**", answer)
