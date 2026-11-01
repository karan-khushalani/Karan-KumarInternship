import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faqs = {
    "What is CodeAlpha?": "CodeAlpha is a tech internship platform.",
    "How long is internship?": "1 month with 4 tasks.",
    "Is it free?": "Yes, free with certificate on completion of 2 tasks.",
    "What is AI?": "Artificial Intelligence is making machines think like humans."
}
questions = list(faqs.keys())
answers = list(faqs.values())
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(questions)
st.title("🤖 FAQ Chatbot - CodeAlpha")
query = st.text_input("Ask your question:")
if query:
    q_vec = vectorizer.transform([query])
    sim = cosine_similarity(q_vec, tfidf)
    best_idx = sim.argmax()
    st.write(f"**Bot:** {answers[best_idx]}")
