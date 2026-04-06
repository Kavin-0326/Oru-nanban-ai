import streamlit as st
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import faiss
from dotenv import load_dotenv
import os

# -----------------------------
# ⚙️ PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Oru Nanban",
    page_icon="logo." \
    "png",
    layout="centered"
)

# -----------------------------
# 🔐 LOAD ENV
# -----------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# 📚 SAMPLE DATA
# -----------------------------
documents = [
    "I feel sad and lonely",
    "I am stressed about exams",
    "I feel very happy today",
    "I am angry at my friend",
    "I feel anxious and worried"
]

# -----------------------------
# 🧠 TF-IDF + FAISS
# -----------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents).toarray()

dimension = tfidf_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(tfidf_matrix.astype('float32'))

# -----------------------------
# 💾 MEMORY
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# 🎨 HEADER (LOGO + TITLE)
# -----------------------------
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
st.image("logo.png", width=120)
st.markdown("<h2>Oru Nanban</h2>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# 💬 INPUT UI
# -----------------------------
col1, col2 = st.columns([4,1])

with col1:
    user_input = st.text_input("", placeholder="Type your message...", key="input1")

with col2:
    send = st.button("➤")

# -----------------------------
# 🔍 SIMILARITY
# -----------------------------
def get_similar_text(query):
    query_vec = vectorizer.transform([query]).toarray().astype('float32')
    D, I = index.search(query_vec, k=1)
    return documents[I[0][0]]

# -----------------------------
# 🤖 CHAT LOGIC
# -----------------------------
if send and user_input:

    similar_text = get_similar_text(user_input)

    history_text = "\n".join(
        [f"{role}: {msg}" for role, msg in st.session_state.chat_history]
    )

    prompt = f"""
You are ORU NANBAN, a caring emotional support AI.
- Speak like a close friend
- Support Tamil, English, Telugu, Tanglish
- Reply in same language
- Be calm and supportive

Conversation History:
{history_text}

Similar Context:
{similar_text}

User: {user_input}
"""

    with st.spinner("Nanban is typing... 💭"):
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

    bot_reply = response.choices[0].message.content

    st.session_state.chat_history.append(("User", user_input))
    st.session_state.chat_history.append(("Nanban", bot_reply))

# -----------------------------
# 💬 CHAT DISPLAY (BUBBLES)
# -----------------------------
for role, msg in st.session_state.chat_history:
    if role == "User":
        st.markdown(
            f"<div style='text-align:right; background:#DCF8C6; padding:10px; border-radius:10px; margin:5px'>{msg}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='text-align:left; background:#F1F0F0; padding:10px; border-radius:10px; margin:5px'>{msg}</div>",
            unsafe_allow_html=True
        )