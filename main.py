import streamlit as st
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import faiss
from dotenv import load_dotenv
import os

if "user_data" not in st.session_state:
    st.session_state.user_data = None

if "page" not in st.session_state:
    st.session_state.page = "login"

if "user_data" not in st.session_state:
    st.session_state.user_data = {}
    
# -----------------------------
# ⚙️ PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Oru Nanban",
    page_icon="logo.png",
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

if st.session_state.user_data is None:

    st.title("🧠 Oru Nanban - Welcome")

    st.subheader("👤 User Details")

    name = st.text_input("Full Name")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    dob = st.date_input("Date of Birth")

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    language = st.selectbox("Preferred Language", ["Tamil", "English", "Telugu"])

    college = st.text_input("College / School")
    hobbies = st.text_input("Hobbies")
    likes = st.text_input("Likes")
    dislikes = st.text_input("Dislikes")

    emergency_contact = st.text_input("Emergency Contact Number")

    mental_state = st.selectbox(
        "How are you feeling today?",
        ["Happy 😊", "Okay 🙂", "Sad 😔", "Very Stressed 😣"]
    )

    if st.button("Start Chat 💬"):

        if name and username and password:
            st.session_state.user_data = {
                "name": name,
                "language": language,
                "mental_state": mental_state,
                "emergency": emergency_contact
            }
            st.success("Welcome " + name)
            st.rerun()
        else:
            st.error("Please fill required fields")

    st.stop()

col1, col2 = st.columns([4,1])

with col1:
    user_input = st.text_input("", placeholder="Type your message...", key="input1")

with col2:
    send = st.button("➤")

if st.session_state.page == "login":

    st.title("🔐 Oru Nanban Login")

    option = st.selectbox("Choose", ["Login", "Signup"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Signup":
        confirm = st.text_input("Confirm Password", type="password")

    if st.button("Continue"):

        if username and password:
            st.session_state.user_data["username"] = username

            if option == "Signup":
                if password != confirm:
                    st.error("Passwords do not match")
                    st.stop()

            st.session_state.page = "verification"
            st.rerun()
        else:
            st.error("Fill all fields")

    st.stop()

if st.session_state.page == "verification":

    st.title("📄 Basic Details")

    dob = st.date_input("Date of Birth")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    if st.button("Next ➡️"):
        st.session_state.user_data["dob"] = str(dob)
        st.session_state.user_data["gender"] = gender

        st.session_state.page = "emergency"
        st.rerun()

    st.stop()

if st.session_state.page == "emergency":

    st.title("🚨 Emergency Contacts")

    name1 = st.text_input("Trusted Person 1 Name")
    phone1 = st.text_input("Phone Number 1")

    name2 = st.text_input("Trusted Person 2 Name")
    phone2 = st.text_input("Phone Number 2")

    if st.button("Finish ✅"):

        st.session_state.user_data["emergency"] = [
            {"name": name1, "phone": phone1},
            {"name": name2, "phone": phone2},
        ]

        st.session_state.page = "chat"
        st.rerun()

    st.stop()



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

    user = st.session_state.user_data  # ✅ NEW LINE

    prompt = f"""
You are ORU NANBAN, a caring emotional support AI.

User Name: {user['name']}
Preferred Language: {user['language']}
Mental State: {user['mental_state']}

- Speak like a close friend
- Support Tamil, English, Telugu, Tanglish
- Reply in user's language
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

    # 🚨 OPTIONAL SAFETY MESSAGE
    if "Very Stressed" in user["mental_state"]:
        st.warning("⚠️ If you feel overwhelmed, please contact your emergency person.")

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

if st.session_state.page == "chat":

    st.success("Welcome " + st.session_state.user_data["username"])

    # 👉 your logo
    st.image("logo.png", width=120)

    # 👉 your existing chatbot code continues here