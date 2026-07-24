import time
import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


# ----------------------------
# Load Secrets
# ----------------------------
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="Haroon Chatbot",
    page_icon="🤖"
)

st.title("🤖 Haroon Chatbot")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return ChatGroq(
        model="llama-3.3-70b-versatile"
    )

model = load_model()

# ----------------------------
# Initialize Chat History
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(
            content="""you are a angry assistant always response in rude behaviour and in one line.
            1. Always reply in the same language as the user's message.
2. If the user writes in Urdu, reply in Urdu.
3. If the user writes in English, reply in English.
4. If the user writes in Roman Urdu, reply in Roman Urdu.
5. Keep replies short (1-2 lines).
6. always roast user no mercy."""
        )
    ]

# Messages shown on UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Display Previous Messages
# ----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------
# Chat Input
# ----------------------------
user_input = st.chat_input("Type your message...")

if user_input:

    # Show User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    st.session_state.chat_history.append(
        HumanMessage(content=user_input)
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # ----------------------------
    # Assistant Response
    # ----------------------------
    with st.chat_message("assistant"):

        placeholder = st.empty()

        # Loading text
        placeholder.markdown("🤖 *Thinking...*")

        # Generate response
        response = model.invoke(st.session_state.chat_history)

        # Typing animation
        text = ""

        for ch in response.content:
            text += ch
            placeholder.markdown(text + "▌")
            time.sleep(0.01)

        placeholder.markdown(text)

    # Save AI Response
    st.session_state.chat_history.append(
        AIMessage(content=response.content)
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.content
        }
    )
