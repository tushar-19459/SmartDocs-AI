import os
import json
import shutil
import streamlit as st

from ingest import ingest
from chat import ask_question

# ---------------------------------------------------
# Config
# ---------------------------------------------------

UPLOAD_FOLDER = "uploads"
PROFILE_PATH = "../knowledge_base/customer_support_profile.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.set_page_config(
    page_title="Intelligent Document QA",
    page_icon="📄",
    layout="wide"
)

# ---------------------------------------------------
# Session State
# ---------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:

    st.title("📚 Knowledge Base")

    st.markdown("---")

    if os.path.exists(PROFILE_PATH):

        with open(PROFILE_PATH, "r", encoding="utf8") as f:
            profile = json.load(f)

        st.success("Knowledge Base Loaded")

        st.write("### Title")
        st.write(profile["title"])

        st.write("### Domain")
        st.write(profile["domain"])

        st.write("### Topics")

        for topic in profile["topics"]:
            st.write(f"• {topic}")

    else:

        st.warning("No Knowledge Base Found")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------
# Main Page
# ---------------------------------------------------

st.title("📄 Intelligent Document Question Answering")

st.write(
    "Upload a PDF and ask questions about the document."
)

st.markdown("---")

# ---------------------------------------------------
# Upload PDF
# ---------------------------------------------------

uploaded_file = st.file_uploader(

    "Upload PDF",

    type=["pdf"]
)

if uploaded_file is not None:

    save_path = os.path.join(
        UPLOAD_FOLDER,
        uploaded_file.name
    )

    with open(save_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)

    st.success("PDF Uploaded")

    if st.button("Create Knowledge Base"):

        with st.spinner("Creating Knowledge Base..."):

            ingest(save_path)

        st.success("Knowledge Base Created!")

        st.info("Refresh the page to load the profile.")

st.markdown("---")

# ---------------------------------------------------
# Chat History
# ---------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ---------------------------------------------------
# Chat Input
# ---------------------------------------------------

question = st.chat_input("Ask a question...")

if question:

    # -----------------------------
    # Display User Message
    # -----------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # -----------------------------
    # Assistant
    # -----------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching Knowledge Base..."):

            answer, queries, sources = ask_question(question)

        st.markdown(answer)

        # Save Answer

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # -----------------------------
        # Rewritten Queries
        # -----------------------------

        with st.expander("🔍 Rewritten Queries"):

            for q in queries:

                st.write(f"• {q}")

        # -----------------------------
        # Sources
        # -----------------------------

        with st.expander("📄 Sources Used"):

            for i, chunk in enumerate(sources, start=1):

                st.markdown(f"### Source {i}")

                st.write(
                    f"**Page:** {chunk['metadata']['page']}"
                )

                if "score" in chunk:
                    st.write(
                        f"**Score:** {chunk['score']:.4f}"
                    )

                st.text(
                    chunk["document"][:700] + "..."
                )

                st.divider()