from pathlib import Path
import uuid

import streamlit as st

from main import ask_question, check_document

DATA_DIR = Path("../data")
DATA_DIR.mkdir(exist_ok=True)


def get_current_pdf():
    pdfs = list(DATA_DIR.glob("*.pdf"))
    return pdfs[0] if pdfs else None


def replace_pdf(uploaded_file):
    """Replace the existing PDF with the uploaded one."""

    # Remove existing PDF
    for pdf in DATA_DIR.glob("*.pdf"):
        pdf.unlink()

    # Save uploaded PDF
    save_path = DATA_DIR / uploaded_file.name

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return save_path


st.set_page_config(
    page_title="SmartDocs AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 SmartDocs AI")
st.write("Ask questions about your documents.")

# ---------------- Session State ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ---------------- Sidebar ---------------- #

with st.sidebar:

    st.header("📄 Current Document")

    current_pdf = get_current_pdf()

    if current_pdf:
        st.success(current_pdf.name)
    else:
        st.warning("No PDF found.")

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a new PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.button("Replace Document", use_container_width=True):

            with st.spinner("Updating knowledge base..."):

                replace_pdf(uploaded_file)

                # This will rebuild only if the document changed
                check_document()

            # Start a fresh conversation
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())

            st.success("Document uploaded successfully!")

            # Refresh once so the new filename is shown
            st.rerun()

    st.divider()

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# ---------------- Chat ---------------- #

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask something...")

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = ask_question(
                prompt,
                thread_id=st.session_state.thread_id
            )

            st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )